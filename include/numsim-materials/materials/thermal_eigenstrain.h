#ifndef NUMSIM_MATERIALS_THERMAL_EIGENSTRAIN_H
#define NUMSIM_MATERIALS_THERMAL_EIGENSTRAIN_H

#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

/// Isotropic thermal eigenstrain eps_th = alpha (T - T_ref) I with the expansion coefficient a produced
/// property ("alpha_source": a constant_scalar's "alpha::value" or a cure-dependent producer) and the
/// temperature a history source ("temperature_source", "state"). Output "eigenstrain".
template <typename Traits>
class thermal_eigenstrain final : public material_base<thermal_eigenstrain<Traits>, Traits> {
public:
  using base = material_base<thermal_eigenstrain<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;

  template <typename... Args>
  explicit thermal_eigenstrain(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_eigenstrain(base::template add_output<tensor2>("eigenstrain")),
        m_alpha_name(connection_source::parse(base::template get_parameter<std::string>("alpha_source"))),
        m_alpha(base::template add_input<value_type>(m_alpha_name.material, m_alpha_name.property, EdgeKind::Global)),
        m_T_ref(base::template get_parameter<value_type>("reference_temperature")),
        m_T(base::template add_input_history<value_type>(
            connection_source{base::template get_parameter<std::string>("temperature_source"), "state"}, EdgeKind::Global)) {
    base::add_callback(&thermal_eigenstrain::compute, m_eigenstrain).reads(m_alpha, m_T).partials(m_eigenstrain, m_alpha, m_T);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "alpha_source");
    optional<value_type>(para, "reference_temperature", value_type{0});
    optional<std::string>(para, "temperature_source", std::string{"temperature"});
    return para;
  }

  void compute() {
    m_eigenstrain = m_alpha.get() * (m_T.new_value() - m_T_ref) * tmech::eye<value_type, Dim, 2>();
    if (base::want_partials()) {
      base::partial(m_eigenstrain, m_alpha) = (m_T.new_value() - m_T_ref) * tmech::eye<value_type, Dim, 2>();
      base::partial(m_eigenstrain, m_T) = m_alpha.get() * tmech::eye<value_type, Dim, 2>();
    }
  }

private:
  tensor2 &m_eigenstrain;
  connection_source const m_alpha_name;
  input_property<value_type, property_traits> const &m_alpha;
  value_type const &m_T_ref;
  input_history<value_type, property_traits> const &m_T;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_THERMAL_EIGENSTRAIN_H
