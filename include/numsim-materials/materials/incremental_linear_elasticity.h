#ifndef NUMSIM_MATERIALS_INCREMENTAL_LINEAR_ELASTICITY_H
#define NUMSIM_MATERIALS_INCREMENTAL_LINEAR_ELASTICITY_H

#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

/// Hypoelastic update sigma_{n+1} = sigma_n + C : (eps_{n+1} - eps_n) with the stress as history
/// and the tangent read from another material, so a path-dependent tangent (cure, damage,
/// linear-comparison composites) integrates incrementally. Outputs "stress" (history), "tangent".
///
/// Parameters: "strain_source", "strain_property", "tangent_source", "tangent_property".
template <typename Traits>
class incremental_linear_elasticity final : public material_base<incremental_linear_elasticity<Traits>, Traits> {
public:
  using base = material_base<incremental_linear_elasticity<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit incremental_linear_elasticity(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_stress(base::template add_history_output<tensor2>("stress")),
        m_tangent(base::template add_output<tensor4>("tangent")),
        m_strain(base::template add_input_history<tensor2>(
            connection_source{base::template get_parameter<std::string>("strain_source"), base::template get_parameter<std::string>("strain_property")},
            EdgeKind::Global)),
        m_C(base::template add_input<tensor4>(base::template get_parameter<std::string>("tangent_source"),
                                              base::template get_parameter<std::string>("tangent_property"), EdgeKind::Global)) {
    base::add_callback(&incremental_linear_elasticity::copy_tangent, m_tangent).reads(m_C);
    base::add_callback(&incremental_linear_elasticity::update, m_stress).reads(m_strain, m_tangent);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "strain_source");
    optional<std::string>(para, "strain_property", std::string{"strain"});
    required<std::string>(para, "tangent_source");
    optional<std::string>(para, "tangent_property", std::string{"tangent"});
    return para;
  }

  void copy_tangent() noexcept { m_tangent = m_C.get(); }

  void update() noexcept {
    m_stress.new_value() = m_stress.old_value() + tmech::dcontract(m_tangent, tensor2{m_strain.new_value() - m_strain.old_value()});
  }

private:
  history_property<tensor2> &m_stress;
  tensor4 &m_tangent;
  input_history<tensor2, property_traits> const &m_strain;
  input_property<tensor4, property_traits> const &m_C;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_INCREMENTAL_LINEAR_ELASTICITY_H
