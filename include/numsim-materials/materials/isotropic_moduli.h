#ifndef NUMSIM_MATERIALS_ISOTROPIC_MODULI_H
#define NUMSIM_MATERIALS_ISOTROPIC_MODULI_H

#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/isotropize.h"
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

/// The moduli of an isotropic tangent as scalar outputs "K", "G", "poisson_ratio", so that consumers
/// of nu (Eshelby nodes) follow a tangent that evolves. A tangent off its isotropic projection by more
/// than "tolerance" (relative) is rejected; a non-positive tolerance projects without the check (the
/// effective medium of a self-consistent scheme, an algorithmic tangent). Parameters: "tangent_source",
/// "tangent_property", "tolerance".
template <typename Traits>
class isotropic_moduli final : public material_base<isotropic_moduli<Traits>, Traits> {
public:
  using base = material_base<isotropic_moduli<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit isotropic_moduli(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_K(base::template add_output<value_type>("K")),
        m_G(base::template add_output<value_type>("G")),
        m_nu(base::template add_output<value_type>("poisson_ratio")),
        m_tolerance(base::template get_parameter<value_type>("tolerance")),
        m_C(base::template add_input<tensor4>(base::template get_parameter<std::string>("tangent_source"),
                                              base::template get_parameter<std::string>("tangent_property"), EdgeKind::Global)) {
    base::add_callback(&isotropic_moduli::compute, m_K, m_G, m_nu).reads_all_inputs();
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "tangent_source");
    optional<std::string>(para, "tangent_property", std::string{"tangent"});
    optional<value_type>(para, "tolerance", value_type{1e-8});
    return para;
  }

  /// One callback for the three outputs.
  void compute() {
    auto const [K, G]{m_tolerance > 0 ? require_isotropic(m_C.get(), base::name(), m_tolerance) : isotropize(m_C.get())};
    m_K = K;
    m_G = G;
    m_nu = poisson_ratio(K, G);
  }

private:
  value_type &m_K;
  value_type &m_G;
  value_type &m_nu;
  value_type const &m_tolerance;
  input_property<tensor4, property_traits> const &m_C;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_ISOTROPIC_MODULI_H
