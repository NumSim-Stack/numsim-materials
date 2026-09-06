#ifndef LINEAR_ISOTROPIC_HARDENING_H
#define LINEAR_ISOTROPIC_HARDENING_H

#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Linear isotropic hardening: H(κ) = K * κ
///
/// Outputs:
///   "hardening_stress"    — scalar: K * κ
///   "hardening_modulus"   — scalar: dH/dκ = K (constant)
///
/// Inputs:
///   source::equivalent_plastic_strain — scalar κ from plasticity material
template <typename Traits>
class linear_isotropic_hardening final
    : public material_base<linear_isotropic_hardening<Traits>, Traits> {
public:
  using base = material_base<linear_isotropic_hardening<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit linear_isotropic_hardening(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_H(base::template add_output<value_type>(
            "hardening_stress", &linear_isotropic_hardening::compute)),
        m_dH(base::template add_output<value_type>("hardening_modulus")),
        m_K(base::template get_parameter<value_type>("K")),
        m_source(base::template get_parameter<std::string>("source")),
        m_kappa(base::template add_input<value_type>(
            m_source, "equivalent_plastic_strain", EdgeKind::Local))
  {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("source").template add<is_required>();
    para.template insert<value_type>("K").template add<is_required>();
    return para;
  }

  void compute() {
    const auto kappa = m_kappa.get();
    m_H = m_K * kappa;
    m_dH = m_K;
  }

private:
  value_type& m_H;
  value_type& m_dH;
  const value_type& m_K;
  const std::string& m_source;
  const input_property<value_type, property_traits>& m_kappa;
};

} // namespace numsim::materials

#endif // LINEAR_ISOTROPIC_HARDENING_H
