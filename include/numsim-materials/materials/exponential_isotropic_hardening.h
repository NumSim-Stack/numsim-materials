#ifndef NUMSIM_MATERIALS_EXPONENTIAL_ISOTROPIC_HARDENING_H
#define NUMSIM_MATERIALS_EXPONENTIAL_ISOTROPIC_HARDENING_H

#include <cmath>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Exponential saturation hardening: H(α) = K_inf * (1 - exp(-delta * α))
///
/// Outputs:
///   "hardening_stress"    — scalar: K_inf * (1 - exp(-delta * α))
///   "hardening_modulus"   — scalar: dH/dα = K_inf * delta * exp(-delta * α)
///
/// Inputs:
///   source::equivalent_plastic_strain — scalar α from plasticity material
template <typename Traits>
class exponential_isotropic_hardening final
    : public material_base<exponential_isotropic_hardening<Traits>, Traits> {
public:
  using base = material_base<exponential_isotropic_hardening<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit exponential_isotropic_hardening(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_H(base::template add_output<value_type>(
            "hardening_stress", &exponential_isotropic_hardening::compute)),
        m_dH(base::template add_output<value_type>("hardening_modulus")),
        m_K_inf(base::template get_parameter<value_type>("K_inf")),
        m_delta(base::template get_parameter<value_type>("delta")),
        m_source(base::template get_parameter<std::string>("source")),
        m_alpha(base::template add_input<value_type>(
            m_source, "equivalent_plastic_strain", EdgeKind::Local))
  {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("source").template add<is_required>();
    para.template insert<value_type>("K_inf").template add<is_required>();
    para.template insert<value_type>("delta").template add<is_required>();
    return para;
  }

  void compute() {
    const auto alpha = m_alpha.get();
    const auto exp_term = std::exp(-m_delta * alpha);
    m_H = m_K_inf * (value_type{1} - exp_term);
    m_dH = m_K_inf * m_delta * exp_term;
  }

private:
  value_type& m_H;
  value_type& m_dH;
  const value_type& m_K_inf;
  const value_type& m_delta;
  const std::string& m_source;
  const input_property<value_type, property_traits>& m_alpha;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_EXPONENTIAL_ISOTROPIC_HARDENING_H
