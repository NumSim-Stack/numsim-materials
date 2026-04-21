#ifndef NUMSIM_MATERIALS_EXPONENTIAL_DAMAGE_LAW_H
#define NUMSIM_MATERIALS_EXPONENTIAL_DAMAGE_LAW_H

#include <cmath>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Exponential damage evolution law.
///
/// D(kappa) = 1 - (kappa_0 / kappa) * exp(beta * (kappa_0 - kappa))
/// dD/dkappa = (kappa_0 / kappa^2) * exp(beta*(kappa_0-kappa))
///           + (kappa_0 * beta / kappa) * exp(beta*(kappa_0-kappa))
///
/// Outputs:
///   "damage"   — scalar D in [0, 1)
///   "d_damage" — scalar dD/dkappa
///
/// Inputs:
///   yield_source::kappa — damage multiplier from yield function
template <typename Traits>
class exponential_damage_law final
    : public material_base<exponential_damage_law<Traits>, Traits> {
public:
  using base = material_base<exponential_damage_law<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit exponential_damage_law(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_damage(base::template add_output<value_type>(
            "damage", &exponential_damage_law::compute)),
        m_d_damage(base::template add_output<value_type>("d_damage")),
        m_kappa_0(base::template get_parameter<value_type>("kappa_0")),
        m_beta(base::template get_parameter<value_type>("beta")),
        m_yield_source(base::template get_parameter<std::string>("yield_source")),
        m_kappa(base::template add_input<value_type>(
            m_yield_source, "kappa", EdgeKind::Global))
  {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("yield_source").template add<is_required>();
    para.template insert<value_type>("kappa_0").template add<is_required>();
    para.template insert<value_type>("beta").template add<is_required>();
    return para;
  }

  void compute() {
    const auto kappa = m_kappa.get();

    if (kappa <= m_kappa_0 || kappa < value_type{1e-30}) {
      m_damage = value_type{0};
      m_d_damage = value_type{0};
      return;
    }

    const auto exp_term = std::exp(m_beta * (m_kappa_0 - kappa));
    const auto ratio = m_kappa_0 / kappa;

    m_damage = std::min(value_type{1} - ratio * exp_term, value_type{0.999});
    m_d_damage = ratio * exp_term * (value_type{1} / kappa + m_beta);
  }

private:
  value_type& m_damage;
  value_type& m_d_damage;
  const value_type& m_kappa_0;
  const value_type& m_beta;
  const std::string& m_yield_source;
  const input_property<value_type, property_traits>& m_kappa;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_EXPONENTIAL_DAMAGE_LAW_H
