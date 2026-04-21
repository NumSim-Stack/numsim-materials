#ifndef NUMSIM_MATERIALS_LINEAR_DAMAGE_LAW_H
#define NUMSIM_MATERIALS_LINEAR_DAMAGE_LAW_H

#include <algorithm>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Linear damage evolution law.
///
/// D(kappa) = min((kappa - kappa_0) / (kappa_f - kappa_0), 1)
/// dD/dkappa = 1 / (kappa_f - kappa_0)    when kappa_0 < kappa < kappa_f
///
/// Parameters:
///   kappa_0: damage onset threshold
///   kappa_f: full damage strain (D=1)
template <typename Traits>
class linear_damage_law final
    : public material_base<linear_damage_law<Traits>, Traits> {
public:
  using base = material_base<linear_damage_law<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit linear_damage_law(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_damage(base::template add_output<value_type>(
            "damage", &linear_damage_law::compute)),
        m_d_damage(base::template add_output<value_type>("d_damage")),
        m_kappa_0(base::template get_parameter<value_type>("kappa_0")),
        m_kappa_f(base::template get_parameter<value_type>("kappa_f")),
        m_yield_source(base::template get_parameter<std::string>("yield_source")),
        m_kappa(base::template add_input<value_type>(
            m_yield_source, "kappa", EdgeKind::Global))
  {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("yield_source").template add<is_required>();
    para.template insert<value_type>("kappa_0").template add<is_required>();
    para.template insert<value_type>("kappa_f").template add<is_required>();
    return para;
  }

  void compute() {
    const auto kappa = m_kappa.get();
    const auto range = m_kappa_f - m_kappa_0;

    if (kappa <= m_kappa_0 || range < value_type{1e-30}) {
      m_damage = value_type{0};
      m_d_damage = value_type{0};
    } else if (kappa >= m_kappa_f) {
      m_damage = value_type{0.999};
      m_d_damage = value_type{0};
    } else {
      m_damage = (kappa - m_kappa_0) / range;
      m_d_damage = value_type{1} / range;
    }
  }

private:
  value_type& m_damage;
  value_type& m_d_damage;
  const value_type& m_kappa_0;
  const value_type& m_kappa_f;
  const std::string& m_yield_source;
  const input_property<value_type, property_traits>& m_kappa;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_LINEAR_DAMAGE_LAW_H
