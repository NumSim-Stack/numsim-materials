#ifndef NUMSIM_MATERIALS_SATURATION_DAMAGE_LAW_H
#define NUMSIM_MATERIALS_SATURATION_DAMAGE_LAW_H

#include <cmath>
#include <utility>

#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/materials/detail/scalar_damage_law.h"

namespace numsim::materials {

/// Saturating law d = K (1 - exp(-m (kappa - kappa_0))) above the threshold; K in (0, 1] is the final damage,
/// capped at d_max. Parameters "K", "m" > 0, "kappa_0", "d_max", "yield_source"; outputs "damage", "d_damage".
template <typename Traits>
class saturation_damage_law final : public detail::scalar_damage_law<saturation_damage_law<Traits>, Traits> {
public:
  using base = detail::scalar_damage_law<saturation_damage_law<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit saturation_damage_law(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_K(base::template get_parameter<value_type>("K")),
        m_m(base::template get_parameter<value_type>("m")) {
    base::require(m_K > 0 && m_K <= 1, "K, the final damage, must lie in (0, 1]");
    base::require(m_m > 0, "m must be positive");
    base::register_output();
  }

  static constexpr char const *type_name() noexcept { return "saturation_damage_law"; }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<value_type>(para, "K");
    required<value_type>(para, "m");
    optional<value_type>(para, "kappa_0", value_type{0});
    return para;
  }

  [[nodiscard]] std::pair<value_type, value_type> evaluate(value_type kappa) const noexcept {
    auto const e{std::exp(-m_m * (kappa - base::kappa_0()))};
    return {m_K * (1 - e), m_K * m_m * e};
  }

private:
  value_type const &m_K;
  value_type const &m_m;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_SATURATION_DAMAGE_LAW_H
