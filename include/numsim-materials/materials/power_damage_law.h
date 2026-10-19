#ifndef NUMSIM_MATERIALS_POWER_DAMAGE_LAW_H
#define NUMSIM_MATERIALS_POWER_DAMAGE_LAW_H

#include <cmath>
#include <utility>

#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/materials/detail/scalar_damage_law.h"

namespace numsim::materials {

/// Power law d = K (kappa - kappa_0)^m above the threshold, capped at d_max. Parameters "K" > 0, "m" > 0,
/// "kappa_0", "d_max", "yield_source"; outputs "damage", "d_damage".
template <typename Traits>
class power_damage_law final : public detail::scalar_damage_law<power_damage_law<Traits>, Traits> {
public:
  using base = detail::scalar_damage_law<power_damage_law<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit power_damage_law(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_K(base::template get_parameter<value_type>("K")),
        m_m(base::template get_parameter<value_type>("m")) {
    base::require(m_K > 0, "K must be positive");
    base::require(m_m > 0, "m must be positive");
    base::register_output();
  }

  static constexpr char const *type_name() noexcept { return "power_damage_law"; }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<value_type>(para, "K");
    required<value_type>(para, "m");
    optional<value_type>(para, "kappa_0", value_type{0});
    return para;
  }

  [[nodiscard]] std::pair<value_type, value_type> evaluate(value_type kappa) const noexcept {
    auto const x{kappa - base::kappa_0()};
    return {m_K * std::pow(x, m_m), m_K * m_m * std::pow(x, m_m - 1)};
  }

private:
  value_type const &m_K;
  value_type const &m_m;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_POWER_DAMAGE_LAW_H
