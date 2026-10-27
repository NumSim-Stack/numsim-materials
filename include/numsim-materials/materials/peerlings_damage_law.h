#ifndef NUMSIM_MATERIALS_PEERLINGS_DAMAGE_LAW_H
#define NUMSIM_MATERIALS_PEERLINGS_DAMAGE_LAW_H

#include <cmath>
#include <utility>

#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/materials/detail/scalar_damage_law.h"

namespace numsim::materials {

/// Peerlings' law d = 1 - kappa_0 / kappa (1 - alpha + alpha exp(beta (kappa_0 - kappa))), tending to 1 for
/// alpha = 1 and to 1 - (1 - alpha) kappa_0 / kappa otherwise; capped at d_max. Parameters "kappa_0" > 0,
/// "alpha" in [0, 1], "beta" >= 0, "d_max", "yield_source"; outputs "damage", "d_damage".
template <typename Traits>
class peerlings_damage_law final : public detail::scalar_damage_law<peerlings_damage_law<Traits>, Traits> {
public:
  using base = detail::scalar_damage_law<peerlings_damage_law<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit peerlings_damage_law(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_alpha(base::template get_parameter<value_type>("alpha")),
        m_beta(base::template get_parameter<value_type>("beta")) {
    base::require(base::kappa_0() > 0, "kappa_0 must be positive");
    base::require(m_alpha >= 0 && m_alpha <= 1, "alpha must lie in [0, 1]");
    base::require(m_beta >= 0, "beta must be non-negative");
    base::register_output();
  }

  static constexpr char const *type_name() noexcept { return "peerlings_damage_law"; }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<value_type>(para, "kappa_0");
    required<value_type>(para, "alpha");
    required<value_type>(para, "beta");
    return para;
  }

  [[nodiscard]] std::pair<value_type, value_type> evaluate(value_type kappa) const noexcept {
    auto const kappa_0{base::kappa_0()};
    auto const e{std::exp(m_beta * (kappa_0 - kappa))};
    auto const g{1 - m_alpha + m_alpha * e}; // d = 1 - kappa_0 g / kappa
    return {1 - kappa_0 * g / kappa, kappa_0 * (g / (kappa * kappa) + m_alpha * m_beta * e / kappa)};
  }

private:
  value_type const &m_alpha;
  value_type const &m_beta;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_PEERLINGS_DAMAGE_LAW_H
