#ifndef EXPONENTIAL_DAMAGE_LAW_H
#define EXPONENTIAL_DAMAGE_LAW_H

#include <cmath>
#include <utility>

#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/materials/detail/scalar_damage_law.h"

namespace numsim::materials {

/// Exponential law d = 1 - kappa_0 / kappa exp(beta (kappa_0 - kappa)) above the threshold, capped at d_max.
/// Parameters "kappa_0" > 0, "beta" >= 0, "d_max", "yield_source"; outputs "damage", "d_damage".
template <typename Traits>
class exponential_damage_law final : public detail::scalar_damage_law<exponential_damage_law<Traits>, Traits> {
public:
  using base = detail::scalar_damage_law<exponential_damage_law<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit exponential_damage_law(Args &&...args)
      : base(std::forward<Args>(args)...), m_beta(base::template get_parameter<value_type>("beta")) {
    base::require(base::kappa_0() > 0, "kappa_0 must be positive");
    base::require(m_beta >= 0, "beta must be non-negative");
    base::register_output();
  }

  static constexpr char const *type_name() noexcept { return "exponential_damage_law"; }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<value_type>(para, "kappa_0");
    required<value_type>(para, "beta");
    return para;
  }

  [[nodiscard]] std::pair<value_type, value_type> evaluate(value_type kappa) const noexcept {
    auto const ratio{base::kappa_0() / kappa};
    auto const e{std::exp(m_beta * (base::kappa_0() - kappa))};
    return {1 - ratio * e, ratio * e * (1 / kappa + m_beta)};
  }

private:
  value_type const &m_beta;
};

} // namespace numsim::materials

#endif // EXPONENTIAL_DAMAGE_LAW_H
