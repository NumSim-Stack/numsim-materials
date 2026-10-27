#ifndef LINEAR_DAMAGE_LAW_H
#define LINEAR_DAMAGE_LAW_H

#include <utility>

#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/materials/detail/scalar_damage_law.h"

namespace numsim::materials {

/// Linear law d = (kappa - kappa_0) / (kappa_f - kappa_0) between the threshold and kappa_f, capped at d_max.
/// Parameters "kappa_0", "kappa_f" >= kappa_0, "d_max", "yield_source"; outputs "damage", "d_damage".
template <typename Traits>
class linear_damage_law final : public detail::scalar_damage_law<linear_damage_law<Traits>, Traits> {
public:
  using base = detail::scalar_damage_law<linear_damage_law<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit linear_damage_law(Args &&...args)
      : base(std::forward<Args>(args)...), m_kappa_f(base::template get_parameter<value_type>("kappa_f")) {
    base::require(m_kappa_f >= base::kappa_0(), "kappa_f must not be below kappa_0"); // equal: full damage right past the threshold
    base::register_output();
  }

  static constexpr char const *type_name() noexcept { return "linear_damage_law"; }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<value_type>(para, "kappa_0");
    required<value_type>(para, "kappa_f");
    return para;
  }

  /// Full damage from kappa_f on: the base caps it at d_max with a zero slope.
  [[nodiscard]] std::pair<value_type, value_type> evaluate(value_type kappa) const noexcept {
    auto const range{m_kappa_f - base::kappa_0()};
    return kappa >= m_kappa_f ? std::pair{value_type{1}, value_type{0}} : std::pair{(kappa - base::kappa_0()) / range, 1 / range};
  }

private:
  value_type const &m_kappa_f;
};

} // namespace numsim::materials

#endif // LINEAR_DAMAGE_LAW_H
