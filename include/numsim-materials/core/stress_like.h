#ifndef NUMSIM_MATERIALS_STRESS_LIKE_H
#define NUMSIM_MATERIALS_STRESS_LIKE_H

#include <tmech/tmech.h>

#include <cstddef>
#include <type_traits>

namespace numsim::materials {

/// A symmetric rank-2 tensor that is stress-like: STATEV stores it without the engineering-shear factor.
template <typename T, std::size_t Dim = 3> struct stress_like : tmech::tensor<T, Dim, 2> {
  using base = tmech::tensor<T, Dim, 2>;
  using base::base;
  using base::operator=;
  stress_like() = default;
  stress_like(base const &t) : base(t) {}
};

}  // namespace numsim::materials

/// stress_like takes part in tmech expressions like the tensor it derives from.
template <typename T, std::size_t Dim> struct tmech::is_tensor_type<numsim::materials::stress_like<T, Dim>> {
  using type = std::true_type;
  static constexpr bool value = true;
};

#endif  // NUMSIM_MATERIALS_STRESS_LIKE_H
