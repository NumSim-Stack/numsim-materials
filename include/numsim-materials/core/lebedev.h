#ifndef NUMSIM_MATERIALS_CORE_LEBEDEV_H
#define NUMSIM_MATERIALS_CORE_LEBEDEV_H

#include <array>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/lebedev_tables.h"

/// Lebedev quadrature on the unit sphere; weights sum to one (spherical means).
namespace numsim::materials::lebedev {

using detail::node;

/// Degrees of polynomial exactness with a table: 3 (6 points) ... 23 (194 points).
inline constexpr std::array<std::size_t, 8> precisions_available{3, 5, 7, 9, 11, 13, 15, 23};

inline constexpr std::span<std::size_t const> precisions() { return precisions_available; }

inline std::span<node const> rule(std::size_t precision) {
  switch (precision) {
  case 3: return detail::precision_3;
  case 5: return detail::precision_5;
  case 7: return detail::precision_7;
  case 9: return detail::precision_9;
  case 11: return detail::precision_11;
  case 13: return detail::precision_13;
  case 15: return detail::precision_15;
  case 23: return detail::precision_23;
  default: throw std::invalid_argument("lebedev: no rule of precision " + std::to_string(precision) + " (3, 5, 7, 9, 11, 13, 15, 23)");
  }
}

/// Mean of f over the unit sphere; f takes a unit vector tmech::tensor<T, 3, 1>.
template <typename T = double, typename F>
auto average(std::size_t precision, F &&f) {
  tmech::tensor<T, 3, 1> n;
  using result = decltype(f(n));
  result sum{};
  for (auto const &p : rule(precision)) {
    n(0) = T(p.x);
    n(1) = T(p.y);
    n(2) = T(p.z);
    sum += T(p.w) * f(n);
  }
  return sum;
}

template <typename T = double, typename F>
auto integrate(std::size_t precision, F &&f) {
  return 4 * std::numbers::pi_v<T> * average<T>(precision, std::forward<F>(f));
}

} // namespace numsim::materials::lebedev

#endif // NUMSIM_MATERIALS_CORE_LEBEDEV_H
