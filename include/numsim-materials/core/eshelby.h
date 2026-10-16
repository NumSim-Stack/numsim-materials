#ifndef NUMSIM_MATERIALS_CORE_ESHELBY_H
#define NUMSIM_MATERIALS_CORE_ESHELBY_H

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>

#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/elliptic_integrals.h"
#include "numsim-materials/core/tensor_identities.h"

/// Eshelby tensors of ellipsoidal inclusions in an isotropic matrix (Mura, Micromechanics of
/// defects in solids, ch. 11). Components in the principal frame; orientation by `rotate`.
namespace numsim::materials::eshelby {

namespace detail {

template <typename T>
void check_poisson(T nu) {
  if (!(nu > -1 && nu < T{0.5}))
    throw std::domain_error("eshelby: Poisson's ratio must lie in (-1, 0.5)");
}

/// S from the integrals I_i, I_ij and the squared semi-axes sq (Mura 11.16).
template <typename T>
tmech::tensor<T, 3, 4> from_integrals(T nu, std::array<T, 3> const &sq, std::array<T, 3> const &I,
                                      std::array<std::array<T, 3>, 3> const &Iij) {
  tmech::tensor<T, 3, 4> S;
  T const f{1 / (8 * std::numbers::pi_v<T> * (1 - nu))};
  for (std::size_t i{0}; i < 3; ++i) {
    S(i, i, i, i) = f * (3 * sq[i] * Iij[i][i] + (1 - 2 * nu) * I[i]);
    for (std::size_t j{0}; j < 3; ++j) {
      if (j == i)
        continue;
      S(i, i, j, j) = f * (sq[j] * Iij[i][j] - (1 - 2 * nu) * I[i]);
      S(i, j, i, j) = S(j, i, j, i) = S(i, j, j, i) = S(j, i, i, j) =
          f / 2 * ((sq[i] + sq[j]) * Iij[i][j] + (1 - 2 * nu) * (I[i] + I[j]));
    }
  }
  return S;
}

} // namespace detail

/// Sphere: S = (1 + nu) / (3 (1 - nu)) Pvol + 2 (4 - 5 nu) / (15 (1 - nu)) Pdev.
template <typename T>
tmech::tensor<T, 3, 4> sphere(T nu) {
  detail::check_poisson(nu);
  return tmech::tensor<T, 3, 4>{(1 + nu) / (3 * (1 - nu)) * vol_projector<T, 3>() +
                                2 * (4 - 5 * nu) / (15 * (1 - nu)) * dev_projector<T, 3>()};
}

/// Elliptic cylinder along `axis` (0, 1, 2) with semi-axis `a` along the next axis (cyclic) and
/// `b` along the one after (Mura 11.21).
template <typename T>
tmech::tensor<T, 3, 4> elliptic_cylinder(T nu, T a, T b, std::size_t axis) {
  detail::check_poisson(nu);
  if (axis > 2)
    throw std::out_of_range("eshelby::elliptic_cylinder: axis must be 0, 1 or 2");
  if (!(a > 0 && b > 0) || !std::isfinite(a) || !std::isfinite(b))
    throw std::domain_error("eshelby::elliptic_cylinder: semi-axes must be positive and finite");
  std::size_t const k{axis}, i{(axis + 1) % 3}, j{(axis + 2) % 3};
  T const v{1 / (2 * (1 - nu))}, s{(a + b) * (a + b)}, m{1 - 2 * nu};
  tmech::tensor<T, 3, 4> S;
  S(i, i, i, i) = v * ((b * b + 2 * a * b) / s + m * b / (a + b));
  S(j, j, j, j) = v * ((a * a + 2 * a * b) / s + m * a / (a + b));
  S(i, i, j, j) = v * (b * b / s - m * b / (a + b));
  S(j, j, i, i) = v * (a * a / s - m * a / (a + b));
  S(j, j, k, k) = v * 2 * nu * a / (a + b);
  S(i, i, k, k) = v * 2 * nu * b / (a + b);
  S(i, j, i, j) = S(j, i, j, i) = S(i, j, j, i) = S(j, i, i, j) = v * ((a * a + b * b) / (2 * s) + m / 2);
  S(j, k, j, k) = S(k, j, k, j) = S(j, k, k, j) = S(k, j, j, k) = a / (2 * (a + b));
  S(k, i, k, i) = S(i, k, i, k) = S(k, i, i, k) = S(i, k, k, i) = b / (2 * (a + b));
  return S;
}

/// Circular cylinder along `axis` (0, 1, 2).
template <typename T>
tmech::tensor<T, 3, 4> cylinder(T nu, std::size_t axis) {
  return elliptic_cylinder(nu, T{1}, T{1}, axis);
}

/// Circular cylinder along the unit vector `direction`.
template <typename T>
tmech::tensor<T, 3, 4> cylinder(T nu, tmech::tensor<T, 3, 1> const &direction) {
  return rotate(cylinder(nu, std::size_t{0}), rotation_to(direction));
}

/// Ellipsoid with semi-axes a1, a2, a3 along x1, x2, x3 (any order, equal axes allowed, one may
/// be infinite: then the elliptic cylinder).
template <typename T>
tmech::tensor<T, 3, 4> ellipsoid(T nu, T a1, T a2, T a3) {
  detail::check_poisson(nu);
  std::array<T, 3> const a{a1, a2, a3};
  std::size_t infinite{3};
  for (std::size_t i{0}; i < 3; ++i) {
    if (!(a[i] > 0))
      throw std::domain_error("eshelby::ellipsoid: semi-axes must be positive");
    if (std::isinf(a[i])) {
      if (infinite != 3)
        throw std::domain_error("eshelby::ellipsoid: at most one semi-axis may be infinite");
      infinite = i;
    }
  }
  if (infinite != 3)
    return elliptic_cylinder(nu, a[(infinite + 1) % 3], a[(infinite + 2) % 3], infinite);

  constexpr T pi{std::numbers::pi_v<T>};
  std::array<T, 3> const sq{a1 * a1, a2 * a2, a3 * a3};
  std::array<T, 3> I;
  for (std::size_t i{0}; i < 3; ++i)
    I[i] = 4 * pi / 3 * a1 * a2 * a3 * elliptic::rd(sq[(i + 1) % 3], sq[(i + 2) % 3], sq[i]);

  auto const equal = [&](std::size_t i, std::size_t j) { return std::abs(a[i] - a[j]) <= T{1e-12} * std::max(a[i], a[j]); };
  std::array<std::array<T, 3>, 3> Iij{};
  for (std::size_t i{0}; i < 3; ++i) // distinct pairs first
    for (std::size_t j{0}; j < 3; ++j)
      if (i != j && !equal(i, j))
        Iij[i][j] = (I[j] - I[i]) / (sq[i] - sq[j]);
  for (std::size_t i{0}; i < 3; ++i) // equal pairs from the sum rules (I_ii = I_ij when a_i = a_j)
    for (std::size_t j{0}; j < 3; ++j)
      if (i != j && equal(i, j)) {
        std::size_t const k{3 - i - j};
        Iij[i][j] = equal(i, k) ? 4 * pi / (5 * sq[i]) : pi / sq[i] - Iij[i][k] / 4;
      }
  for (std::size_t i{0}; i < 3; ++i)
    Iij[i][i] = (4 * pi / sq[i] - Iij[i][(i + 1) % 3] - Iij[i][(i + 2) % 3]) / 3;
  return detail::from_integrals(nu, sq, I, Iij);
}

} // namespace numsim::materials::eshelby

#endif // NUMSIM_MATERIALS_CORE_ESHELBY_H
