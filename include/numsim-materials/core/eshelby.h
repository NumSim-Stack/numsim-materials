#ifndef NUMSIM_MATERIALS_CORE_ESHELBY_H
#define NUMSIM_MATERIALS_CORE_ESHELBY_H

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>

#include <boost/math/special_functions/ellint_rd.hpp>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/tensor_identities.h"

/// Eshelby tensors of ellipsoidal inclusions in an isotropic matrix (Mura, Micromechanics of
/// defects in solids, ch. 11); nu is the matrix Poisson's ratio, S has the minor symmetries only.
namespace numsim::materials::eshelby {

namespace detail {

template <typename T>
void check_poisson(T nu) {
  if (!(nu > -1 && nu < T{0.5}))
    throw std::domain_error("eshelby: Poisson's ratio must lie in (-1, 0.5)");
}

/// Positive semi-axes, at most one infinite (its index).
template <typename T>
std::optional<std::size_t> infinite_axis(std::array<T, 3> const &a, char const *who) {
  std::optional<std::size_t> infinite;
  for (std::size_t i{0}; i < 3; ++i) {
    if (!(a[i] > 0))
      throw std::domain_error(std::string{who} + ": semi-axes must be positive");
    if (std::isinf(a[i])) {
      if (infinite)
        throw std::domain_error(std::string{who} + ": at most one semi-axis may be infinite");
      infinite = i;
    }
  }
  return infinite;
}

/// Orthotropic S in the principal frame: S_iijj = A_ij on the projectors e_i (x) e_i, S_ijij = s_ij (i != j).
template <typename T>
tmech::tensor<T, 3, 4> orthotropic(std::array<std::array<T, 3>, 3> const &A, std::array<std::array<T, 3>, 3> const &s) noexcept {
  std::array<tmech::tensor<T, 3, 2>, 3> E;
  for (std::size_t i{0}; i < 3; ++i) {
    tmech::tensor<T, 3, 1> e;
    e(i) = 1;
    E[i] = tmech::otimes(e, e);
  }
  tmech::tensor<T, 3, 4> S;
  for (std::size_t i{0}; i < 3; ++i)
    for (std::size_t j{0}; j < 3; ++j) {
      S += A[i][j] * tmech::otimes(E[i], E[j]);
      if (i != j)
        S += s[i][j] * (tmech::otimesu(E[i], E[j]) + tmech::otimesl(E[i], E[j]));
    }
  return S;
}

/// S from the integrals I_i, I_ij and the squared semi-axes sq (Mura 11.16).
template <typename T>
tmech::tensor<T, 3, 4> from_integrals(T nu, std::array<T, 3> const &sq, std::array<T, 3> const &I,
                                      std::array<std::array<T, 3>, 3> const &Iij) noexcept {
  T const f{1 / (8 * std::numbers::pi_v<T> * (1 - nu))}, m{1 - 2 * nu};
  std::array<std::array<T, 3>, 3> A{}, s{};
  for (std::size_t i{0}; i < 3; ++i) {
    A[i][i] = f * (3 * sq[i] * Iij[i][i] + m * I[i]);
    for (std::size_t j{0}; j < 3; ++j)
      if (j != i) {
        A[i][j] = f * (sq[j] * Iij[i][j] - m * I[i]);
        s[i][j] = f / 2 * ((sq[i] + sq[j]) * Iij[i][j] + m * (I[i] + I[j]));
      }
  }
  return orthotropic(A, s);
}

} // namespace detail

/// Sphere: S = (1 + nu) / (3 (1 - nu)) Pvol + 2 (4 - 5 nu) / (15 (1 - nu)) Pdev.
template <typename T>
[[nodiscard]] tmech::tensor<T, 3, 4> sphere(T nu) {
  detail::check_poisson(nu);
  return tmech::tensor<T, 3, 4>{(1 + nu) / (3 * (1 - nu)) * vol_projector<T, 3>() +
                                2 * (4 - 5 * nu) / (15 * (1 - nu)) * dev_projector<T, 3>()};
}

/// Elliptic cylinder along `axis` (0, 1, 2) with semi-axis `a` along the next axis (cyclic) and
/// `b` along the one after (Mura 11.21).
template <typename T>
[[nodiscard]] tmech::tensor<T, 3, 4> elliptic_cylinder(T nu, T a, T b, std::size_t axis) {
  detail::check_poisson(nu);
  if (axis > 2)
    throw std::out_of_range("eshelby::elliptic_cylinder: axis must be 0, 1 or 2");
  if (!(a > 0 && b > 0) || !std::isfinite(a) || !std::isfinite(b))
    throw std::domain_error("eshelby::elliptic_cylinder: semi-axes must be positive and finite");
  std::size_t const k{axis}, i{(axis + 1) % 3}, j{(axis + 2) % 3};
  T const v{1 / (2 * (1 - nu))}, sum{(a + b) * (a + b)}, m{1 - 2 * nu};
  std::array<std::array<T, 3>, 3> A{}, s{};
  A[i][i] = v * ((b * b + 2 * a * b) / sum + m * b / (a + b));
  A[j][j] = v * ((a * a + 2 * a * b) / sum + m * a / (a + b));
  A[i][j] = v * (b * b / sum - m * b / (a + b));
  A[j][i] = v * (a * a / sum - m * a / (a + b));
  A[j][k] = v * 2 * nu * a / (a + b);
  A[i][k] = v * 2 * nu * b / (a + b);
  s[i][j] = s[j][i] = v * ((a * a + b * b) / (2 * sum) + m / 2);
  s[j][k] = s[k][j] = a / (2 * (a + b));
  s[k][i] = s[i][k] = b / (2 * (a + b));
  return detail::orthotropic(A, s);
}

/// Circular cylinder along `axis` (0, 1, 2).
template <typename T>
[[nodiscard]] tmech::tensor<T, 3, 4> cylinder(T nu, std::size_t axis) {
  return elliptic_cylinder(nu, T{1}, T{1}, axis);
}

/// Circular cylinder along the unit vector `direction`.
template <typename T>
[[nodiscard]] tmech::tensor<T, 3, 4> cylinder(T nu, tmech::tensor<T, 3, 1> const &direction) {
  return rotate(cylinder(nu, std::size_t{0}), rotation_to(direction));
}

/// Ellipsoid with semi-axes a1, a2, a3 along x1, x2, x3 (any order, equal axes allowed, one may
/// be infinite: then the elliptic cylinder).
template <typename T>
[[nodiscard]] tmech::tensor<T, 3, 4> ellipsoid(T nu, T a1, T a2, T a3) {
  detail::check_poisson(nu);
  std::array<T, 3> const a{a1, a2, a3};
  if (auto const infinite{detail::infinite_axis(a, "eshelby::ellipsoid")})
    return elliptic_cylinder(nu, a[(*infinite + 1) % 3], a[(*infinite + 2) % 3], *infinite);

  constexpr T pi{std::numbers::pi_v<T>};
  std::array<T, 3> const sq{a1 * a1, a2 * a2, a3 * a3};
  std::array<T, 3> I;
  for (std::size_t i{0}; i < 3; ++i)
    I[i] = 4 * pi / 3 * a1 * a2 * a3 * boost::math::ellint_rd(sq[(i + 1) % 3], sq[(i + 2) % 3], sq[i]);

  // the difference rule cancels for close axes; below sqrt(eps) the sum rules take over
  auto const equal = [&](std::size_t i, std::size_t j) {
    return std::abs(a[i] - a[j]) <= std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(a[i], a[j]);
  };
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
