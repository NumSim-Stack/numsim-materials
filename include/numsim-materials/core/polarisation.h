#ifndef NUMSIM_MATERIALS_CORE_POLARISATION_H
#define NUMSIM_MATERIALS_CORE_POLARISATION_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/tensor_identities.h"

/// Hill's polarisation tensor P of an ellipsoid in an anisotropic medium C (S = P : C):
/// P = (a1 a2 a3 / 4 pi) int_{|xi| = 1} sym[xi_j N_ik xi_l] / |A xi|^3 dS, N = (C_ijkl xi_j xi_l)^-1,
/// as a quadrature rule of directions that depends on the shape only.
namespace numsim::materials::polarisation {

template <typename T>
struct node {
  tmech::tensor<T, 3, 1> xi;
  T weight;
};

/// Gauss-Legendre nodes and weights on [-1, 1].
template <typename T>
std::vector<std::pair<T, T>> gauss_legendre(std::size_t n) {
  std::vector<std::pair<T, T>> points(n);
  for (std::size_t i{0}; i < n; ++i) {
    T x{std::cos(std::numbers::pi_v<T> * (T(i) + T{0.75}) / (T(n) + T{0.5}))}, dp{0};
    for (int newton{0}; newton < 100; ++newton) {
      T p0{1}, p1{x};
      for (std::size_t k{2}; k <= n; ++k) {
        T const p2{((2 * T(k) - 1) * x * p1 - (T(k) - 1) * p0) / T(k)};
        p0 = p1;
        p1 = p2;
      }
      dp = T(n) * (x * p1 - p0) / (x * x - 1);
      T const dx{p1 / dp};
      x -= dx;
      if (std::abs(dx) <= 4 * std::numeric_limits<T>::epsilon())
        break;
    }
    points[i] = {x, 2 / ((1 - x * x) * dp * dp)};
  }
  return points;
}

/// The m-th of n trapezoid points in u, tan(phi) = stretch tan(u) gathering them towards phi = pi / 2:
/// cos(phi), sin(phi) and the step in phi.
template <typename T>
std::array<T, 3> azimuth_point(std::size_t m, std::size_t n, T stretch) {
  T const u{2 * std::numbers::pi_v<T> * T(m) / T(n)}, cu{std::cos(u)}, su{std::sin(u)};
  T const d{cu * cu + stretch * stretch * su * su};
  return {cu / std::sqrt(d), stretch * su / std::sqrt(d), 2 * std::numbers::pi_v<T> / T(n) * stretch / d};
}

/// P from a rule in the medium C; one 3x3 inverse per direction.
template <typename T>
[[nodiscard]] tmech::tensor<T, 3, 4> hill(std::span<node<T> const> rule, tmech::tensor<T, 3, 4> const &C) {
  using tensor2 = tmech::tensor<T, 3, 2>;
  tmech::tensor<T, 3, 4> const Ct{tmech::basis_change<tmech::sequence<1, 3, 2, 4>>(C)}; // Ct_ikjl = C_ijkl
  tmech::tensor<T, 3, 4> X;
  for (auto const &[xi, w] : rule) {
    tensor2 const xx{tmech::otimes(xi, xi)};
    tensor2 const N{tmech::inv(tensor2{tmech::dcontract(Ct, xx)})};
    X += w * tmech::otimesu(N, xx);
  }
  auto const &Is{sym_identity<T, 3>()};
  return tmech::tensor<T, 3, 4>{tmech::dcontract(Is, tmech::tensor<T, 3, 4>{tmech::dcontract(X, Is)})};
}

/// Ellipsoid with semi-axes a in its frame, R taking the frame to the global one; one semi-axis may be infinite.
/// Gauss-Legendre in the polar angle on panels graded towards the weight's peak, trapezoid in the azimuth.
template <typename T>
[[nodiscard]] std::vector<node<T>> ellipsoid(std::array<T, 3> const &a, tmech::tensor<T, 3, 2> const &R, std::size_t order) {
  constexpr T pi{std::numbers::pi_v<T>};
  if (order < 2)
    throw std::invalid_argument("polarisation: the quadrature order must be at least two");
  std::array<tmech::tensor<T, 3, 1>, 3> e;
  for (std::size_t i{0}; i < 3; ++i) {
    tmech::tensor<T, 3, 1> ei;
    ei(i) = 1;
    e[i] = R * ei;
  }
  std::size_t const azimuth{4 * order};
  std::vector<node<T>> rule;

  std::array<std::size_t, 3> idx{0, 1, 2};
  std::ranges::sort(idx, [&](std::size_t i, std::size_t j) { return a[i] > a[j]; });
  for (auto const x : a)
    if (!(x > 0))
      throw std::invalid_argument("polarisation: semi-axes must be positive");
  if (std::isinf(a[idx[1]]))
    throw std::invalid_argument("polarisation: at most one semi-axis may be infinite");

  if (std::isinf(a[idx[0]])) { // cylinder: (1 / 2 pi) int a_q a_r / (a_q^2 cos^2 + a_r^2 sin^2) Gamma dphi across the axis
    T const aq{a[idx[1]]}, ar{a[idx[2]]}, stretch{std::sqrt(aq / ar)};
    for (std::size_t m{0}; m < azimuth; ++m) {
      auto const [cp, sp, dphi]{azimuth_point(m, azimuth, stretch)};
      rule.push_back({tmech::tensor<T, 3, 1>{cp * e[idx[1]] + sp * e[idx[2]]}, aq * ar / (aq * aq * cp * cp + ar * ar * sp * sp) * dphi / (2 * pi)});
    }
    return rule;
  }

  T const a_max{a[idx[0]]}, a_mid{a[idx[1]]}, a_min{a[idx[2]]};
  bool const prolate{a_mid / a_min <= a_max / a_mid}; // peak on the equator of the longest axis, else at the shortest pole
  std::size_t const p{prolate ? idx[0] : idx[2]}, q{prolate ? idx[1] : idx[0]}, r{prolate ? idx[2] : idx[1]};
  T const stretch{std::sqrt(a[q] / a[r])}, scale{a_min / a_max};

  std::vector<T> edges{0}; // distances from the peak, doubling from the peak's width
  for (T x{scale}; x < pi / 4; x *= 2)
    edges.push_back(x);
  edges.push_back(pi / 2);
  auto const gl{gauss_legendre<T>(order)};
  T const det{a[0] * a[1] * a[2]};
  for (std::size_t k{0}; k + 1 < edges.size(); ++k)
    for (auto const &[t, wt] : gl) {
      T const half{(edges[k + 1] - edges[k]) / 2}, tau{edges[k] + half * (1 + t)};
      T const theta{prolate ? pi / 2 - tau : tau}, ct{std::cos(theta)}, st{std::sin(theta)};
      for (std::size_t m{0}; m < azimuth; ++m) {
        auto const [cp, sp, dphi]{azimuth_point(m, azimuth, stretch)};
        T const Axi{std::sqrt(a[p] * a[p] * ct * ct + st * st * (a[q] * a[q] * cp * cp + a[r] * a[r] * sp * sp))};
        T const w{det / (Axi * Axi * Axi) * st * half * wt * dphi / (2 * pi)}; // the upper hemisphere twice
        rule.push_back({tmech::tensor<T, 3, 1>{ct * e[p] + st * (cp * e[q] + sp * e[r])}, w});
      }
    }
  return rule;
}

} // namespace numsim::materials::polarisation

#endif // NUMSIM_MATERIALS_CORE_POLARISATION_H
