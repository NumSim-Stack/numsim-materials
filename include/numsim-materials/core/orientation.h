#ifndef NUMSIM_MATERIALS_CORE_ORIENTATION_H
#define NUMSIM_MATERIALS_CORE_ORIENTATION_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/math/special_functions/ellint_rd.hpp>
#include <tmech/tmech.h>
#include "numsim-materials/core/lebedev.h"

/// Fibre orientation: the angular central Gaussian (ACG) distribution with a given second-order
/// orientation tensor A, and averages over it by mapped Lebedev nodes.
namespace numsim::materials::orientation {

/// ACG parameters in the principal frame of A: p(n) ~ (n . B^-1 n)^-3/2 with B = diag(b), det b = 1,
/// so that <n_i n_j> = a_i delta_ij with a_i = R_D(b_j, b_k, b_i) / 3.
template <typename T>
struct acg {
  std::array<T, 3> b{1, 1, 1};
  std::array<tmech::tensor<T, 3, 1>, 3> axes{}; ///< principal directions of A
  bool aligned{false};                          ///< A = e (x) e: a single direction, axes[0]
};

namespace detail {
template <typename T>
std::array<T, 3> moments(std::array<T, 3> const &b) {
  return {boost::math::ellint_rd(b[1], b[2], b[0]) / 3, boost::math::ellint_rd(b[2], b[0], b[1]) / 3, boost::math::ellint_rd(b[0], b[1], b[2]) / 3};
}
} // namespace detail

/// Fits the ACG to a symmetric orientation tensor with trace one and non-negative eigenvalues.
/// Eigenvalues below `floor` are raised to it (planar or aligned distributions), except the
/// fully aligned case, which is kept as a single direction.
template <typename T>
acg<T> fit_acg(tmech::tensor<T, 3, 2> const &A, T floor = T{1e-6}, T tol = T{1e-12}, int max_iter = 100) {
  if (tmech::norm(tmech::tensor<T, 3, 2>{A - tmech::trans(A)}) > T{1e-12} || std::abs(tmech::trace(A) - 1) > T{1e-10})
    throw std::invalid_argument("orientation: the orientation tensor must be symmetric with trace one");
  auto eig{tmech::eigen_decomposition(A)};
  auto const [eigval, eigvec]{eig.decompose()};
  acg<T> out;
  std::array<T, 3> a;
  for (std::size_t i{0}; i < 3; ++i) {
    a[i] = eigval[i];
    out.axes[i] = eigvec[i];
    if (a[i] < -T{1e-10})
      throw std::invalid_argument("orientation: the orientation tensor has a negative eigenvalue");
  }
  // largest eigenvalue first
  std::array<std::size_t, 3> order{0, 1, 2};
  std::sort(order.begin(), order.end(), [&](auto i, auto j) { return a[i] > a[j]; });
  std::array<T, 3> as;
  std::array<tmech::tensor<T, 3, 1>, 3> axes;
  for (std::size_t i{0}; i < 3; ++i) {
    as[i] = a[order[i]];
    axes[i] = out.axes[order[i]];
  }
  out.axes = axes;
  if (as[0] > 1 - T{1e-10}) {
    out.aligned = true;
    return out;
  }
  T sum{0};
  for (auto &v : as)
    sum += (v = std::max(v, floor));
  for (auto &v : as)
    v /= sum;

  // Newton on x = (ln b1, ln b2), b3 = 1 / (b1 b2); residual: first two moments
  std::array<T, 2> x{0, 0};
  auto const residual = [&](std::array<T, 2> const &y) {
    std::array<T, 3> const b{std::exp(y[0]), std::exp(y[1]), std::exp(-y[0] - y[1])};
    auto const m{detail::moments(b)};
    return std::array<T, 2>{m[0] - as[0], m[1] - as[1]};
  };
  for (int it{0};; ++it) {
    auto const r{residual(x)};
    if (std::hypot(r[0], r[1]) < tol)
      break;
    if (it == max_iter)
      throw std::runtime_error("orientation: the ACG fit did not converge");
    T const h{1e-6};
    std::array<std::array<T, 2>, 2> J; // J[i][j] = d r_i / d x_j
    for (std::size_t j{0}; j < 2; ++j) {
      auto xp{x}, xm{x};
      xp[j] += h;
      xm[j] -= h;
      auto const rp{residual(xp)}, rm{residual(xm)};
      for (std::size_t i{0}; i < 2; ++i)
        J[i][j] = (rp[i] - rm[i]) / (2 * h);
    }
    T const det{J[0][0] * J[1][1] - J[0][1] * J[1][0]};
    x[0] -= (J[1][1] * r[0] - J[0][1] * r[1]) / det;
    x[1] -= (-J[1][0] * r[0] + J[0][0] * r[1]) / det;
  }
  out.b = {std::exp(x[0]), std::exp(x[1]), std::exp(-x[0] - x[1])};
  return out;
}

/// Directions and weights: the Lebedev nodes u mapped to n = (u / sqrt(b)) / |u / sqrt(b)| in the
/// principal frame (a uniform direction through B^-1/2 is ACG distributed). Weights sum to one.
template <typename T>
std::vector<std::pair<tmech::tensor<T, 3, 1>, T>> nodes(acg<T> const &d, std::size_t precision) {
  std::vector<std::pair<tmech::tensor<T, 3, 1>, T>> out;
  if (d.aligned) {
    out.emplace_back(d.axes[0], T{1});
    return out;
  }
  std::array<T, 3> const s{std::sqrt(d.b[0]), std::sqrt(d.b[1]), std::sqrt(d.b[2])};
  for (auto const &q : lebedev::rule(precision)) {
    std::array<T, 3> const c{T(q.x) / s[0], T(q.y) / s[1], T(q.z) / s[2]};
    T const length{std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])};
    tmech::tensor<T, 3, 1> n;
    for (std::size_t i{0}; i < 3; ++i)
      n += c[i] / length * d.axes[i];
    out.emplace_back(n, T(q.w));
  }
  return out;
}

/// The nodes with one direction of each antipodal pair carrying both weights: n and -n are one fibre.
template <typename T>
std::vector<std::pair<tmech::tensor<T, 3, 1>, T>> half_nodes(acg<T> const &d, std::size_t precision) {
  std::vector<std::pair<tmech::tensor<T, 3, 1>, T>> out;
  for (auto const &[n, w] : nodes(d, precision)) {
    auto const partner{std::ranges::find_if(out, [&](auto const &kept) { return tmech::norm(tmech::tensor<T, 3, 1>{kept.first + n}) < T{1e-12}; })};
    if (partner == out.end())
      out.emplace_back(n, w);
    else
      partner->second += w;
  }
  return out;
}

/// Mean of f(n) over the ACG distribution with orientation tensor A.
template <typename T, typename F>
auto average(tmech::tensor<T, 3, 2> const &A, std::size_t precision, F &&f) {
  auto const ns{nodes(fit_acg(A), precision)};
  using result = decltype(f(ns.front().first));
  result sum{};
  for (auto const &[n, w] : ns)
    sum += w * f(n);
  return sum;
}

} // namespace numsim::materials::orientation

#endif // NUMSIM_MATERIALS_CORE_ORIENTATION_H
