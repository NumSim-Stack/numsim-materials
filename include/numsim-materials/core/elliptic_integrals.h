#ifndef NUMSIM_MATERIALS_CORE_ELLIPTIC_INTEGRALS_H
#define NUMSIM_MATERIALS_CORE_ELLIPTIC_INTEGRALS_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

/// Carlson's symmetric elliptic integrals by the duplication algorithm
/// (Carlson 1995, Numerical computation of real or complex elliptic integrals).
namespace numsim::materials::elliptic {

namespace detail {
template <typename T>
void require(bool ok, char const *what) {
  if (!ok)
    throw std::domain_error(what);
}
} // namespace detail

/// R_C(x, y) = R_F(x, y, y); y < 0 gives the Cauchy principal value.
template <typename T>
T rc(T x, T y) {
  detail::require<T>(x >= 0 && y != 0, "rc: needs x >= 0 and y != 0");
  if (y < 0)
    return std::sqrt(x / (x - y)) * rc(x - y, -y);
  constexpr T tol{0.0012};
  while (true) {
    auto const lambda{2 * std::sqrt(x) * std::sqrt(y) + y};
    x = (x + lambda) / 4;
    y = (y + lambda) / 4;
    auto const mu{(x + 2 * y) / 3};
    auto const s{(y - mu) / mu};
    if (std::abs(s) < tol)
      return (1 + s * s * (T{0.3} + s * (T{1} / 7 + s * (T{0.375} + s * T{9} / 22)))) / std::sqrt(mu);
  }
}

/// R_F(x, y, z), at most one argument zero.
template <typename T>
T rf(T x, T y, T z) {
  detail::require<T>(std::min({x, y, z}) >= 0 && x + y > 0 && y + z > 0 && z + x > 0, "rf: needs non-negative arguments, at most one zero");
  constexpr T tol{0.0025};
  while (true) {
    auto const sx{std::sqrt(x)}, sy{std::sqrt(y)}, sz{std::sqrt(z)};
    auto const lambda{sx * (sy + sz) + sy * sz};
    x = (x + lambda) / 4;
    y = (y + lambda) / 4;
    z = (z + lambda) / 4;
    auto const mu{(x + y + z) / 3};
    auto const X{1 - x / mu}, Y{1 - y / mu}, Z{1 - z / mu};
    if (std::max({std::abs(X), std::abs(Y), std::abs(Z)}) < tol) {
      auto const e2{X * Y - Z * Z}, e3{X * Y * Z};
      return (1 + (e2 / 24 - T{0.1} - 3 * e3 / 44) * e2 + e3 / 14) / std::sqrt(mu);
    }
  }
}

/// R_D(x, y, z) = R_J(x, y, z, z); z > 0, at most one of x, y zero.
template <typename T>
T rd(T x, T y, T z) {
  detail::require<T>(std::min(x, y) >= 0 && x + y > 0 && z > 0, "rd: needs x, y >= 0 (not both zero) and z > 0");
  constexpr T tol{0.0015};
  T sum{0}, fac{1};
  while (true) {
    auto const sx{std::sqrt(x)}, sy{std::sqrt(y)}, sz{std::sqrt(z)};
    auto const lambda{sx * (sy + sz) + sy * sz};
    sum += fac / (sz * (z + lambda));
    fac /= 4;
    x = (x + lambda) / 4;
    y = (y + lambda) / 4;
    z = (z + lambda) / 4;
    auto const mu{(x + y + 3 * z) / 5};
    auto const X{(mu - x) / mu}, Y{(mu - y) / mu}, Z{(mu - z) / mu};
    if (std::max({std::abs(X), std::abs(Y), std::abs(Z)}) < tol) {
      auto const ea{X * Y}, eb{Z * Z}, ec{ea - eb}, ed{ea - 6 * eb}, ee{ed + ec + ec};
      return 3 * sum + fac * (1 + ed * (-T{3} / 14 + T{9} / 88 * ed - T{4.5} / 26 * Z * ee) +
                              Z * (ee / 6 + Z * (-T{9} / 22 * ec + Z * T{3} / 26 * ea))) / (mu * std::sqrt(mu));
    }
  }
}

/// R_J(x, y, z, p); p > 0, at most one of x, y, z zero.
template <typename T>
T rj(T x, T y, T z, T p) {
  detail::require<T>(std::min({x, y, z}) >= 0 && x + y > 0 && y + z > 0 && z + x > 0 && p > 0,
                     "rj: needs non-negative x, y, z (at most one zero) and p > 0");
  constexpr T tol{0.0015};
  T sum{0}, fac{1};
  while (true) {
    auto const sx{std::sqrt(x)}, sy{std::sqrt(y)}, sz{std::sqrt(z)};
    auto const lambda{sx * (sy + sz) + sy * sz};
    auto const alpha{p * (sx + sy + sz) + sx * sy * sz};
    auto const beta{p * (p + lambda) * (p + lambda)};
    sum += fac * rc(alpha * alpha, beta);
    fac /= 4;
    x = (x + lambda) / 4;
    y = (y + lambda) / 4;
    z = (z + lambda) / 4;
    p = (p + lambda) / 4;
    auto const mu{(x + y + z + 2 * p) / 5};
    auto const X{(mu - x) / mu}, Y{(mu - y) / mu}, Z{(mu - z) / mu}, P{(mu - p) / mu};
    if (std::max({std::abs(X), std::abs(Y), std::abs(Z), std::abs(P)}) < tol) {
      auto const ea{X * (Y + Z) + Y * Z}, eb{X * Y * Z}, ec{P * P}, ed{ea - 3 * ec}, ee{eb + 2 * P * (ea - ec)};
      constexpr T c1{T{3} / 14}, c2{T{1} / 3}, c3{T{3} / 22}, c4{T{3} / 26};
      return 3 * sum + fac * (1 + ed * (-c1 + T{0.75} * c1 * ed - T{1.5} * c4 * ee) + eb * (c2 / 2 + P * (-2 * c3 + P * c4)) +
                              P * ea * (c2 - P * c3) - c2 * P * ec) / (mu * std::sqrt(mu));
    }
  }
}

/// Complete elliptic integrals of the first and second kind, modulus k (not the parameter m = k^2).
template <typename T>
T complete_k(T k) {
  auto const m{k * k};
  return m == 1 ? std::numeric_limits<T>::infinity() : rf(T{0}, 1 - m, T{1});
}

template <typename T>
T complete_e(T k) {
  auto const m{k * k};
  return m == 1 ? T{1} : rf(T{0}, 1 - m, T{1}) - m / 3 * rd(T{0}, 1 - m, T{1});
}

} // namespace numsim::materials::elliptic

#endif // NUMSIM_MATERIALS_CORE_ELLIPTIC_INTEGRALS_H
