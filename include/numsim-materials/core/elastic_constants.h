#ifndef NUMSIM_MATERIALS_CORE_ELASTIC_CONSTANTS_H
#define NUMSIM_MATERIALS_CORE_ELASTIC_CONSTANTS_H

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>

#include <tmech/tmech.h>
#include "numsim-materials/core/tensor_identities.h"

/// Isotropic elastic constants, the isotropic stiffness, and rotations of tensors.
namespace numsim::materials {

template <typename T>
[[nodiscard]] constexpr T bulk_modulus(T E, T nu) noexcept { return E / (3 * (1 - 2 * nu)); }
template <typename T>
[[nodiscard]] constexpr T shear_modulus(T E, T nu) noexcept { return E / (2 * (1 + nu)); }
template <typename T>
[[nodiscard]] constexpr T youngs_modulus(T K, T G) noexcept { return 9 * K * G / (3 * K + G); }
template <typename T>
[[nodiscard]] constexpr T poisson_ratio(T K, T G) noexcept { return (3 * K - 2 * G) / (2 * (3 * K + G)); }
template <typename T>
[[nodiscard]] constexpr T lame_lambda(T K, T G) noexcept { return K - 2 * G / 3; }
template <typename T>
[[nodiscard]] [[nodiscard]] constexpr T bulk_modulus_from_lame(T lambda, T G) noexcept { return lambda + 2 * G / 3; }

/// C = K I (x) I + 2 G Pdev (the K I (x) I form, equal to 3 K Pvol only at Dim = 3).
template <typename T, std::size_t Dim>
[[nodiscard]] tmech::tensor<T, Dim, 4> isotropic_stiffness(T K, T G) noexcept {
  auto const I{tmech::eye<T, Dim, 2>()};
  return tmech::tensor<T, Dim, 4>{K * tmech::otimes(I, I) + 2 * G * dev_projector<T, Dim>()};
}

/// R A R^T
template <typename T, std::size_t Dim>
[[nodiscard]] tmech::tensor<T, Dim, 2> rotate(tmech::tensor<T, Dim, 2> const &A, tmech::tensor<T, Dim, 2> const &R) noexcept {
  return tmech::tensor<T, Dim, 2>{R * A * tmech::trans(R)};
}

/// C'_ijkl = R_ia R_jb R_kc R_ld C_abcd
template <typename T, std::size_t Dim>
[[nodiscard]] tmech::tensor<T, Dim, 4> rotate(tmech::tensor<T, Dim, 4> const &C, tmech::tensor<T, Dim, 2> const &R) noexcept {
  tmech::tensor<T, Dim, 2> const Rt{tmech::trans(R)};
  return tmech::tensor<T, Dim, 4>{tmech::dcontract(tmech::dcontract(tmech::otimesu(R, R), C), tmech::otimesu(Rt, Rt))};
}

/// Rotation by `angle` about the unit vector `axis` (Rodrigues); [axis]_x = -(epsilon . axis).
template <typename T>
[[nodiscard]] tmech::tensor<T, 3, 2> rotation(tmech::tensor<T, 3, 1> const &axis, T angle) noexcept {
  auto const I{tmech::eye<T, 3, 2>()};
  return tmech::tensor<T, 3, 2>{std::cos(angle) * I - std::sin(angle) * (tmech::levi_civita<T, 3>() * axis) +
                                (1 - std::cos(angle)) * tmech::otimes(axis, axis)};
}

/// A rotation taking e_1 onto the unit vector `direction` (the identity for e_1 itself).
template <typename T>
[[nodiscard]] tmech::tensor<T, 3, 2> rotation_to(tmech::tensor<T, 3, 1> const &direction) noexcept {
  constexpr T tol{8 * std::numeric_limits<T>::epsilon()};
  tmech::tensor<T, 3, 1> e1;
  e1(0) = 1;
  auto const c{direction(0)}; // e1 . d
  if (c > 1 - tol)
    return tmech::tensor<T, 3, 2>{tmech::eye<T, 3, 2>()};
  tmech::tensor<T, 3, 1> axis;
  if (c < -1 + tol) {
    axis(1) = 1; // half turn about e2
    return rotation(axis, std::numbers::pi_v<T>);
  }
  axis = tmech::cross(e1, direction);
  axis /= tmech::norm(axis);
  return rotation(axis, std::acos(c));
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_ELASTIC_CONSTANTS_H
