#ifndef NUMSIM_MATERIALS_CORE_ELASTIC_CONSTANTS_H
#define NUMSIM_MATERIALS_CORE_ELASTIC_CONSTANTS_H

#include <cmath>
#include <cstddef>

#include <tmech/tmech.h>
#include "numsim-materials/core/tensor_identities.h"

/// Isotropic elastic constants, the isotropic stiffness, and rotations of tensors.
namespace numsim::materials {

template <typename T>
constexpr T bulk_modulus(T E, T nu) { return E / (3 * (1 - 2 * nu)); }
template <typename T>
constexpr T shear_modulus(T E, T nu) { return E / (2 * (1 + nu)); }
template <typename T>
constexpr T youngs_modulus(T K, T G) { return 9 * K * G / (3 * K + G); }
template <typename T>
constexpr T poisson_ratio(T K, T G) { return (3 * K - 2 * G) / (2 * (3 * K + G)); }
template <typename T>
constexpr T lame_lambda(T K, T G) { return K - 2 * G / 3; }
template <typename T>
constexpr T bulk_modulus_from_lame(T lambda, T G) { return lambda + 2 * G / 3; }

/// C = K I (x) I + 2 G Pdev (the K I (x) I form, equal to 3 K Pvol only at Dim = 3).
template <typename T, std::size_t Dim>
tmech::tensor<T, Dim, 4> isotropic_stiffness(T K, T G) {
  auto const I{tmech::eye<T, Dim, 2>()};
  return tmech::tensor<T, Dim, 4>{K * tmech::otimes(I, I) + 2 * G * dev_projector<T, Dim>()};
}

/// R A R^T
template <typename T, std::size_t Dim>
tmech::tensor<T, Dim, 2> rotate(tmech::tensor<T, Dim, 2> const &A, tmech::tensor<T, Dim, 2> const &R) {
  return tmech::tensor<T, Dim, 2>{R * A * tmech::trans(R)};
}

/// C'_ijkl = R_ia R_jb R_kc R_ld C_abcd
template <typename T, std::size_t Dim>
tmech::tensor<T, Dim, 4> rotate(tmech::tensor<T, Dim, 4> const &C, tmech::tensor<T, Dim, 2> const &R) {
  tmech::tensor<T, Dim, 2> const Rt{tmech::trans(R)};
  return tmech::tensor<T, Dim, 4>{tmech::dcontract(tmech::dcontract(tmech::otimesu(R, R), C), tmech::otimesu(Rt, Rt))};
}

/// Rotation by `angle` about the unit vector `axis` (Rodrigues).
template <typename T>
tmech::tensor<T, 3, 2> rotation(tmech::tensor<T, 3, 1> const &axis, T angle) {
  auto const I{tmech::eye<T, 3, 2>()};
  tmech::tensor<T, 3, 2> W; // [axis]_x
  W(0, 1) = -axis(2);
  W(0, 2) = axis(1);
  W(1, 0) = axis(2);
  W(1, 2) = -axis(0);
  W(2, 0) = -axis(1);
  W(2, 1) = axis(0);
  return tmech::tensor<T, 3, 2>{std::cos(angle) * I + std::sin(angle) * W + (1 - std::cos(angle)) * tmech::otimes(axis, axis)};
}

/// A rotation taking e_1 onto the unit vector `direction` (the identity for e_1 itself).
template <typename T>
tmech::tensor<T, 3, 2> rotation_to(tmech::tensor<T, 3, 1> const &direction) {
  tmech::tensor<T, 3, 1> e1;
  e1(0) = 1;
  auto const c{direction(0)}; // e1 . d
  if (c > 1 - T{1e-14})
    return tmech::tensor<T, 3, 2>{tmech::eye<T, 3, 2>()};
  tmech::tensor<T, 3, 1> axis;
  if (c < -1 + T{1e-14}) {
    axis(1) = 1; // half turn about e2
    return rotation(axis, std::acos(T{-1}));
  }
  axis = tmech::cross(e1, direction);
  axis /= tmech::norm(axis);
  return rotation(axis, std::acos(c));
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_ELASTIC_CONSTANTS_H
