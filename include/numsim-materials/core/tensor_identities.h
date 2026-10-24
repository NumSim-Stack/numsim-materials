#ifndef NUMSIM_MATERIALS_CORE_TENSOR_IDENTITIES_H
#define NUMSIM_MATERIALS_CORE_TENSOR_IDENTITIES_H

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <tmech/tmech.h>

/// Fourth-order identities and projectors, built once per (T, Dim), and the symmetric 3D basis.
namespace numsim::materials {

template <typename T, std::size_t Dim>
tmech::tensor<T, Dim, 4> const &sym_identity() noexcept {
  static tmech::tensor<T, Dim, 4> const value{[] {
    auto const I{tmech::eye<T, Dim, 2>()};
    return tmech::tensor<T, Dim, 4>{(tmech::otimesu(I, I) + tmech::otimesl(I, I)) * T{0.5}};
  }()};
  return value;
}

template <typename T, std::size_t Dim>
tmech::tensor<T, Dim, 4> const &skew_identity() noexcept {
  static tmech::tensor<T, Dim, 4> const value{[] {
    auto const I{tmech::eye<T, Dim, 2>()};
    return tmech::tensor<T, Dim, 4>{(tmech::otimesu(I, I) - tmech::otimesl(I, I)) * T{0.5}};
  }()};
  return value;
}

/// I (x) I / Dim
template <typename T, std::size_t Dim>
tmech::tensor<T, Dim, 4> const &vol_projector() noexcept {
  static tmech::tensor<T, Dim, 4> const value{[] {
    auto const I{tmech::eye<T, Dim, 2>()};
    return tmech::tensor<T, Dim, 4>{tmech::otimes(I, I) / T{Dim}};
  }()};
  return value;
}

/// Isym - I (x) I / Dim
template <typename T, std::size_t Dim>
tmech::tensor<T, Dim, 4> const &dev_projector() noexcept {
  static tmech::tensor<T, Dim, 4> const value{sym_identity<T, Dim>() - vol_projector<T, Dim>()};
  return value;
}

/// Symmetric component names and indices in Abaqus/Standard (UMAT, CalculiX) order 11, 22, 33, 12, 13, 23.
/// Not the Abaqus/Explicit (VUMAT) order 11, 22, 33, 12, 23, 31, nor Voigt order 11, 22, 33, 23, 13, 12.
inline constexpr std::array<std::string_view, 6> symmetric_components{"11", "22", "33", "12", "13", "23"};
inline constexpr std::array<std::pair<std::size_t, std::size_t>, 6> symmetric_indices{
    {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}}};

inline std::size_t component_index(std::string_view name) {
  for (std::size_t a{0}; a < symmetric_components.size(); ++a)
    if (symmetric_components[a] == name)
      return a;
  throw std::invalid_argument("unknown component '" + std::string{name} + "' (11, 22, 33, 12, 13, 23)");
}

/// Orthonormal basis of symmetric 3D tensors in symmetric_components order.
template <typename T>
std::array<tmech::tensor<T, 3, 2>, 6> const &symmetric_basis() noexcept {
  static auto const basis = [] {
    std::array<tmech::tensor<T, 3, 2>, 6> b;
    for (std::size_t a{0}; a < 6; ++a) {
      auto const [i, j]{symmetric_indices[a]};
      b[a](i, j) = b[a](j, i) = i == j ? T{1} : T{1} / std::sqrt(T{2});
    }
    return b;
  }();
  return basis;
}

/// All nine components: the symmetric six first, then 21, 31, 32.
inline constexpr std::array<std::string_view, 9> full_components{"11", "22", "33", "12", "13", "23", "21", "31", "32"};
inline constexpr std::array<std::pair<std::size_t, std::size_t>, 9> full_indices{
    {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}, {1, 0}, {2, 0}, {2, 1}}};

inline std::size_t full_component_index(std::string_view name) {
  for (std::size_t a{0}; a < full_components.size(); ++a)
    if (full_components[a] == name)
      return a;
  throw std::invalid_argument("unknown component '" + std::string{name} + "' (11, 22, 33, 12, 13, 23, 21, 31, 32)");
}

/// Basis e_i (x) e_j of all 3D rank-2 tensors, in full_components order.
template <typename T>
std::array<tmech::tensor<T, 3, 2>, 9> const &full_basis() noexcept {
  static auto const basis = [] {
    std::array<tmech::tensor<T, 3, 2>, 9> b;
    for (std::size_t a{0}; a < 9; ++a)
      b[a](full_indices[a].first, full_indices[a].second) = T{1};
    return b;
  }();
  return basis;
}

/// Tensor from its nine components in full_components order.
template <typename T>
tmech::tensor<T, 3, 2> general_tensor(std::span<T const, 9> v) noexcept {
  tmech::tensor<T, 3, 2> t;
  for (std::size_t a{0}; a < 9; ++a)
    t(full_indices[a].first, full_indices[a].second) = v[a];
  return t;
}

/// Symmetric tensor from its components in symmetric_components order (tensor, not engineering, shears).
template <typename T>
tmech::tensor<T, 3, 2> symmetric_tensor(std::span<T const, 6> v) noexcept {
  tmech::tensor<T, 3, 2> t;
  for (std::size_t a{0}; a < 6; ++a) {
    auto const [i, j]{symmetric_indices[a]};
    t(i, j) = t(j, i) = v[a];
  }
  return t;
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_TENSOR_IDENTITIES_H
