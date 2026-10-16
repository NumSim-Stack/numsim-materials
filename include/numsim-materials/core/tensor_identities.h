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
tmech::tensor<T, Dim, 4> const &sym_identity() {
  static tmech::tensor<T, Dim, 4> const value{[] {
    auto const I{tmech::eye<T, Dim, 2>()};
    return tmech::tensor<T, Dim, 4>{(tmech::otimesu(I, I) + tmech::otimesl(I, I)) * T{0.5}};
  }()};
  return value;
}

template <typename T, std::size_t Dim>
tmech::tensor<T, Dim, 4> const &skew_identity() {
  static tmech::tensor<T, Dim, 4> const value{[] {
    auto const I{tmech::eye<T, Dim, 2>()};
    return tmech::tensor<T, Dim, 4>{(tmech::otimesu(I, I) - tmech::otimesl(I, I)) * T{0.5}};
  }()};
  return value;
}

/// I (x) I / Dim
template <typename T, std::size_t Dim>
tmech::tensor<T, Dim, 4> const &vol_projector() {
  static tmech::tensor<T, Dim, 4> const value{[] {
    auto const I{tmech::eye<T, Dim, 2>()};
    return tmech::tensor<T, Dim, 4>{tmech::otimes(I, I) / T{Dim}};
  }()};
  return value;
}

/// Isym - I (x) I / Dim
template <typename T, std::size_t Dim>
tmech::tensor<T, Dim, 4> const &dev_projector() {
  static tmech::tensor<T, Dim, 4> const value{sym_identity<T, Dim>() - vol_projector<T, Dim>()};
  return value;
}

/// Component names in Abaqus order (11, 22, 33, 12, 13, 23) and their indices.
inline constexpr std::array<std::string_view, 6> voigt_components{"11", "22", "33", "12", "13", "23"};
inline constexpr std::array<std::pair<std::size_t, std::size_t>, 6> voigt_indices{
    {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}}};

inline std::size_t component_index(std::string_view name) {
  for (std::size_t a{0}; a < voigt_components.size(); ++a)
    if (voigt_components[a] == name)
      return a;
  throw std::invalid_argument("unknown component '" + std::string{name} + "' (11, 22, 33, 12, 13, 23)");
}

/// Orthonormal basis of symmetric 3D tensors in Abaqus order.
template <typename T>
std::array<tmech::tensor<T, 3, 2>, 6> const &symmetric_basis() {
  static auto const basis = [] {
    std::array<tmech::tensor<T, 3, 2>, 6> b;
    for (std::size_t a{0}; a < 6; ++a) {
      auto const [i, j]{voigt_indices[a]};
      b[a](i, j) = b[a](j, i) = i == j ? T{1} : T{1} / std::sqrt(T{2});
    }
    return b;
  }();
  return basis;
}

/// Symmetric tensor from its components in Abaqus order (tensor, not engineering, shears).
template <typename T>
tmech::tensor<T, 3, 2> symmetric_tensor(std::span<T const, 6> v) {
  tmech::tensor<T, 3, 2> t;
  for (std::size_t a{0}; a < 6; ++a) {
    auto const [i, j]{voigt_indices[a]};
    t(i, j) = t(j, i) = v[a];
  }
  return t;
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_TENSOR_IDENTITIES_H
