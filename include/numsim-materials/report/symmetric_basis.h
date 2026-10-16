#ifndef NUMSIM_MATERIALS_REPORT_SYMMETRIC_BASIS_H
#define NUMSIM_MATERIALS_REPORT_SYMMETRIC_BASIS_H

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <tmech/tmech.h>

/// Orthonormal basis of symmetric 3D tensors and the identities built from it.
namespace numsim::materials::report {

/// Component names in Abaqus order; the basis below follows it.
inline constexpr std::array<std::string_view, 6> components{"11", "22", "33", "12", "13", "23"};

inline constexpr std::array<std::pair<std::size_t, std::size_t>, 6> component_indices{
    {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}}};

inline std::size_t component_index(std::string_view name) {
  for (std::size_t a{0}; a < components.size(); ++a)
    if (components[a] == name)
      return a;
  throw std::invalid_argument("unknown component '" + std::string{name} + "' (11, 22, 33, 12, 13, 23)");
}

template <typename T>
std::array<tmech::tensor<T, 3, 2>, 6> const &symmetric_basis() {
  static auto const basis = [] {
    std::array<tmech::tensor<T, 3, 2>, 6> b;
    for (std::size_t a{0}; a < 6; ++a) {
      auto const [i, j]{component_indices[a]};
      b[a](i, j) = b[a](j, i) = i == j ? T{1} : T{1} / std::sqrt(T{2});
    }
    return b;
  }();
  return basis;
}

template <typename T>
tmech::tensor<T, 3, 4> const &symmetric_identity() {
  static tmech::tensor<T, 3, 4> const Is{[] {
    auto const I{tmech::eye<T, 3, 2>()};
    return tmech::tensor<T, 3, 4>{(tmech::otimesu(I, I) + tmech::otimesl(I, I)) * T{0.5}};
  }()};
  return Is;
}

template <typename T>
tmech::tensor<T, 3, 4> const &skew_identity() {
  static tmech::tensor<T, 3, 4> const Iskw{[] {
    auto const I{tmech::eye<T, 3, 2>()};
    return tmech::tensor<T, 3, 4>{(tmech::otimesu(I, I) - tmech::otimesl(I, I)) * T{0.5}};
  }()};
  return Iskw;
}

/// Symmetric tensor from its components [11, 22, 33, 12, 13, 23].
template <typename T>
tmech::tensor<T, 3, 2> symmetric_tensor(std::span<T const, 6> v) {
  tmech::tensor<T, 3, 2> t;
  for (std::size_t a{0}; a < 6; ++a) {
    auto const [i, j]{component_indices[a]};
    t(i, j) = t(j, i) = v[a];
  }
  return t;
}

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_SYMMETRIC_BASIS_H
