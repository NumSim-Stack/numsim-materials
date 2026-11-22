#ifndef NUMSIM_MATERIALS_CORE_DERIVATIVE_KERNELS_H
#define NUMSIM_MATERIALS_CORE_DERIVATIVE_KERNELS_H

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <Eigen/Core>
#include "numsim-materials/core/derivatives.h"

namespace numsim::materials::derivatives {

/// out += P * in over column-major blocks: P rows x inner, in inner x cols, out rows x cols.
template <typename T>
using product_kernel = void (*)(T const* P, T const* in, T* out, Eigen::Index cols) noexcept;

/// The product with rows and inner fixed: one fixed-size matrix-vector product per column.
template <typename T, int Rows, int Inner>
void multiply_add(T const* P, T const* in, T* out, Eigen::Index cols) noexcept {
  Eigen::Map<Eigen::Matrix<T, Rows, Inner> const> const p{P};
  for (Eigen::Index j{0}; j < cols; ++j)
    Eigen::Map<Eigen::Matrix<T, Rows, 1>>{out + j * Rows}.noalias() +=
        p * Eigen::Map<Eigen::Matrix<T, Inner, 1> const>{in + j * Inner};
}

/// The kernel of a block between two node values (widths 1, W and W^2); chosen once per plan step.
template <typename T, std::size_t Dim>
product_kernel<T> product_kernel_for(std::size_t rows, std::size_t inner) {
  constexpr int W{static_cast<int>(pair_width<Dim>)};
  constexpr std::array<int, 3> widths{1, W, W * W};
  constexpr auto table{[&]<std::size_t... k>(std::index_sequence<k...>) {
    return std::array<product_kernel<T>, 9>{&multiply_add<T, widths[k / 3], widths[k % 3]>...};
  }(std::make_index_sequence<9>{})};
  auto const index{[&](std::size_t w) {
    for (std::size_t i{0}; i < widths.size(); ++i)
      if (w == static_cast<std::size_t>(widths[i]))
        return i;
    throw std::logic_error("derivatives: no node value has " + std::to_string(w) + " Mandel components");
  }};
  return table[index(rows) * 3 + index(inner)];
}

} // namespace numsim::materials::derivatives

#endif // NUMSIM_MATERIALS_CORE_DERIVATIVE_KERNELS_H
