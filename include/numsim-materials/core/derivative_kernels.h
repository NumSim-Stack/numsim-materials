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

/// out += P over a rows x cols block: a step from a seed, whose D is the identity in the seed's columns.
template <typename T>
using seed_kernel = void (*)(T const* P, T* out) noexcept;

template <typename T, int Rows, int Cols>
void add_block(T const* P, T* out) noexcept {
  Eigen::Map<Eigen::Matrix<T, Rows, Cols>>{out} += Eigen::Map<Eigen::Matrix<T, Rows, Cols> const>{P};
}

namespace detail {
/// The Mandel widths of node values: 1, W and W^2.
template <std::size_t Dim>
inline constexpr std::array<int, 3> node_widths{1, static_cast<int>(pair_width<Dim>), static_cast<int>(pair_width<Dim> * pair_width<Dim>)};

template <std::size_t Dim>
std::size_t width_index(std::size_t w) {
  for (std::size_t i{0}; i < node_widths<Dim>.size(); ++i)
    if (w == static_cast<std::size_t>(node_widths<Dim>[i]))
      return i;
  throw std::logic_error("derivatives: no node value has " + std::to_string(w) + " Mandel components");
}
} // namespace detail

/// The kernel of a block between two node values (widths 1, W and W^2); chosen once per plan step.
template <typename T, std::size_t Dim>
product_kernel<T> product_kernel_for(std::size_t rows, std::size_t inner) {
  constexpr auto& widths{detail::node_widths<Dim>};
  constexpr auto table{[&]<std::size_t... k>(std::index_sequence<k...>) {
    return std::array<product_kernel<T>, 9>{&multiply_add<T, widths[k / 3], widths[k % 3]>...};
  }(std::make_index_sequence<9>{})};
  return table[detail::width_index<Dim>(rows) * 3 + detail::width_index<Dim>(inner)];
}

/// The seed kernel of a block between two node values.
template <typename T, std::size_t Dim>
seed_kernel<T> seed_kernel_for(std::size_t rows, std::size_t cols) {
  constexpr auto& widths{detail::node_widths<Dim>};
  constexpr auto table{[&]<std::size_t... k>(std::index_sequence<k...>) {
    return std::array<seed_kernel<T>, 9>{&add_block<T, widths[k / 3], widths[k % 3]>...};
  }(std::make_index_sequence<9>{})};
  return table[detail::width_index<Dim>(rows) * 3 + detail::width_index<Dim>(cols)];
}

} // namespace numsim::materials::derivatives

#endif // NUMSIM_MATERIALS_CORE_DERIVATIVE_KERNELS_H
