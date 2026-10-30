#ifndef NUMSIM_MATERIALS_CORE_DERIVATIVES_H
#define NUMSIM_MATERIALS_CORE_DERIVATIVES_H

#include <array>
#include <cstddef>
#include <functional>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <tmech/tmech.h>
#include "numsim-materials/core/history_property.h"
#include "numsim-materials/core/input_types.h"
#include "numsim-materials/core/property.h"
#include "numsim-materials/core/property_traits.h"

/// Partial derivatives of a callback's outputs with respect to its reads (#217), packed into Mandel blocks.
///
/// Every index pair maps to one Mandel component with the weight sqrt(2) off the diagonal, the layout of
/// solvers/unknown_layout.h. The weights make A : B = mandel(A) . mandel(B), so the chain rule over any
/// middle quantity (scalar, tensor2, tensor4) is a plain matrix product of the packed blocks.
namespace numsim::materials::derivatives {

/// Rank of a value the graph differentiates: 0 for a scalar.
template <typename V>
constexpr std::size_t rank_of() {
  if constexpr (std::is_arithmetic_v<V>)
    return 0;
  else
    return V::rank();
}

/// The value behind an output or a read: T of history_property<T>, input_property<T> and input_history<T>.
template <typename A>
struct value_of {
  using type = A;
};
template <typename T, typename Tr>
struct value_of<numsim_core::history_property<T, Tr>> {
  using type = T;
};
template <typename T, typename Tr>
struct value_of<input_property<T, Tr>> {
  using type = T;
};
template <typename T, typename Tr>
struct value_of<input_history<T, Tr>> {
  using type = T;
};
template <typename A>
using value_of_t = typename value_of<A>::type;

/// A value with a derivative: a scalar, or a symmetric tensor of rank 2 or 4 (minor symmetric).
template <typename V, typename T, std::size_t Dim>
concept node_value = std::is_same_v<V, T> || std::is_same_v<V, tmech::tensor<T, Dim, 2>> ||
                     std::is_same_v<V, tmech::tensor<T, Dim, 4>>;

/// The type a derivative is read as: node_value's scalar, or a tensor of rank 2 to 8 (d tensor4 / d tensor4), of the same T and Dim.
template <typename R, typename T, std::size_t Dim>
concept derivative_value = node_value<R, T, Dim> || std::is_same_v<R, tmech::tensor<T, Dim, 6>> ||
                           std::is_same_v<R, tmech::tensor<T, Dim, 8>>;

template <typename T, std::size_t Dim, std::size_t Rank>
struct partial_value {
  using type = tmech::tensor<T, Dim, Rank>;
};
template <typename T, std::size_t Dim>
struct partial_value<T, Dim, 0> {
  using type = T;
};
/// d out / d in: rank(out) + rank(in), a scalar for two scalars.
template <typename Out, typename In, typename T, std::size_t Dim>
using partial_t = typename partial_value<T, Dim, rank_of<Out>() + rank_of<In>()>::type;

/// Mandel components of a symmetric index pair.
template <std::size_t Dim>
inline constexpr std::size_t pair_width{Dim * (Dim + 1) / 2};

/// Mandel components of a value of rank `rank`, one Mandel index per index pair.
template <std::size_t Dim>
constexpr std::size_t width(std::size_t rank) {
  std::size_t w{1};
  for (std::size_t pair{0}; pair < rank / 2; ++pair)
    w *= pair_width<Dim>;
  return w;
}

namespace detail {
/// The index pairs in tmech's Mandel order: 11, 22, 33, 23, 13, 12 (2D: 11, 22, 12).
template <std::size_t Dim>
constexpr auto pairs() {
  if constexpr (Dim == 2)
    return std::array<std::array<std::size_t, 2>, 3>{{{0, 0}, {1, 1}, {0, 1}}};
  else
    return std::array<std::array<std::size_t, 2>, 6>{{{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}}};
}

/// The Mandel component of the index pair (i, j).
template <std::size_t Dim>
constexpr std::size_t component(std::size_t i, std::size_t j) {
  if (i == j)
    return i;
  if constexpr (Dim == 2)
    return 2;
  else
    return 6 - i - j;
}

template <typename V, std::size_t... I>
constexpr decltype(auto) entry(V& v, std::array<std::size_t, sizeof...(I)> const& idx, std::index_sequence<I...>) {
  return v(idx[I]...);
}

constexpr std::size_t power(std::size_t base, std::size_t exponent) {
  std::size_t p{1};
  while (exponent-- > 0)
    p *= base;
  return p;
}
} // namespace detail

/// Packs P = d out / d in (rank(out) = OutRank) into a width(OutRank) x width(rank(P) - OutRank) column-major block;
/// each index pair is projected onto its symmetric part. Index loops: packing is the boundary to the matrices.
template <std::size_t Dim, std::size_t OutRank, typename P, typename T>
void pack(P const& p, T* block) {
  if constexpr (std::is_arithmetic_v<P>) {
    *block = p;
  } else {
    constexpr std::size_t rank{P::rank()}, n{rank / 2}, W{pair_width<Dim>};
    constexpr std::size_t rows{width<Dim>(OutRank)}, cols{width<Dim>(rank - OutRank)};
    constexpr auto pair{detail::pairs<Dim>()};
    std::array<std::size_t, n> m{};
    std::array<std::size_t, rank> idx{};
    for (std::size_t flat{0}; flat < rows * cols; ++flat) {
      for (std::size_t k{n}, rest{flat}; k-- > 0; rest /= W)
        m[k] = rest % W; // the first pair varies slowest
      T sum{0};
      for (std::size_t swaps{0}; swaps < (std::size_t{1} << n); ++swaps) {
        for (std::size_t k{0}; k < n; ++k) {
          auto const flip{(swaps >> k) & 1};
          idx[2 * k] = pair[m[k]][flip];
          idx[2 * k + 1] = pair[m[k]][1 - flip];
        }
        sum += detail::entry(p, idx, std::make_index_sequence<rank>{});
      }
      T weight{T{1} / T(std::size_t{1} << n)};
      for (auto const mk : m)
        if (mk >= Dim)
          weight *= std::numbers::sqrt2_v<T>;
      block[flat / cols + (flat % cols) * rows] = weight * sum;
    }
  }
}

/// The inverse of pack(): a minor-symmetric tensor (or a scalar) from its column-major Mandel block.
template <std::size_t Dim, std::size_t OutRank, typename P, typename T>
void unpack(T const* block, P& p) {
  if constexpr (std::is_arithmetic_v<P>) {
    p = *block;
  } else {
    constexpr std::size_t rank{P::rank()}, W{pair_width<Dim>};
    constexpr std::size_t rows{width<Dim>(OutRank)}, cols{width<Dim>(rank - OutRank)};
    std::array<std::size_t, rank> idx{};
    for (std::size_t flat{0}; flat < detail::power(Dim, rank); ++flat) {
      for (std::size_t k{rank}, rest{flat}; k-- > 0; rest /= Dim)
        idx[k] = rest % Dim;
      std::size_t mandel{0};
      T weight{1};
      for (std::size_t k{0}; k < rank; k += 2) {
        auto const c{detail::component<Dim>(idx[k], idx[k + 1])};
        mandel = mandel * W + c;
        if (c >= Dim)
          weight *= std::numbers::sqrt2_v<T>;
      }
      detail::entry(p, idx, std::make_index_sequence<rank>{}) = block[mandel / cols + (mandel % cols) * rows] / weight;
    }
  }
}

/// The value of `source` as V: a property's value or a history property's new value.
template <typename V>
V& value_in(property_base& source) {
  if (auto* plain{dynamic_cast<property<V>*>(&source)})
    return plain->get();
  if (auto* history{dynamic_cast<history_property<V, property_traits>*>(&source)})
    return history->new_value();
  throw std::logic_error("derivatives: '" + source.traits().id.owner + "::" + source.traits().id.name +
                         "' holds another type than its reader");
}

/// The rank of the value `prop` holds when it has a derivative (node_value), else nothing.
template <typename T, std::size_t Dim>
std::optional<std::size_t> property_rank(property_base const& prop) {
  auto const holds{[&]<typename V>(std::type_identity<V>) {
    return dynamic_cast<property<V> const*>(&prop) || dynamic_cast<history_property<V, property_traits> const*>(&prop);
  }};
  if (holds(std::type_identity<T>{}))
    return 0;
  if (holds(std::type_identity<tmech::tensor<T, Dim, 2>>{}))
    return 2;
  if (holds(std::type_identity<tmech::tensor<T, Dim, 4>>{}))
    return 4;
  return std::nullopt;
}

/// A read the check perturbs one Mandel component at a time: its value is held by a property, given as `source`.
template <typename T>
class read_probe {
public:
  virtual ~read_probe() = default;
  /// Mandel components of the read.
  virtual std::size_t width() const noexcept = 0;
  /// Mandel component k of the read's value, held by `source`.
  virtual T read_component(property_base& source, std::size_t k) const = 0;
  /// Adds h to Mandel component k of the read's value; the returned function restores the value.
  virtual std::function<void()> perturb(property_base& source, std::size_t k, T h) const = 0;
};

template <typename T, std::size_t Dim, typename V>
class typed_read : public read_probe<T> {
public:
  std::size_t width() const noexcept override { return derivatives::width<Dim>(rank_of<V>()); }

  T read_component(property_base& source, std::size_t k) const override {
    std::array<T, derivatives::width<Dim>(rank_of<V>())> packed{};
    derivatives::pack<Dim, rank_of<V>()>(value_in<V>(source), packed.data());
    return packed[k];
  }

  std::function<void()> perturb(property_base& source, std::size_t k, T h) const override {
    auto& v{value_in<V>(source)};
    std::function<void()> restore{[&v, old = v] { v = old; }};
    if constexpr (std::is_arithmetic_v<V>) {
      v += h;
    } else {
      std::array<T, derivatives::width<Dim>(rank_of<V>())> unit{};
      unit[k] = h;
      V step;
      derivatives::unpack<Dim, rank_of<V>()>(unit.data(), step);
      v = V{v + step};
    }
    return restore;
  }
};

/// One declared partial d output / d read of a callback: the value the callback fills, packed on demand.
template <typename T>
class partial_block : public read_probe<T> {
public:
  partial_block(std::size_t rows, std::size_t cols) noexcept : m_rows(rows), m_cols(cols) {}
  partial_block(partial_block const&) = delete;
  partial_block& operator=(partial_block const&) = delete;

  std::size_t rows() const noexcept { return m_rows; }
  std::size_t cols() const noexcept { return m_cols; }
  std::size_t width() const noexcept override { return m_cols; }
  /// The partial as a rows() x cols() column-major Mandel block.
  virtual void pack(T* block) const = 0;
  /// The output's value, rows() Mandel components.
  virtual void pack_output(T* value) const = 0;

private:
  std::size_t m_rows, m_cols;
};

template <typename T, std::size_t Dim, typename Out, typename In>
class typed_partial final : public partial_block<T> {
public:
  using output_value = value_of_t<Out>;
  using read_value = value_of_t<In>;
  static constexpr std::size_t out_rank{rank_of<output_value>()};

  explicit typed_partial(Out const& output)
      : partial_block<T>{derivatives::width<Dim>(out_rank), derivatives::width<Dim>(rank_of<read_value>())}, m_output(output) {}

  partial_t<output_value, read_value, T, Dim> value{};

  void pack(T* block) const override { derivatives::pack<Dim, out_rank>(value, block); }

  void pack_output(T* out) const override {
    if constexpr (requires(Out const& o) { o.new_value(); })
      derivatives::pack<Dim, out_rank>(m_output.new_value(), out);
    else
      derivatives::pack<Dim, out_rank>(m_output, out);
  }

  T read_component(property_base& source, std::size_t k) const override { return m_read.read_component(source, k); }

  std::function<void()> perturb(property_base& source, std::size_t k, T h) const override { return m_read.perturb(source, k, h); }

private:
  Out const& m_output;
  typed_read<T, Dim, read_value> m_read;
};

/// A requested derivative as the host reads it, unpacked from its accumulated block.
template <typename T>
class derivative_slot_base {
public:
  virtual ~derivative_slot_base() = default;
  virtual void unpack(T const* block) = 0;
};

template <typename T, std::size_t Dim, typename R>
class derivative_slot final : public derivative_slot_base<T> {
public:
  explicit derivative_slot(std::size_t out_rank) : m_out_rank(out_rank) {}

  R value{};

  void unpack(T const* block) override {
    switch (m_out_rank) {
    case 0: derivatives::unpack<Dim, 0>(block, value); break;
    case 2:
      if constexpr (rank_of<R>() >= 2)
        derivatives::unpack<Dim, 2>(block, value);
      break;
    case 4:
      if constexpr (rank_of<R>() >= 4)
        derivatives::unpack<Dim, 4>(block, value);
      break;
    default: break;
    }
  }

private:
  std::size_t m_out_rank;
};

} // namespace numsim::materials::derivatives

#endif // NUMSIM_MATERIALS_CORE_DERIVATIVES_H
