#ifndef NUMSIM_MATERIALS_CORE_DERIVATIVES_H
#define NUMSIM_MATERIALS_CORE_DERIVATIVES_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
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

namespace detail {
/// Index tables of pack() and unpack() for a tensor of rank `Rank` split at `OutRank`, over its contiguous storage
/// (the first index slowest). The weight of an entry is sqrt(2)^count, count being its off-diagonal pairs.
template <std::size_t Dim, std::size_t Rank, std::size_t OutRank>
struct mandel_tables {
  static constexpr std::size_t n{Rank / 2}, W{pair_width<Dim>}, copies{std::size_t{1} << n};
  static constexpr std::size_t rows{width<Dim>(OutRank)}, cols{width<Dim>(Rank - OutRank)}, size{power(Dim, Rank)};

  /// tensor entry -> its position in the column-major block, and its off-diagonal pairs
  std::array<std::uint16_t, size> position{};
  std::array<std::uint8_t, size> count{};
  /// block position -> the `copies` tensor entries of its index pairs, and the off-diagonal pairs
  std::array<std::array<std::uint16_t, copies>, rows * cols> entries{};
  std::array<std::uint8_t, rows * cols> weight_count{};

  constexpr mandel_tables() {
    for (std::size_t flat{0}; flat < size; ++flat) {
      std::size_t mandel{0}, off{0}, rest{flat};
      std::array<std::size_t, Rank> idx{};
      for (std::size_t k{Rank}; k-- > 0; rest /= Dim)
        idx[k] = rest % Dim;
      for (std::size_t k{0}; k < Rank; k += 2) {
        auto const c{component<Dim>(idx[k], idx[k + 1])};
        mandel = mandel * W + c;
        off += c >= Dim;
      }
      position[flat] = static_cast<std::uint16_t>(mandel / cols + (mandel % cols) * rows);
      count[flat] = static_cast<std::uint8_t>(off);
    }
    constexpr auto pair{pairs<Dim>()};
    for (std::size_t mandel{0}; mandel < rows * cols; ++mandel) {
      std::array<std::size_t, n> m{};
      std::size_t rest{mandel}, off{0};
      for (std::size_t k{n}; k-- > 0; rest /= W) {
        m[k] = rest % W;
        off += m[k] >= Dim;
      }
      auto const q{mandel / cols + (mandel % cols) * rows};
      weight_count[q] = static_cast<std::uint8_t>(off);
      for (std::size_t swaps{0}; swaps < copies; ++swaps) {
        std::size_t flat{0};
        for (std::size_t k{0}; k < n; ++k) {
          auto const flip{(swaps >> k) & 1};
          flat = (flat * Dim + pair[m[k]][flip]) * Dim + pair[m[k]][1 - flip];
        }
        entries[q][swaps] = static_cast<std::uint16_t>(flat);
      }
    }
  }
};
template <std::size_t Dim, std::size_t Rank, std::size_t OutRank>
inline constexpr mandel_tables<Dim, Rank, OutRank> mandel_table{};

/// The sum of the tensor entries of block position Q (its index pairs swapped), indices as constants.
template <std::size_t Dim, std::size_t Rank, std::size_t OutRank, std::size_t Q, typename T, std::size_t... C>
T entry_sum(T const* data, std::index_sequence<C...>) noexcept {
  return (data[mandel_table<Dim, Rank, OutRank>.entries[Q][C]] + ...);
}

/// pack() and unpack() of up to this many entries run with the table indices as constants.
inline constexpr std::size_t unroll_limit{256};

/// sqrt(2)^c for c = 0..n, multiplied up one factor at a time.
template <typename T, std::size_t N>
constexpr std::array<T, N + 1> sqrt2_powers() {
  std::array<T, N + 1> w{};
  w[0] = T{1};
  for (std::size_t c{1}; c <= N; ++c)
    w[c] = w[c - 1] * std::numbers::sqrt2_v<T>;
  return w;
}
} // namespace detail

template <std::size_t Dim, std::size_t Rank, std::size_t OutRank, typename T>
void pack_data(T const* data, T* block);
template <std::size_t Dim, std::size_t Rank, std::size_t OutRank, typename T>
void unpack_data(T const* block, T* data);

/// Packs P = d out / d in (rank(out) = OutRank) into a width(OutRank) x width(rank(P) - OutRank) column-major block;
/// each index pair is projected onto its symmetric part. Table driven: packing is the boundary to the matrices.
template <std::size_t Dim, std::size_t OutRank, typename P, typename T>
void pack(P const& p, T* block) {
  if constexpr (std::is_arithmetic_v<P>)
    *block = p;
  else
    pack_data<Dim, P::rank(), OutRank>(p.raw_data(), block);
}

/// pack() of a tensor of rank `Rank` given by its storage.
template <std::size_t Dim, std::size_t Rank, std::size_t OutRank, typename T>
void pack_data(T const* data, T* block) {
  if constexpr (Rank == 0) {
    *block = *data;
  } else {
    constexpr std::size_t rank{Rank};
    using table_type = detail::mandel_tables<Dim, rank, OutRank>;
    constexpr auto const& table{detail::mandel_table<Dim, rank, OutRank>};
    constexpr auto scale{detail::sqrt2_powers<T, table.n>()};
    if constexpr (table.rows * table.cols <= detail::unroll_limit) {
      [&]<std::size_t... q>(std::index_sequence<q...>) {
        ((block[q] = (scale[table.weight_count[q]] / T(table_type::copies)) *
                     detail::entry_sum<Dim, rank, OutRank, q>(data, std::make_index_sequence<table_type::copies>{})),
         ...);
      }(std::make_index_sequence<table.rows * table.cols>{});
    } else {
      for (std::size_t q{0}; q < table.rows * table.cols; ++q) {
        T sum{0};
        for (auto const flat : table.entries[q])
          sum += data[flat];
        block[q] = (scale[table.weight_count[q]] / T(table.copies)) * sum;
      }
    }
  }
}

/// The inverse of pack(): a minor-symmetric tensor (or a scalar) from its column-major Mandel block.
template <std::size_t Dim, std::size_t OutRank, typename P, typename T>
void unpack(T const* block, P& p) {
  if constexpr (std::is_arithmetic_v<P>)
    p = *block;
  else
    unpack_data<Dim, P::rank(), OutRank>(block, p.raw_data());
}

/// unpack() into the storage of a tensor of rank `Rank`.
template <std::size_t Dim, std::size_t Rank, std::size_t OutRank, typename T>
void unpack_data(T const* block, T* data) {
  if constexpr (Rank == 0) {
    *data = *block;
  } else {
    constexpr std::size_t rank{Rank};
    constexpr auto const& table{detail::mandel_table<Dim, rank, OutRank>};
    constexpr auto scale{detail::sqrt2_powers<T, table.n>()};
    if constexpr (table.size <= detail::unroll_limit) {
      [&]<std::size_t... flat>(std::index_sequence<flat...>) {
        ((data[flat] = block[table.position[flat]] / scale[table.count[flat]]), ...);
      }(std::make_index_sequence<table.size>{});
    } else {
      for (std::size_t flat{0}; flat < table.size; ++flat)
        data[flat] = block[table.position[flat]] / scale[table.count[flat]];
    }
  }
}

namespace detail {
/// f.template operator()<OutRank, InRank>() for a system row of rank 0 or 2 and a column of rank 0, 2 or 4.
template <typename F>
void with_ranks(std::size_t out_rank, std::size_t in_rank, F&& f) {
  auto const in{[&]<std::size_t A>() {
    switch (in_rank) {
    case 0: f.template operator()<A, 0>(); return;
    case 2: f.template operator()<A, 2>(); return;
    case 4: f.template operator()<A, 4>(); return;
    default: throw std::logic_error("derivatives: no node value has rank " + std::to_string(in_rank));
    }
  }};
  switch (out_rank) {
  case 0: in.template operator()<0>(); return;
  case 2: in.template operator()<2>(); return;
  default: throw std::logic_error("derivatives: a system row has rank 0 or 2, not " + std::to_string(out_rank));
  }
}
} // namespace detail

/// pack() of d out / d in (ranks known at run time) given in full notation: the boundary to an LU solve.
template <std::size_t Dim, typename T>
void pack_full(std::size_t out_rank, std::size_t in_rank, T const* data, T* block) {
  detail::with_ranks(out_rank, in_rank, [&]<std::size_t A, std::size_t B>() { pack_data<Dim, A + B, A>(data, block); });
}

/// unpack() of d out / d in into full notation.
template <std::size_t Dim, typename T>
void unpack_full(std::size_t out_rank, std::size_t in_rank, T const* block, T* data) {
  detail::with_ranks(out_rank, in_rank, [&]<std::size_t A, std::size_t B>() { unpack_data<Dim, A + B, A>(block, data); });
}

/// Entries of a value of rank `rank` in full notation.
template <std::size_t Dim>
constexpr std::size_t full_size(std::size_t rank) {
  return detail::power(Dim, rank);
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

/// A distinct address per type: identifies a type-erased block without a dynamic_cast.
template <typename B>
inline constexpr char type_tag{};

/// Who reads a partial block, as bits: a fill writes the blocks of the readers asking.
enum reader : std::uint8_t { derivative_reader = 1, jacobian_reader = 2, every_reader = 3 };

/// One declared partial d output / d read of a callback: the value the callback fills, packed on demand.
template <typename T>
class partial_block : public read_probe<T> {
public:
  partial_block(std::size_t out_rank, std::size_t in_rank, std::size_t rows, std::size_t cols) noexcept
      : m_out_rank(out_rank), m_in_rank(in_rank), m_rows(rows), m_cols(cols) {}
  partial_block(partial_block const&) = delete;
  partial_block& operator=(partial_block const&) = delete;

  std::size_t out_rank() const noexcept { return m_out_rank; }
  std::size_t in_rank() const noexcept { return m_in_rank; }
  std::size_t rows() const noexcept { return m_rows; }
  std::size_t cols() const noexcept { return m_cols; }
  /// The partial as its tmech value: the plans read it here.
  virtual void const* object() const noexcept = 0;
  std::size_t width() const noexcept override { return m_cols; }
  /// The partial as a rows() x cols() column-major Mandel block.
  virtual void pack(T* block) const = 0;
  /// The output's value, rows() Mandel components.
  virtual void pack_output(T* value) const = 0;
  /// The value of a scalar by scalar partial, else null.
  virtual T const* scalar() const noexcept { return nullptr; }
  /// |P - sym(P)| / |P| over every index pair: plans use partials as written, so they must be minor-symmetric.
  virtual T asymmetry() const = 0;

  /// The readers (reader bits) of the block.
  std::uint8_t readers() const noexcept { return m_readers; }
  void add_readers(std::uint8_t readers) noexcept { m_readers |= readers; }

private:
  std::size_t m_out_rank, m_in_rank, m_rows, m_cols;
  std::uint8_t m_readers{0};
};

template <typename T, std::size_t Dim, typename Out, typename In>
class typed_partial final : public partial_block<T> {
public:
  using output_value = value_of_t<Out>;
  using read_value = value_of_t<In>;
  static constexpr std::size_t out_rank{rank_of<output_value>()};

  explicit typed_partial(Out const& output)
      : partial_block<T>{out_rank, rank_of<read_value>(), derivatives::width<Dim>(out_rank), derivatives::width<Dim>(rank_of<read_value>())},
        m_output(output) {}

  partial_t<output_value, read_value, T, Dim> value{};

  void pack(T* block) const override { derivatives::pack<Dim, out_rank>(value, block); }

  void const* object() const noexcept override { return &value; }

  T asymmetry() const override {
    if constexpr (std::is_arithmetic_v<decltype(value)>) {
      return T{0};
    } else {
      std::vector<T> block(this->rows() * this->cols());
      pack(block.data());
      auto symmetric{std::make_unique<std::remove_cvref_t<decltype(value)>>()};
      derivatives::unpack<Dim, out_rank>(block.data(), *symmetric);
      T const size{tmech::norm(value)};
      return size > T{0} ? tmech::norm(value - *symmetric) / size : T{0};
    }
  }

  T const* scalar() const noexcept override {
    if constexpr (std::is_same_v<decltype(value), T>)
      return &value;
    else
      return nullptr;
  }

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

/// A value of a plan, d node / d seed, held as its tmech type; a requested derivative is the host's view of one.
template <typename T>
class value_holder {
public:
  virtual ~value_holder() = default;
  /// The tmech value, for the kernels.
  virtual void* object() noexcept = 0;
  /// Its storage in full notation, for the Mandel boundary of an LU.
  virtual T* data() noexcept = 0;
  virtual void fill(T v) noexcept = 0;
  void zero() noexcept { fill(T{0}); }
};

template <typename T, typename V>
class typed_value final : public value_holder<T> {
public:
  V value{};

  typed_value() noexcept { this->zero(); }
  void* object() noexcept override { return &value; }
  T* data() noexcept override {
    if constexpr (std::is_arithmetic_v<V>)
      return &value;
    else
      return value.raw_data();
  }
  void fill(T v) noexcept override {
    if constexpr (std::is_arithmetic_v<V>)
      value = v;
    else
      value.fill(v);
  }
};

template <typename T>
using derivative_slot_base = value_holder<T>;
template <typename T, std::size_t, typename R>
using derivative_slot = typed_value<T, R>;

} // namespace numsim::materials::derivatives

#endif // NUMSIM_MATERIALS_CORE_DERIVATIVES_H
