#ifndef NUMSIM_MATERIALS_UNKNOWN_LAYOUT_H
#define NUMSIM_MATERIALS_UNKNOWN_LAYOUT_H

#include <cstddef>
#include <string>
#include <type_traits>
#include <Eigen/Dense>
#include <tmech/tmech.h>
#include "numsim-materials/core/input_types.h"
#include "numsim-materials/core/property_traits.h"
#include "numsim-materials/core/unknown_spec.h"

/// Serialization layer between tmech tensors and the flat Eigen system solved
/// by vector_newton.
///
/// Each piece of the local system — every unknown, every residual entry, every
/// Jacobian block — carries its own layout object. The solver holds a vector of
/// layouts for the state, a vector for the residual, and a matrix for the
/// Jacobian, and knows only the abstract base. The concrete layout is a
/// compile-time type that binds to one property's data and packs it.
///
/// Packing is Mandel, not Voigt, because the flat solve must be equivalent to
/// the tensor system:
///
///     A : B          == mandel(A) . mandel(B)
///     (D : X)        == mandel4(D) * mandel2(X)
///
/// The sqrt(2) weights on the off-diagonal slots make the map an isometry, so
/// the LU solve and the infinity-norm convergence test carry over unchanged.
/// Applying the *same* weights uniformly to residual, unknown, and both sides
/// of every Jacobian block is what makes this work; engineering Voigt would
/// need compensating factors of 2 scattered through the tangent.
namespace numsim::materials {

// ---------------------------------------------------------------------------
// Unknown kinds
// ---------------------------------------------------------------------------

/// An unknown's symmetry cannot be recovered from its C++ type: tmech's
/// tensor<T,Dim,2> is symmetry-agnostic storage (even inv() takes a symmetry
/// sequence at the call site), and property_traits carries no shape metadata.
/// So the kind is declared explicitly as a tag.

template<typename T>
struct scalar_unknown {
  using value_type = T;
  static constexpr std::size_t rank  = 0;
  static constexpr std::size_t width = 1;
};

template<typename T, std::size_t Dim>
struct sym_tensor_unknown {
  using value_type = tmech::tensor<T, Dim, 2>;
  static constexpr std::size_t rank  = 2;
  static constexpr std::size_t width = (Dim == 2 ? 3 : 6);
  static constexpr std::size_t dim   = Dim;
};

// The runtime counterparts (unknown_kind, unknown_spec, block_ref) live in
// core/unknown_spec.h so the JSON layer can read them without pulling in tmech.

namespace layout_detail {

template<typename U> struct is_unknown_kind : std::false_type {};
template<typename T> struct is_unknown_kind<scalar_unknown<T>> : std::true_type {};
template<typename T, std::size_t D>
struct is_unknown_kind<sym_tensor_unknown<T, D>> : std::true_type {};

/// Dimension of a kind pair — scalars carry none, so take it from whichever
/// side is a tensor. A scalar/scalar block never needs a dimension.
template<typename UI, typename UJ> struct pair_dim
    : std::integral_constant<std::size_t, UI::dim> {};
template<typename T, typename UJ> struct pair_dim<scalar_unknown<T>, UJ>
    : std::integral_constant<std::size_t, UJ::dim> {};
template<typename UI, typename T> struct pair_dim<UI, scalar_unknown<T>>
    : std::integral_constant<std::size_t, UI::dim> {};
template<typename T1, typename T2>
struct pair_dim<scalar_unknown<T1>, scalar_unknown<T2>>
    : std::integral_constant<std::size_t, 0> {};

/// Value type of a Jacobian block: rank is the sum of the two unknowns' ranks.
template<typename T, std::size_t Dim, std::size_t Rank>
struct block_value { using type = tmech::tensor<T, Dim, Rank>; };
template<typename T, std::size_t Dim>
struct block_value<T, Dim, 0> { using type = T; };

template<typename T, typename UI, typename UJ>
using block_value_t =
    typename block_value<T, pair_dim<UI, UJ>::value, UI::rank + UJ::rank>::type;

/// Pack a tmech tensor into contiguous Mandel storage.
/// rank 2 -> `width` components; rank 4 -> width*width, ROW-MAJOR (ptr[i*W+j]).
template<typename Tensor, typename T>
inline void pack(const Tensor& t, T* dst) {
  tmech::convert_tensor_to_mandel(t, dst);
}

/// Unpack contiguous Mandel storage back into a rank-2 tmech tensor.
template<typename T, std::size_t Dim>
inline void unpack2(const T* src, tmech::tensor<T, Dim, 2>& out) {
  const tmech::adaptor<const T, Dim, 2, tmech::mandel<Dim>> view(src);
  out = view;
}

/// Unpack a contiguous ROW-MAJOR Mandel matrix into a rank-4 tmech tensor.
/// The inverse of pack() for rank 4; the result is minor-symmetric by
/// construction, since that is all Mandel's 36-of-81 storage can represent.
template<typename T, std::size_t Dim>
inline void unpack4(const T* src, tmech::tensor<T, Dim, 4>& out) {
  const tmech::adaptor<const T, Dim, 4, tmech::mandel<Dim>> view(src);
  out = view;
}

/// Strided Eigen views onto a sub-block of a column-major buffer, used instead
/// of hand-written index arithmetic so the row-major (tmech) to column-major
/// (Eigen) conversion is declarative.
///
/// A single ROW must be viewed as a vector with INNER stride ld, not as a 1 x C
/// matrix with outer stride: Eigen treats a fixed-size Matrix<T,1,C> as a vector
/// at compile time, linearizes it with inner stride 1, and silently ignores the
/// outer stride — writing the row contiguously instead of across the buffer.
template<typename T, std::size_t C, typename Ptr>
inline auto row_view(Ptr* base, std::size_t ld) {
  using scalar_t = std::remove_const_t<Ptr>;
  using vec_t = Eigen::Matrix<scalar_t, int(C), 1>;
  using mapped_t =
      std::conditional_t<std::is_const_v<Ptr>, const vec_t, vec_t>;
  return Eigen::Map<mapped_t, Eigen::Unaligned, Eigen::InnerStride<>>(
      base, Eigen::InnerStride<>(static_cast<Eigen::Index>(ld)));
}

/// A genuine two-dimensional block (R, C both > 1), where outer stride is the
/// correct notion.
template<typename T, std::size_t R, std::size_t C, typename Ptr>
inline auto block_view(Ptr* base, std::size_t ld) {
  using scalar_t = std::remove_const_t<Ptr>;
  using mat_t = Eigen::Matrix<scalar_t, int(R), int(C)>;
  using mapped_t =
      std::conditional_t<std::is_const_v<Ptr>, const mat_t, mat_t>;
  return Eigen::Map<mapped_t, Eigen::Unaligned, Eigen::OuterStride<>>(
      base, Eigen::OuterStride<>(static_cast<Eigen::Index>(ld)));
}

} // namespace layout_detail

// ---------------------------------------------------------------------------
// Abstract bases — all the solver knows
// ---------------------------------------------------------------------------

/// Serializes one bound piece of the system into the flat column-major buffer.
template<typename T>
class layout_base {
public:
  virtual ~layout_base() = default;

  virtual std::size_t rows() const noexcept = 0;
  virtual std::size_t cols() const noexcept = 0;

  /// Write the bound data into `dst` at (r0, c0) of a column-major buffer
  /// whose leading dimension (column stride) is `ld`.
  virtual void gather(T* dst, std::size_t r0, std::size_t c0,
                      std::size_t ld) const = 0;

  /// Re-evaluate the producing material, if this piece is graph-driven.
  /// A no-op for pieces the solver owns.
  virtual void update_source() const {}
};

/// A state piece additionally accepts the trial iterate back from the solver.
template<typename T>
class state_layout_base : public layout_base<T> {
public:
  /// Read `rows()` components starting at `src[r0]` into the bound unknown.
  virtual void scatter(const T* src, std::size_t r0) = 0;
};

// ---------------------------------------------------------------------------
// State layouts — bound to the solver's OWN output properties
// ---------------------------------------------------------------------------

/// The solver produces the unknowns and consumes residual/Jacobian, mirroring
/// backward_euler's graph mode (it produces "delta", the function material
/// reads it). So a state layout binds to a mutable output reference, while
/// residual and block layouts bind to read-only inputs.
template<typename T, typename U>
class state_layout final : public state_layout_base<T> {
  static_assert(layout_detail::is_unknown_kind<U>::value,
                "state_layout: U must be scalar_unknown or sym_tensor_unknown");

public:
  using value_type = typename U::value_type;

  explicit state_layout(value_type& ref) noexcept : m_ref(ref) {}

  std::size_t rows() const noexcept override { return U::width; }
  std::size_t cols() const noexcept override { return 1; }

  void gather(T* dst, std::size_t r0, std::size_t c0,
              std::size_t ld) const override {
    T* out = dst + r0 + c0 * ld;
    if constexpr (U::rank == 0)
      *out = m_ref;
    else
      layout_detail::pack(m_ref, out);   // segment is contiguous
  }

  void scatter(const T* src, std::size_t r0) override {
    if constexpr (U::rank == 0)
      m_ref = src[r0];
    else
      layout_detail::unpack2<T, U::dim>(src + r0, m_ref);
  }

private:
  value_type& m_ref;
};

// ---------------------------------------------------------------------------
// Residual layouts — bound to an input property
// ---------------------------------------------------------------------------

template<typename T, typename U>
class residual_layout final : public layout_base<T> {
  static_assert(layout_detail::is_unknown_kind<U>::value,
                "residual_layout: U must be scalar_unknown or sym_tensor_unknown");

public:
  using value_type = typename U::value_type;
  using input_type = input_property<value_type, property_traits>;

  explicit residual_layout(const input_type& in) noexcept : m_in(in) {}

  std::size_t rows() const noexcept override { return U::width; }
  std::size_t cols() const noexcept override { return 1; }

  void gather(T* dst, std::size_t r0, std::size_t c0,
              std::size_t ld) const override {
    T* out = dst + r0 + c0 * ld;
    if constexpr (U::rank == 0)
      *out = m_in.get();
    else
      layout_detail::pack(m_in.get(), out);
  }

  void update_source() const override { m_in.update_source(); }

private:
  const input_type& m_in;
};

// ---------------------------------------------------------------------------
// Jacobian block layouts — parameterized on the KIND PAIR
// ---------------------------------------------------------------------------

/// The pair, not the value type, because the value type is ambiguous: blocks
/// (scalar, sym3) and (sym3, scalar) are BOTH tensor<T,3,2>, but one packs as a
/// 1x6 row and the other as a 6x1 column.
///
/// Precondition for the rank-4 case: the block must be MINOR-symmetric. Mandel
/// stores 36 of the 81 rank-4 components, so a block without minor symmetry is
/// packed lossily and D:X != M*mandel(X). This holds by construction when both
/// unknowns are symmetric tensors. Major symmetry is NOT required — a
/// non-associative tangent is minor-symmetric and major-asymmetric, and packs
/// correctly.
template<typename T, typename UI, typename UJ>
class jacobian_block_layout final : public layout_base<T> {
  static_assert(layout_detail::is_unknown_kind<UI>::value &&
                    layout_detail::is_unknown_kind<UJ>::value,
                "jacobian_block_layout: kinds must be scalar_unknown or "
                "sym_tensor_unknown");

  static constexpr std::size_t R = UI::width;
  static constexpr std::size_t C = UJ::width;

public:
  using value_type = layout_detail::block_value_t<T, UI, UJ>;
  using input_type = input_property<value_type, property_traits>;

  explicit jacobian_block_layout(const input_type& in) noexcept : m_in(in) {}

  std::size_t rows() const noexcept override { return R; }
  std::size_t cols() const noexcept override { return C; }

  void gather(T* dst, std::size_t r0, std::size_t c0,
              std::size_t ld) const override {
    if constexpr (UI::rank == 0 && UJ::rank == 0) {
      // 1x1 — a single scalar, nothing to lay out.
      dst[r0 + c0 * ld] = m_in.get();
    } else if constexpr (UJ::rank == 0) {
      // Rx1 column: the destination slots are CONTIGUOUS, so tmech can pack
      // straight into them. No scratch, no copy.
      layout_detail::pack(m_in.get(), dst + r0 + c0 * ld);
    } else if constexpr (UI::rank == 0) {
      // 1xC row: pack with tmech, then let Eigen place it across the stride.
      T buf[C];
      layout_detail::pack(m_in.get(), buf);
      layout_detail::row_view<T, C>(dst + r0 + c0 * ld, ld) =
          Eigen::Map<const Eigen::Matrix<T, int(C), 1>>(buf);
    } else {
      // RxC: tmech packs rank-4 ROW-major, the destination is column-major with
      // stride ld. Expressing that as a Map-to-Map assignment lets Eigen do the
      // storage-order conversion, instead of hand-written index arithmetic —
      // which is precisely where a transpose slip would hide.
      T buf[R * C];
      layout_detail::pack(m_in.get(), buf);
      layout_detail::block_view<T, R, C>(dst + r0 + c0 * ld, ld) =
          Eigen::Map<const Eigen::Matrix<T, int(R), int(C), Eigen::RowMajor>>(buf);
    }
  }

  void update_source() const override { m_in.update_source(); }

private:
  const input_type& m_in;
};

// ---------------------------------------------------------------------------
// Strain-derivative layouts — the scatter direction
// ---------------------------------------------------------------------------

/// Writes a rows() x cols() block of a flat column-major buffer back into a
/// bound tensor. The inverse of layout_base::gather, and the piece the
/// implicit-function-theorem tangent needs: dx/deps comes out of the linear
/// solve as an N x W matrix and has to become per-unknown tensors again.
template<typename T>
class block_scatter_base {
public:
  virtual ~block_scatter_base() = default;
  virtual std::size_t rows() const noexcept = 0;
  virtual std::size_t cols() const noexcept = 0;
  virtual void scatter_block(const T* src, std::size_t r0, std::size_t c0,
                             std::size_t ld) = 0;
};

/// dx_I/deps for one unknown.
///
/// Its shape is that of a Jacobian block whose column kind is the (symmetric)
/// strain, so the rank is U::rank + 2: a scalar unknown gives a rank-2 tensor
/// laid along a 1 x W row, a symmetric-tensor unknown gives a rank-4 tensor in
/// a W x W block. Unlike jacobian_block_layout this binds to a MUTABLE output
/// reference, because the solver produces this quantity rather than consuming
/// it.
template<typename T, std::size_t Dim, typename U>
class strain_derivative_layout final : public block_scatter_base<T> {
  static_assert(layout_detail::is_unknown_kind<U>::value,
                "strain_derivative_layout: U must be scalar_unknown or "
                "sym_tensor_unknown");

  using strain_kind = sym_tensor_unknown<T, Dim>;
  static constexpr std::size_t R = U::width;
  static constexpr std::size_t C = strain_kind::width;

public:
  using value_type = layout_detail::block_value_t<T, U, strain_kind>;

  explicit strain_derivative_layout(value_type& ref) noexcept : m_ref(ref) {}

  std::size_t rows() const noexcept override { return R; }
  std::size_t cols() const noexcept override { return C; }

  void scatter_block(const T* src, std::size_t r0, std::size_t c0,
                     std::size_t ld) override {
    if constexpr (U::rank == 0) {
      // 1 x C row -> rank-2 tensor
      T buf[C];
      Eigen::Map<Eigen::Matrix<T, int(C), 1>> contiguous(buf);
      contiguous = layout_detail::row_view<T, C>(src + r0 + c0 * ld, ld);
      layout_detail::unpack2<T, Dim>(buf, m_ref);
    } else {
      // R x C block -> rank-4 tensor. tmech reads rank-4 Mandel storage
      // ROW-major and the source is column-major with stride ld, so the Map
      // assignment mirrors the conversion jacobian_block_layout does outbound.
      T buf[R * C];
      Eigen::Map<Eigen::Matrix<T, int(R), int(C), Eigen::RowMajor>> contiguous(buf);
      contiguous = layout_detail::block_view<T, R, C>(src + r0 + c0 * ld, ld);
      layout_detail::unpack4<T, Dim>(buf, m_ref);
    }
  }

private:
  value_type& m_ref;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_UNKNOWN_LAYOUT_H
