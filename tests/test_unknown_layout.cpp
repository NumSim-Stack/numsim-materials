#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/property.h"
#include "numsim-materials/core/input_types.h"
#include "numsim-materials/core/property_traits.h"
#include "numsim-materials/solvers/unknown_layout.h"

namespace {

using namespace numsim::materials;

using T = double;
constexpr std::size_t Dim = 3;
using tensor2 = tmech::tensor<T, Dim, 2>;
using tensor4 = tmech::tensor<T, Dim, 4>;

using scalar_k = scalar_unknown<T>;
using sym_k    = sym_tensor_unknown<T, Dim>;

// Round-trip and isometry are covered upstream in tmech's own suite; what is
// tested here is the serialization into a strided column-major buffer.

/// Wire a standalone property to an input_property so a layout can bind to it
/// without standing up a whole material graph.
template<typename V>
struct wired_input {
  property_traits traits{};
  std::unique_ptr<property<V>> prop;
  input_property<V, property_traits> in{"owner", "name"};

  explicit wired_input(const V& value)
      : prop(make_property<V, property_traits>(value, traits)) {
    in.wire(*prop);
  }
};

tensor2 make_sym(T a, T b, T c, T d, T e, T f) {
  return tensor2{a, d, e,
                 d, b, f,
                 e, f, c};
}

// --- widths and offsets ----------------------------------------------------

TEST(UnknownLayout, WidthsAndAccumulatedOffsets) {
  T dgamma{0};
  tensor2 beta{};
  state_layout<T, scalar_k> s0(dgamma);
  state_layout<T, sym_k>    s1(beta);

  EXPECT_EQ(s0.rows(), 1u);
  EXPECT_EQ(s1.rows(), 6u);
  EXPECT_EQ(s0.cols(), 1u);
  EXPECT_EQ(s1.cols(), 1u);

  std::vector<state_layout_base<T>*> state{&s0, &s1};
  std::size_t N = 0;
  std::vector<std::size_t> offset;
  for (auto* s : state) { offset.push_back(N); N += s->rows(); }

  EXPECT_EQ(N, 7u) << "1 scalar + 1 symmetric 3D tensor";
  EXPECT_EQ(offset[0], 0u);
  EXPECT_EQ(offset[1], 1u);
}

// --- scatter/gather round-trip through the flat buffer ---------------------

TEST(UnknownLayout, ScatterGatherRoundTripsMixedState) {
  T dgamma{0};
  tensor2 beta{};
  state_layout<T, scalar_k> s0(dgamma);
  state_layout<T, sym_k>    s1(beta);

  const T flat_in[7]{0.375,
                     1.1, 2.2, 3.3,                 // 11 22 33
                     0.565685424949238,             // sqrt2 * 0.4  (23)
                     0.707106781186548,             // sqrt2 * 0.5  (13)
                     0.848528137423857};            // sqrt2 * 0.6  (12)

  s0.scatter(flat_in, 0);
  s1.scatter(flat_in, 1);

  EXPECT_DOUBLE_EQ(dgamma, 0.375);
  EXPECT_NEAR(beta(0, 0), 1.1, 1e-14);
  EXPECT_NEAR(beta(1, 2), 0.4, 1e-14) << "off-diagonals carry the sqrt(2) weight";
  EXPECT_NEAR(beta(0, 1), 0.6, 1e-14);

  T flat_out[7]{};
  s0.gather(flat_out, 0, 0, 7);
  s1.gather(flat_out, 1, 0, 7);
  for (int i = 0; i < 7; ++i)
    EXPECT_NEAR(flat_out[i], flat_in[i], 1e-14) << "component " << i;
}

// --- isometry across the bridge -------------------------------------------

TEST(UnknownLayout, PackingPreservesTheInnerProduct) {
  const auto A = make_sym(1.1, 2.2, 3.3, 0.6, 0.5, 0.4);
  const auto B = make_sym(2.0, -1.0, 0.7, -0.3, 1.2, -0.8);

  tensor2 ta = A, tb = B;
  state_layout<T, sym_k> la(ta), lb(tb);

  T ma[6]{}, mb[6]{};
  la.gather(ma, 0, 0, 6);
  lb.gather(mb, 0, 0, 6);

  T dot = 0;
  for (int i = 0; i < 6; ++i) dot += ma[i] * mb[i];
  EXPECT_NEAR(dot, tmech::dcontract(A, B), 1e-12)
      << "A:B must equal mandel(A).mandel(B) — this is what Voigt fails";
}

// --- rank-4 block: no transpose, at a non-zero offset and stride -----------

TEST(UnknownLayout, MajorAsymmetricBlockLandsUntransposed) {
  // Minor-symmetric, major-ASYMMETRIC: exactly the (C:N)(x)(M:C) structure of a
  // non-associative algorithmic tangent. A row/column-major slip is invisible
  // for a major-symmetric block and wrong here.
  const auto P = make_sym(1.1, 2.2, 3.3, 0.6, 0.5, 0.4);
  const auto Q = make_sym(2.0, -1.0, 0.7, -0.3, 1.2, -0.8);
  const tensor4 D = tmech::otimes(P, Q);

  wired_input<tensor4> w(D);
  jacobian_block_layout<T, sym_k, sym_k> block(w.in);
  EXPECT_EQ(block.rows(), 6u);
  EXPECT_EQ(block.cols(), 6u);

  // Place it inside a larger column-major system at (1,1) with ld = 7.
  constexpr std::size_t N = 7;
  T J[N * N]{};
  block.gather(J, 1, 1, N);

  // Reference: the same tensor packed straight to a contiguous row-major 6x6.
  T ref[36]{};
  tmech::convert_tensor_to_mandel(D, ref);

  for (std::size_t i = 0; i < 6; ++i)
    for (std::size_t j = 0; j < 6; ++j)
      EXPECT_NEAR(J[(1 + i) + (1 + j) * N], ref[i * 6 + j], 1e-14)
          << "block entry (" << i << "," << j << ")";

  ASSERT_NE(ref[1], ref[6]) << "test tensor must be major-asymmetric to be a real guard";
}

TEST(UnknownLayout, Rank4BlockReproducesTheTensorAction) {
  const auto P = make_sym(1.1, 2.2, 3.3, 0.6, 0.5, 0.4);
  const auto Q = make_sym(2.0, -1.0, 0.7, -0.3, 1.2, -0.8);
  const tensor4 D = tmech::otimes(P, Q);
  const auto X = make_sym(0.4, 0.9, 1.4, 0.1, -0.2, 0.3);

  wired_input<tensor4> w(D);
  jacobian_block_layout<T, sym_k, sym_k> block(w.in);
  T M[36]{};
  block.gather(M, 0, 0, 6);

  tensor2 tx = X;
  state_layout<T, sym_k> lx(tx);
  T mx[6]{};
  lx.gather(mx, 0, 0, 6);

  // M is column-major here, so index (i,j) is M[i + j*6].
  T my[6]{};
  for (std::size_t i = 0; i < 6; ++i) {
    T s = 0;
    for (std::size_t j = 0; j < 6; ++j) s += M[i + j * 6] * mx[j];
    my[i] = s;
  }

  tensor2 y{};
  state_layout<T, sym_k> ly(y);
  ly.scatter(my, 0);

  const tensor2 y_ref = tmech::dcontract(D, X);
  for (std::size_t i = 0; i < Dim; ++i)
    for (std::size_t j = 0; j < Dim; ++j)
      EXPECT_NEAR(y(i, j), y_ref(i, j), 1e-12)
          << "D:X must equal unmandel(M * mandel(X)) at (" << i << "," << j << ")";
}

// --- row vs column orientation from the SAME tensor type -------------------

TEST(UnknownLayout, ScalarTensorBlocksAreTransposedPlacements) {
  const auto G = make_sym(1.1, 2.2, 3.3, 0.6, 0.5, 0.4);

  wired_input<tensor2> wr(G);
  wired_input<tensor2> wc(G);
  jacobian_block_layout<T, scalar_k, sym_k> row(wr.in);   // 1x6
  jacobian_block_layout<T, sym_k, scalar_k> col(wc.in);   // 6x1

  EXPECT_EQ(row.rows(), 1u);
  EXPECT_EQ(row.cols(), 6u);
  EXPECT_EQ(col.rows(), 6u);
  EXPECT_EQ(col.cols(), 1u);

  constexpr std::size_t N = 7;
  T J[N * N]{};
  row.gather(J, 0, 1, N);   // residual row 0, columns 1..6
  col.gather(J, 1, 0, N);   // rows 1..6, column 0

  T ref[6]{};
  tmech::convert_tensor_to_mandel(G, ref);

  for (std::size_t k = 0; k < 6; ++k) {
    EXPECT_NEAR(J[0 + (1 + k) * N], ref[k], 1e-14) << "row entry " << k;
    EXPECT_NEAR(J[(1 + k) + 0 * N], ref[k], 1e-14) << "column entry " << k;
  }
}

// --- scalar/scalar block ---------------------------------------------------

TEST(UnknownLayout, ScalarScalarBlockIsOneByOne) {
  wired_input<T> w(2.75);
  jacobian_block_layout<T, scalar_k, scalar_k> block(w.in);
  EXPECT_EQ(block.rows(), 1u);
  EXPECT_EQ(block.cols(), 1u);

  constexpr std::size_t N = 7;
  T J[N * N]{};
  block.gather(J, 3, 4, N);
  EXPECT_DOUBLE_EQ(J[3 + 4 * N], 2.75);
}

// --- residual layout -------------------------------------------------------

TEST(UnknownLayout, ResidualLayoutPacksIntoASegment) {
  const auto R = make_sym(1.1, 2.2, 3.3, 0.6, 0.5, 0.4);
  wired_input<tensor2> w(R);
  residual_layout<T, sym_k> res(w.in);

  EXPECT_EQ(res.rows(), 6u);
  EXPECT_EQ(res.cols(), 1u);

  T flat[7]{};
  res.gather(flat, 1, 0, 7);

  T ref[6]{};
  tmech::convert_tensor_to_mandel(R, ref);
  for (int k = 0; k < 6; ++k) EXPECT_NEAR(flat[1 + k], ref[k], 1e-14);
  EXPECT_DOUBLE_EQ(flat[0], 0.0) << "must not write outside its own segment";
}

} // namespace
