#include <gtest/gtest.h>
#include <pthread.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include <tmech/tmech.h>
#include "numsim-materials/core/derivative_kernels.h"
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/partial_check.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/driver/material_point.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/elastic_strain.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/thermal_eigenstrain.h"
#include "numsim-materials/umat/external_state_source.h"
#include "test_fixture_materials.h"
#include "test_support.h"

// Partial derivatives per callback and their forward accumulation (#217).

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using tensor6 = tmech::tensor<T, 3, 6>;
using tensor8 = tmech::tensor<T, 3, 8>;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using numsim::materials::test::add;

tensor2 const I{tmech::eye<T, 3, 2>()};
tensor4 const Is{nm::sym_identity<T, 3>()};

tensor2 sym(T a, T b, T c, T d, T e, T f) {
  tensor2 x;
  x(0, 0) = a, x(1, 1) = b, x(2, 2) = c;
  x(1, 2) = x(2, 1) = d, x(0, 2) = x(2, 0) = e, x(0, 1) = x(1, 0) = f;
  return x;
}
tensor2 const B2{sym(1, -2, 0.5, 0.3, -0.7, 1.1)};
tensor2 const C2{sym(0.4, 0.2, -1, 0.6, 0.1, -0.3)};
tensor2 const E2{sym(-0.5, 1.5, 0.25, -0.2, 0.9, 0.35)};
tensor4 const A4{tensor4{2 * Is + 0.5 * tmech::otimes(I, I) + tmech::otimes(B2, B2)}};
tensor4 const B4{tensor4{3 * Is - tmech::otimes(I, I) + 0.5 * tmech::otimes(C2, C2)}};

// One law per edge of the contraction table: value and its partial in tmech.
struct norm_trace { // scalar of tensor2
  using in = tensor2;
  using out = T;
  static T value(tensor2 const &x) { return tmech::dcontract(x, x) + tmech::trace(x); }
  static tensor2 partial(tensor2 const &x) { return 2 * x + I; }
};
struct square { // scalar of scalar
  using in = T;
  using out = T;
  static T value(T s) { return s * s; }
  static T partial(T s) { return 2 * s; }
};
struct scale_b { // tensor2 of scalar
  using in = T;
  using out = tensor2;
  static tensor2 value(T s) { return s * B2; }
  static tensor2 partial(T) { return B2; }
};
struct map_a { // tensor2 of tensor2
  using in = tensor2;
  using out = tensor2;
  static tensor2 value(tensor2 const &x) { return tmech::dcontract(A4, x); }
  static tensor4 partial(tensor2 const &) { return A4; }
};
struct map_b {
  using in = tensor2;
  using out = tensor2;
  static tensor2 value(tensor2 const &x) { return tmech::dcontract(B4, x); }
  static tensor4 partial(tensor2 const &) { return B4; }
};
struct project_c { // scalar of tensor2, linear
  using in = tensor2;
  using out = T;
  static T value(tensor2 const &x) { return tmech::dcontract(C2, x); }
  static tensor2 partial(tensor2 const &) { return C2; }
};
struct dyad { // tensor4 of tensor2: Y = tr(x) C (x) x
  using in = tensor2;
  using out = tensor4;
  static tensor4 value(tensor2 const &x) { return tmech::trace(x) * tmech::otimes(C2, x); }
  static tensor6 partial(tensor2 const &x) {
    return tensor6{tmech::trace(x) * tmech::otimes(C2, Is) + tmech::otimes(tensor4{tmech::otimes(C2, x)}, I)};
  }
};
struct contract_e { // tensor2 of tensor4: z = Y : E
  using in = tensor4;
  using out = tensor2;
  static tensor2 value(tensor4 const &Y) { return tmech::dcontract(Y, E2); }
  static tensor6 partial(tensor4 const &) { return tmech::otimes(Is, E2); }
};
struct constant_law { // tensor2 of tensor2, independent of its read
  using in = tensor2;
  using out = tensor2;
  static tensor2 value(tensor2 const &) { return B2; }
  static tensor4 partial(tensor2 const &) { return tensor4{}; }
};
// y = c tr(x)^3, about 1e8 at a strain of 1e-3; the partial is off by `error` (relative).
template <int Error>
struct cubic_strain {
  using in = tensor2;
  using out = T;
  static constexpr T c{1e19};
  static T value(tensor2 const &x) { return c * std::pow(tmech::trace(x), 3); }
  static tensor2 partial(tensor2 const &x) {
    return tensor2{(1 + Error * T{1e-6}) * 3 * c * std::pow(tmech::trace(x), 2) * I};
  }
};
struct throwing_map { // throws once the read leaves the state it was armed at
  using in = tensor2;
  using out = tensor2;
  static inline tensor2 armed{};
  static inline bool active{false};
  static tensor2 value(tensor2 const &x) {
    if (active && tmech::norm(tensor2{x - armed}) > 0)
      throw std::runtime_error("perturbed state");
    return tmech::dcontract(A4, x);
  }
  static tensor4 partial(tensor2 const &) { return A4; }
};
struct wrong_map { // declares A4 for B4 : x
  using in = tensor2;
  using out = tensor2;
  static tensor2 value(tensor2 const &x) { return tmech::dcontract(B4, x); }
  static tensor4 partial(tensor2 const &) { return A4; }
};

tensor2 const N2{0.9, 0.3, -0.2, 0.1, 1.2, 0.4, -0.5, 0.2, 0.8}; // not symmetric
struct congruence { // tensor2 of tensor2: y = N x N^T, partial N_ik N_jl, symmetric in neither index pair
  using in = tensor2;
  using out = tensor2;
  static tensor2 value(tensor2 const &x) { return tensor2{N2 * x * tmech::trans(N2)}; }
  static tensor4 partial(tensor2 const &) { return tensor4{tmech::otimesu(N2, N2)}; }
};
struct map_y { // tensor4 of tensor4: z = A : Y
  using in = tensor4;
  using out = tensor4;
  static tensor4 value(tensor4 const &Y) { return tmech::dcontract(A4, Y); }
  static tensor8 partial(tensor4 const &) { return tmech::outer_product<tmech::sequence<1, 2, 5, 6>, tmech::sequence<3, 4, 7, 8>>(A4, Is); }
};

/// y = Law::value(x) for x read from "x_source"; "partials" is "declared" (the default), "independent", "on_request"
/// (fill() fills it), "fill_only" (fill() without partials()) or "none"; "unsymmetric" declares y unsymmetric_output().
template <typename Law, typename Traits>
class law_node final : public nm::material_base<law_node<Law, Traits>, Traits> {
public:
  using base = nm::material_base<law_node<Law, Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit law_node(Args &&...args)
      : base(std::forward<Args>(args)...), m_y(base::template add_output<typename Law::out>("y")),
        m_source(nm::connection_source::parse(base::template get_parameter<std::string>("x_source"))),
        m_x(base::template add_input<typename Law::in>(m_source.material, m_source.property, nm::EdgeKind::Global)),
        m_mode(base::template get_parameter<std::string>("partials")) {
    auto handle{base::add_callback(&law_node::compute, m_y)};
    handle.reads(m_x);
    if (m_mode == "declared" || m_mode == "on_request")
      handle.partials(m_y, m_x);
    else if (m_mode == "independent")
      handle.independent_of(m_x);
    if (m_mode == "on_request" || m_mode == "fill_only")
      handle.fills_partials(&law_node::fill);
    if constexpr (!std::is_arithmetic_v<typename Law::out>)
      if (base::template get_parameter<bool>("unsymmetric"))
        base::unsymmetric_output(m_y);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<std::string>(para, "x_source");
    nm::optional<std::string>(para, "partials", std::string{"declared"});
    nm::optional<bool>(para, "unsymmetric", false);
    return para;
  }

  static inline std::function<void()> hook; ///< called by every run, when set

  void compute() {
    ++m_runs;
    if (hook)
      hook();
    m_y = Law::value(m_x.get());
    if (base::want_partials()) {
      base::partial(m_y, m_x) = Law::partial(m_x.get());
      ++m_fills;
    }
  }

  /// "on_request": the partial, at the value compute() left.
  void fill() {
    if (base::want_partial(m_y, m_x)) {
      base::partial(m_y, m_x) = Law::partial(m_x.get());
      ++m_fills;
    }
  }

  int runs() const noexcept { return m_runs; }
  int fills() const noexcept { return m_fills; }

private:
  typename Law::out &m_y;
  nm::connection_source const m_source;
  nm::input_property<typename Law::in, nm::property_traits> const &m_x;
  std::string const &m_mode;
  int m_runs{0};
  int m_fills{0};
};

/// Declares its partials wrongly, as "mistake" says.
template <typename Traits>
class misdeclared final : public nm::material_base<misdeclared<Traits>, Traits> {
public:
  using base = nm::material_base<misdeclared<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit misdeclared(Args &&...args)
      : base(std::forward<Args>(args)...), m_y(base::template add_output<tensor2>("y")),
        m_z(base::template add_output<tensor2>("z")),
        m_x(base::template add_input<tensor2>("strain", "strain", nm::EdgeKind::Global)),
        m_w(base::template add_input<T>("temperature", "state", nm::EdgeKind::Global)) {
    auto const &mistake{base::template get_parameter<std::string>("mistake")};
    base::add_callback(&misdeclared::compute_z, m_z).reads(m_x);
    auto handle{base::add_callback(&misdeclared::compute, m_y)};
    if (mistake == "before_reads")
      handle.partials(m_y, m_x);
    handle.reads(m_x);
    if (mistake == "unread")
      handle.partials(m_y, m_w);
    else if (mistake == "twice")
      handle.partials(m_y, m_x).partials(m_y, m_x);
    else if (mistake == "both")
      handle.partials(m_y, m_x).independent_of(m_x);
    else if (mistake == "other_output")
      handle.partials(m_z, m_x);
    else if (mistake == "handle_undeclared")
      std::ignore = base::partial_of(m_y, m_x);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<std::string>(para, "mistake");
    return para;
  }

  void compute() { m_y = m_x.get(); }
  void compute_z() { m_z = m_x.get(); }

private:
  tensor2 &m_y;
  tensor2 &m_z;
  nm::input_property<tensor2, nm::property_traits> const &m_x;
  nm::input_property<T, nm::property_traits> const &m_w;
};

/// y = x + w I with partials on request through handles; counts the blocks each fill wrote.
template <typename Traits>
class two_reads final : public nm::material_base<two_reads<Traits>, Traits> {
public:
  using base = nm::material_base<two_reads<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit two_reads(Args &&...args)
      : base(std::forward<Args>(args)...), m_y(base::template add_output<tensor2>("y")),
        m_x(base::template add_input<tensor2>("strain", "strain", nm::EdgeKind::Global)),
        m_w(base::template add_input<T>("temperature", "state", nm::EdgeKind::Global)) {
    base::add_callback(&two_reads::compute, m_y).reads(m_x, m_w).partials(m_y, m_x, m_w).fills_partials(&two_reads::fill);
    m_dy_dx = base::partial_of(m_y, m_x);
    m_dy_dw = base::partial_of(m_y, m_w);
  }

  static input_parameter_controller parameters() { return base::parameters(); }

  void compute() { m_y = m_x.get() + m_w.get() * I; }

  void fill() {
    if (m_dy_dx.wanted()) {
      m_dy_dx.value() = Is;
      ++m_x_fills;
    }
    if (m_dy_dw.wanted()) {
      m_dy_dw.value() = I;
      ++m_w_fills;
    }
  }

  int x_fills() const noexcept { return m_x_fills; }
  int w_fills() const noexcept { return m_w_fills; }

private:
  tensor2 &m_y;
  nm::input_property<tensor2, nm::property_traits> const &m_x;
  nm::input_property<T, nm::property_traits> const &m_w;
  typename base::template partial_handle<tensor2, tensor2> m_dy_dx;
  typename base::template partial_handle<tensor2, T> m_dy_dw;
  int m_x_fills{0};
  int m_w_fills{0};
};

/// y = a::y + b::y, both read and differentiated.
template <typename Traits>
class sum_node final : public nm::material_base<sum_node<Traits>, Traits> {
public:
  using base = nm::material_base<sum_node<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit sum_node(Args &&...args)
      : base(std::forward<Args>(args)...), m_y(base::template add_output<tensor2>("y")),
        m_a(base::template add_input<tensor2>("a", "y", nm::EdgeKind::Global)),
        m_b(base::template add_input<tensor2>("b", "y", nm::EdgeKind::Global)) {
    base::add_callback(&sum_node::compute, m_y).reads(m_a, m_b).partials(m_y, m_a, m_b);
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() {
    m_y = m_a.get() + m_b.get();
    if (base::want_partials()) {
      base::partial(m_y, m_a) = Is;
      base::partial(m_y, m_b) = Is;
    }
  }

private:
  tensor2 &m_y;
  nm::input_property<tensor2, nm::property_traits> const &m_a;
  nm::input_property<tensor2, nm::property_traits> const &m_b;
};

/// y = (a::y : a::y) b::y: the partial with respect to b depends on a.
template <typename Traits>
class product_node final : public nm::material_base<product_node<Traits>, Traits> {
public:
  using base = nm::material_base<product_node<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit product_node(Args &&...args)
      : base(std::forward<Args>(args)...), m_y(base::template add_output<tensor2>("y")),
        m_a(base::template add_input<tensor2>("a", "y", nm::EdgeKind::Global)),
        m_b(base::template add_input<tensor2>("b", "y", nm::EdgeKind::Global)) {
    base::add_callback(&product_node::compute, m_y).reads(m_a, m_b).partials(m_y, m_a, m_b);
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() {
    auto const aa{tmech::dcontract(m_a.get(), m_a.get())};
    m_y = aa * m_b.get();
    if (base::want_partials()) {
      base::partial(m_y, m_a) = tensor4{T{2} * tmech::otimes(m_b.get(), m_a.get())};
      base::partial(m_y, m_b) = tensor4{aa * Is};
    }
  }

private:
  tensor2 &m_y;
  nm::input_property<tensor2, nm::property_traits> const &m_a;
  nm::input_property<tensor2, nm::property_traits> const &m_b;
};

/// A context with the sources "strain" and "temperature" and law nodes added by node().
struct chain {
  std::unique_ptr<ctx_type> owner{std::make_unique<ctx_type>()};
  ctx_type &ctx{*owner};

  chain() {
    add<nm::external_strain_source<policy>>(ctx, "strain");
    add<nm::external_scalar_source<policy>>(ctx, "temperature");
  }

  template <typename Law>
  law_node<Law, policy> &node(std::string const &name, std::string const &x_source, std::string const &partials = "declared") {
    return add<law_node<Law, policy>>(ctx, name, {{"x_source", x_source}, {"partials", partials}});
  }

  void load(tensor2 const &strain, T temperature) {
    dynamic_cast<nm::external_strain_source<policy> *>(ctx.find("strain"))->bind(tensor2{}, strain);
    dynamic_cast<nm::external_scalar_source<policy> *>(ctx.find("temperature"))->bind(T{0}, temperature);
    ctx.update();
  }
};

tensor2 const x0{sym(0.3, -0.1, 0.2, 0.05, -0.15, 0.25)};
constexpr T s0{1.7};

template <typename A, typename B>
T distance(A const &a, B const &b) {
  if constexpr (std::is_arithmetic_v<A>)
    return std::abs(a - b);
  else
    return tmech::norm(a - b);
}

// --- the contraction table, one hand-made chain per row ---

TEST(PartialDerivatives, ScalarOfScalarOfTensor) {
  chain c;
  c.node<norm_trace>("y", "strain::strain");
  c.node<square>("z", "y::y");
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor2>("z", "y", "strain", "strain")};
  c.load(x0, s0);
  tensor2 const expected{2 * norm_trace::value(x0) * norm_trace::partial(x0)};
  EXPECT_LT(distance(d, expected), 1e-12);
}

TEST(PartialDerivatives, TensorOfScalarOfTensor) {
  chain c;
  c.node<norm_trace>("y", "strain::strain");
  c.node<scale_b>("z", "y::y");
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor4>("z", "y", "strain", "strain")};
  c.load(x0, s0);
  EXPECT_LT(distance(d, tensor4{tmech::otimes(B2, norm_trace::partial(x0))}), 1e-12);
}

TEST(PartialDerivatives, TensorOfTensorOfTensor) {
  chain c;
  c.node<map_a>("y", "strain::strain");
  c.node<map_b>("z", "y::y");
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor4>("z", "y", "strain", "strain")};
  c.load(x0, s0);
  EXPECT_LT(distance(d, tensor4{tmech::dcontract(B4, A4)}), 1e-12);
}

TEST(PartialDerivatives, ScalarOfTensorOfTensor) {
  chain c;
  c.node<map_a>("y", "strain::strain");
  c.node<project_c>("z", "y::y");
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor2>("z", "y", "strain", "strain")};
  c.load(x0, s0);
  EXPECT_LT(distance(d, tensor2{tmech::dcontract(C2, A4)}), 1e-12);
}

TEST(PartialDerivatives, TensorOfTensorOfScalar) {
  chain c;
  c.node<scale_b>("y", "temperature::state");
  c.node<map_a>("z", "y::y");
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor2>("z", "y", "temperature", "state")};
  c.load(x0, s0);
  EXPECT_LT(distance(d, tensor2{tmech::dcontract(A4, B2)}), 1e-12);
}

TEST(PartialDerivatives, TensorOfFourthOrderTensorOfTensor) {
  chain c;
  c.node<dyad>("y", "strain::strain");
  c.node<contract_e>("z", "y::y");
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor4>("z", "y", "strain", "strain")};
  c.load(x0, s0);
  // z = tr(x) (x : E) C
  tensor4 const expected{tmech::trace(x0) * tmech::otimes(C2, E2) + tmech::dcontract(x0, E2) * tmech::otimes(C2, I)};
  EXPECT_LT(distance(d, expected), 1e-12);
}

TEST(PartialDerivatives, ScalarOfScalarIsAScalar) {
  chain c;
  c.node<square>("y", "temperature::state");
  c.node<square>("z", "y::y");
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<T>("z", "y", "temperature", "state")};
  c.load(x0, s0);
  EXPECT_NEAR(d, 4 * s0 * s0 * s0, 1e-12); // z = s^4
}

TEST(PartialDerivatives, ParallelPathsAreSummed) {
  chain c;
  c.node<map_a>("a", "strain::strain");
  c.node<map_b>("b", "strain::strain");
  add<sum_node<policy>>(c.ctx, "s");
  c.node<project_c>("z", "s::y");
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor2>("z", "y", "strain", "strain")};
  c.load(x0, s0);
  EXPECT_LT(distance(d, tensor2{tmech::dcontract(C2, tensor4{A4 + B4})}), 1e-12);
}

TEST(PartialDerivatives, NoPathGivesZero) {
  chain c;
  c.node<map_a>("y", "strain::strain");
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor2>("y", "y", "temperature", "state")};
  c.load(x0, s0);
  EXPECT_EQ(tmech::norm(d), 0);
}

TEST(PartialDerivatives, TheResultFollowsTheState) {
  chain c;
  c.node<norm_trace>("y", "strain::strain");
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor2>("y", "y", "strain", "strain")};
  for (T const f : {1.0, -2.0, 0.5}) {
    c.load(tensor2{f * x0}, s0);
    EXPECT_LT(distance(d, norm_trace::partial(tensor2{f * x0})), 1e-12);
  }
}

// --- declarations and their errors ---

TEST(PartialDerivatives, AMissingPartialOnARequestedPathThrowsAndNamesThePath) {
  chain c;
  c.node<map_a>("a", "strain::strain");
  c.node<map_b>("b", "a::y", "none");
  c.node<map_a>("c", "b::y");
  c.ctx.finalize();
  try {
    c.ctx.derivative<tensor4>("c", "y", "strain", "strain");
    FAIL() << "a requested path without a partial was accepted";
  } catch (std::logic_error const &e) {
    std::string const what{e.what()};
    EXPECT_TRUE(what.contains("'b::y'")) << what;
    EXPECT_TRUE(what.contains("strain::strain -> a::y -> b::y -> c::y")) << what;
  }
}

TEST(PartialDerivatives, AMissingPartialOffTheRequestedPathsDoesNotThrow) {
  chain c;
  c.node<map_a>("a", "strain::strain");
  c.node<map_b>("b", "a::y", "none");
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor4>("a", "y", "strain", "strain")};
  EXPECT_NO_THROW(c.load(x0, s0));
  EXPECT_LT(distance(d, A4), 1e-12);
}

TEST(PartialDerivatives, IndependentOfDeclaresAZeroPartial) {
  chain c;
  c.node<map_a>("a", "strain::strain");
  c.node<map_b>("b", "a::y", "independent");
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor4>("b", "y", "strain", "strain")};
  c.load(x0, s0);
  EXPECT_EQ(tmech::norm(d), 0);
}

TEST(PartialDerivatives, TheResultTypeMustMatchTheRanks) {
  chain c;
  c.node<map_a>("a", "strain::strain");
  c.ctx.finalize();
  EXPECT_THROW(c.ctx.derivative<tensor2>("a", "y", "strain", "strain"), std::logic_error);
  EXPECT_THROW(c.ctx.derivative<tensor4>("a", "y", "strain", "nothing"), std::runtime_error);
}

TEST(PartialDerivatives, PartialsAreDeclaredForReadsAndOutputsOfTheCallback) {
  for (auto const *mistake : {"unread", "twice", "both", "other_output", "before_reads", "handle_undeclared"}) {
    chain c;
    EXPECT_THROW((add<misdeclared<policy>>(c.ctx, "m", {{"mistake", mistake}})), std::logic_error) << mistake;
  }
}

// --- independent_of() is verified, not trusted (review of #227) ---

TEST(PartialDerivatives, AWrongIndependentOfFailsTheFiniteDifferenceCheck) {
  chain c;
  c.node<map_a>("a", "strain::strain");
  c.node<map_b>("liar", "a::y", "independent"); // B4 : x does depend on x
  c.ctx.finalize();
  c.load(x0, s0);
  auto const mismatches{nm::check_partials(c.ctx, "liar")};
  ASSERT_FALSE(mismatches.empty());
  EXPECT_TRUE(mismatches.front().contains("liar::y")) << mismatches.front();
  EXPECT_TRUE(mismatches.front().contains("a::y")) << mismatches.front();
  EXPECT_TRUE(mismatches.front().contains("independent_of")) << mismatches.front();
  EXPECT_LT(distance(c.ctx.get<tensor2>("liar", "y"), map_b::value(c.ctx.get<tensor2>("a", "y"))), 1e-15) << "the state is restored";
}

TEST(PartialDerivatives, ATrueIndependentOfPassesTheFiniteDifferenceCheck) {
  chain c;
  c.node<map_a>("a", "strain::strain");
  c.node<constant_law>("k", "a::y", "independent");
  c.ctx.finalize();
  c.load(x0, s0);
  EXPECT_TRUE(nm::check_partials(c.ctx, "k").empty());
}

// --- the step follows the read, the allowance the analytic block (review of #227) ---

TEST(PartialDerivatives, ACorrectPartialPassesAtTheScaleOfAStrain) {
  chain c;
  c.node<cubic_strain<0>>("n", "strain::strain");
  c.ctx.finalize();
  c.load(tensor2{1e-3 * x0}, s0);
  ASSERT_GT(std::abs(c.ctx.get<T>("n", "y")), 1e7);
  auto const mismatches{nm::check_partials(c.ctx, "n")};
  EXPECT_TRUE(mismatches.empty()) << mismatches.front();
}

TEST(PartialDerivatives, ASmallRelativeErrorIsDetectedAtOutputsOfOneHundredMillion) {
  chain c;
  c.node<cubic_strain<20>>("n", "strain::strain"); // 2e-5 relative
  c.ctx.finalize();
  c.load(tensor2{1e-3 * x0}, s0);
  ASSERT_GT(std::abs(c.ctx.get<T>("n", "y")), 1e7);
  EXPECT_FALSE(nm::check_partials(c.ctx, "n").empty());
}

// --- a throwing rerun leaves the context as it was (review of #227) ---

TEST(PartialDerivatives, AThrowingPerturbedRerunRestoresReadsAndOutputs) {
  chain c;
  c.node<throwing_map>("t", "strain::strain");
  c.ctx.finalize();
  c.load(x0, s0);
  throwing_map::armed = x0;
  throwing_map::active = true;
  EXPECT_THROW(nm::check_partials(c.ctx, "t"), std::runtime_error);
  throwing_map::active = false;
  EXPECT_EQ(tmech::norm(tensor2{dynamic_cast<nm::external_strain_source<policy> *>(c.ctx.find("strain"))->strain().new_value() - x0}), 0) << "the read";
  EXPECT_LT(distance(c.ctx.get<tensor2>("t", "y"), throwing_map::value(x0)), 1e-15) << "the output";
}

// --- the result type is the one the context can fill (review of #227) ---

template <typename R>
concept derivable = requires(ctx_type &c, std::string const &name) { c.template derivative<R>(name, name, name, name); };

TEST(PartialDerivatives, OnlyTheScalarAndTensorsOfTheContextAreDerivativeTypes) {
  static_assert(derivable<T>);
  static_assert(derivable<tensor2>);
  static_assert(derivable<tensor4>);
  static_assert(derivable<tensor6>);
  static_assert(!derivable<tmech::tensor<T, 2, 4>>); // another dimension would write out of bounds
  static_assert(!derivable<tmech::tensor<float, 3, 2>>);
  static_assert(!derivable<float>);
  static_assert(!derivable<int>);
  static_assert(!derivable<std::string>);
}


TEST(PartialDerivatives, AFourthOrderResultOfATensorIsRankSix) {
  chain c;
  c.node<dyad>("d", "strain::strain");
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor6>("d", "y", "strain", "strain")};
  c.load(x0, s0);
  EXPECT_LT(distance(d, dyad::partial(x0)), 1e-12);
}

// --- derivative() on an evaluated context, and the path query (review of #227) ---

TEST(PartialDerivatives, ADerivativeRequestedAfterAnUpdateIsFilledAtOnce) {
  chain c;
  c.node<map_a>("a", "strain::strain");
  c.node<map_b>("b", "a::y");
  c.ctx.finalize();
  c.load(x0, s0);
  auto const &d{c.ctx.derivative<tensor4>("b", "y", "strain", "strain")};
  EXPECT_LT(distance(d, tensor4{tmech::dcontract(B4, A4)}), 1e-12);
  c.load(tensor2{2 * x0}, s0);
  EXPECT_LT(distance(d, tensor4{tmech::dcontract(B4, A4)}), 1e-12) << "and by the next update";
}

TEST(PartialDerivatives, ADerivativeFromACallbackIsRefused) {
  chain c;
  c.node<map_a>("a", "strain::strain");
  c.ctx.finalize();
  bool refused{false};
  law_node<map_a, policy>::hook = [&] {
    try {
      c.ctx.derivative<tensor4>("a", "y", "strain", "strain");
    } catch (std::logic_error const &) {
      refused = true;
    }
  };
  c.load(x0, s0);
  law_node<map_a, policy>::hook = nullptr;
  EXPECT_TRUE(refused);
}

TEST(PartialDerivatives, HasPathTellsIndependenceFromATypo) {
  chain c;
  c.node<map_a>("y", "strain::strain");
  c.ctx.finalize();
  EXPECT_TRUE(c.ctx.has_path("y", "y", "strain", "strain"));
  EXPECT_TRUE(c.ctx.has_path("y", "y", "y", "y"));
  EXPECT_FALSE(c.ctx.has_path("y", "y", "temperature", "state"));
  EXPECT_FALSE(c.ctx.has_path("strain", "strain", "y", "y"));
  EXPECT_THROW(c.ctx.has_path("y", "y", "temperature", "nothing"), std::runtime_error);
}

// --- zero cost when nothing is requested ---

TEST(PartialDerivatives, NoPartialsAreComputedWithoutARequest) {
  chain c;
  auto &a{c.node<map_a>("a", "strain::strain")};
  auto &b{c.node<map_b>("b", "a::y")};
  auto &q{c.node<map_b>("q", "strain::strain")};
  c.ctx.finalize();
  for (int i{0}; i < 3; ++i)
    c.load(x0, s0);
  EXPECT_EQ(a.runs(), 3);
  EXPECT_EQ(a.fills() + b.fills() + q.fills(), 0);

  c.ctx.derivative<tensor4>("b", "y", "strain", "strain"); // fills once at the held state
  EXPECT_EQ(a.fills(), 1);
  EXPECT_EQ(b.fills(), 1);
  for (int i{0}; i < 2; ++i)
    c.load(x0, s0);
  EXPECT_EQ(a.fills(), 3);
  EXPECT_EQ(b.fills(), 3);
  EXPECT_EQ(q.fills(), 0) << "q is off the requested path";
}

// --- partials filled on request (callback_handle::fills_partials()) ---

TEST(PartialDerivatives, PartialsOnRequestAreFilledOncePerUpdateOfARequestedDerivative) {
  chain c;
  auto &a{c.node<map_a>("a", "strain::strain", "on_request")};
  auto &b{c.node<map_b>("b", "a::y", "on_request")};
  auto &q{c.node<map_b>("q", "strain::strain", "on_request")};
  c.ctx.finalize();
  for (int i{0}; i < 3; ++i)
    c.load(x0, s0);
  EXPECT_EQ(a.fills() + b.fills() + q.fills(), 0) << "update() alone fills nothing";

  auto const &d{c.ctx.derivative<tensor4>("b", "y", "strain", "strain")};
  EXPECT_EQ(a.fills(), 1) << "filled at once at the held state";
  EXPECT_EQ(b.fills(), 1);
  EXPECT_LT(distance(d, tensor4{tmech::dcontract(B4, A4)}), 1e-12);
  for (int i{0}; i < 2; ++i)
    c.load(x0, s0);
  EXPECT_EQ(a.fills(), 3);
  EXPECT_EQ(b.fills(), 3);
  EXPECT_EQ(q.fills(), 0) << "q is off the requested path";
  EXPECT_EQ(a.runs(), 5) << "the values are not recomputed for the partials";
}

TEST(PartialDerivatives, PartialsOnRequestFollowTheState) {
  chain c;
  c.node<norm_trace>("n", "strain::strain", "on_request");
  c.node<square>("s", "n::y", "on_request");
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor2>("s", "y", "strain", "strain")};
  for (T const k : {T{1}, T{-2}}) {
    tensor2 const x{k * x0};
    c.load(x, s0);
    EXPECT_LT(distance(d, tensor2{2 * norm_trace::value(x) * norm_trace::partial(x)}), 1e-12) << k;
  }
}

TEST(PartialDerivatives, AConstantCallbackFillsItsPartialsOnRequestOnce) {
  chain c;
  add<nm::constant_scalar<policy>>(c.ctx, "c", {{"value", s0}});
  auto &s{c.node<square>("s", "c::value", "on_request")};
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<T>("s", "y", "c", "value")};
  for (int i{0}; i < 3; ++i)
    c.load(x0, s0);
  EXPECT_NEAR(d, 2 * s0, 1e-12);
  EXPECT_EQ(s.fills(), 1) << "it did not run again, so its partials stand";
}

TEST(PartialDerivatives, TheFiniteDifferenceCheckFillsPartialsOnRequest) {
  chain c;
  c.node<map_a>("a", "strain::strain", "on_request");
  c.node<wrong_map>("w", "strain::strain", "on_request");
  c.ctx.finalize();
  c.load(x0, s0);
  EXPECT_TRUE(nm::check_partials(c.ctx, "a").empty());
  auto const mismatches{nm::check_partials(c.ctx, "w")};
  ASSERT_FALSE(mismatches.empty());
  EXPECT_TRUE(mismatches.front().contains("w::y")) << mismatches.front();
}

TEST(PartialDerivatives, AHandleFillsOnlyTheBlockOfARequestedDerivative) {
  chain c;
  auto &t{add<two_reads<policy>>(c.ctx, "t")};
  c.ctx.finalize();
  auto const &dw{c.ctx.derivative<tensor2>("t", "y", "temperature", "state")};
  for (int i{0}; i < 3; ++i)
    c.load(x0, s0);
  EXPECT_LT(distance(dw, I), 1e-12);
  EXPECT_EQ(t.w_fills(), 3) << "once per update";
  EXPECT_EQ(t.x_fills(), 0) << "no plan reads d y / d strain";

  auto const &dx{c.ctx.derivative<tensor4>("t", "y", "strain", "strain")};
  c.load(x0, s0);
  EXPECT_LT(distance(dx, Is), 1e-12);
  EXPECT_EQ(t.x_fills(), 2);
}

TEST(PartialDerivatives, ACallbackThatThrowsLeavesItsPartialsToBeFilledAgain) {
  chain c;
  auto &t{add<two_reads<policy>>(c.ctx, "t")};
  c.ctx.finalize();
  (void)c.ctx.derivative<tensor2>("t", "y", "temperature", "state");
  c.load(x0, s0);
  ASSERT_EQ(t.w_fills(), 1);
  auto const *callback{t.partials().front().callback};
  auto const *fill{t.fill_of(callback)};
  ASSERT_NE(fill, nullptr);
  t.fill_partials(*fill);
  EXPECT_EQ(t.w_fills(), 1) << "filled since the callback last ran";
  EXPECT_THROW(t.run_callback(callback, [] { throw std::runtime_error("cutback"); }, fill), std::runtime_error);
  t.fill_partials(*fill);
  EXPECT_EQ(t.w_fills(), 2) << "a throwing run may have changed what the partials depend on";
}

TEST(PartialDerivatives, AnUnsetHandleIsNotWanted) {
  using handle = two_reads<policy>::base::partial_handle<tensor2, T>;
  EXPECT_FALSE(handle{}.wanted());
}

TEST(PartialDerivatives, TheCheckFillsEveryBlockThroughHandles) {
  chain c;
  param_type p;
  p.insert<std::string>("name", "t");
  c.ctx.create<two_reads<policy>>(p);
  c.ctx.finalize();
  c.load(x0, s0);
  EXPECT_TRUE(nm::check_partials(c.ctx, "t").empty());
}

TEST(PartialDerivatives, FillsPartialsWithoutDeclaredPartialsThrows) {
  chain c;
  EXPECT_THROW(c.node<map_a>("a", "strain::strain", "fill_only"), std::logic_error);
}

// --- the per-node finite-difference check ---

TEST(PartialDerivatives, TheFiniteDifferenceCheckPassesCorrectPartials) {
  chain c;
  c.node<norm_trace>("n", "strain::strain");
  c.node<square>("s", "n::y");
  c.node<scale_b>("sb", "s::y");
  c.node<map_a>("a", "sb::y");
  c.node<project_c>("p", "a::y");
  c.node<dyad>("d", "a::y");
  c.node<contract_e>("e", "d::y");
  c.ctx.finalize();
  c.load(x0, s0);
  for (auto const *name : {"n", "s", "sb", "a", "p", "d", "e"})
    EXPECT_TRUE(nm::check_partials(c.ctx, name).empty()) << name;
}

TEST(PartialDerivatives, TheFiniteDifferenceCheckComparesWithThePartialsAtTheState) {
  chain c;
  c.node<map_a>("a", "strain::strain");
  c.node<scale_b>("b", "temperature::state");
  add<product_node<policy>>(c.ctx, "q");
  c.ctx.finalize();
  c.ctx.derivative<tensor4>("q", "y", "strain", "strain"); // keeps the partials of q requested
  c.load(x0, s0);
  auto const mismatches{nm::check_partials(c.ctx, "q")};
  EXPECT_TRUE(mismatches.empty()) << mismatches.front();
}

TEST(PartialDerivatives, TheFiniteDifferenceCheckNamesAWrongPartial) {
  chain c;
  c.node<wrong_map>("w", "strain::strain");
  c.ctx.finalize();
  c.load(x0, s0);
  auto const mismatches{nm::check_partials(c.ctx, "w")};
  ASSERT_FALSE(mismatches.empty());
  EXPECT_TRUE(mismatches.front().contains("w::y")) << mismatches.front();
  EXPECT_TRUE(mismatches.front().contains("strain::strain")) << mismatches.front();
  EXPECT_LT(distance(c.ctx.get<tensor2>("w", "y"), wrong_map::value(x0)), 1e-15) << "the check restores the state";
}

TEST(PartialDerivatives, APartialWithoutMinorSymmetryFailsTheFiniteDifferenceCheck) {
  chain c;
  c.node<congruence>("n", "strain::strain");
  c.ctx.finalize();
  c.load(x0, s0);
  auto const mismatches{nm::check_partials(c.ctx, "n")};
  ASSERT_FALSE(mismatches.empty()) << "a Mandel comparison projects the partial onto its symmetric part";
  EXPECT_TRUE(mismatches.front().contains("minor-symmetric")) << mismatches.front();
}

TEST(PartialDerivatives, TensorOfFourthOrderTensorOfFourthOrderTensor) {
  chain c;
  c.node<dyad>("y", "strain::strain");
  c.node<map_y>("z", "y::y");
  c.node<contract_e>("w", "z::y");
  c.ctx.finalize();
  EXPECT_TRUE(nm::check_partials(c.ctx, "z").empty());
  auto const &dw{c.ctx.derivative<tensor6>("w", "y", "y", "y")};
  auto const &dz{c.ctx.derivative<tensor8>("z", "y", "y", "y")};
  c.load(x0, s0);
  EXPECT_LT(distance(dw, tensor6{tmech::otimes(A4, E2)}), 1e-12) << "w = A : Y : E through a tensor4 seed";
  EXPECT_LT(distance(dz, map_y::partial(tensor4{})), 1e-12);
}

// --- thermal eigenstrain with linear elasticity: the first DDSDDT ---

constexpr T K{100}, G{40}, alpha{1e-5}, T_ref{20};

struct thermoelastic {
  std::unique_ptr<ctx_type> owner{std::make_unique<ctx_type>()};
  ctx_type &ctx{*owner};
  std::unique_ptr<nm::material_point<policy>> point;

  thermoelastic() {
    param_type p;
    add<nm::external_strain_source<policy>>(ctx, "strain");
    add<nm::external_scalar_source<policy>>(ctx, "temperature");
    add<nm::constant_scalar<policy>>(ctx, "alpha", {{"value", alpha}});
    add<nm::thermal_eigenstrain<policy>>(ctx, "thermal",
        {{"alpha_source", "alpha::value"}, {"reference_temperature", T_ref}});
    add<nm::elastic_strain<policy>>(ctx, "elastic_strain",
        {{"strain_source", "strain"}, {"eigenstrain_sources", std::vector<std::string>{"thermal"}}});
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "elastic_strain");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    ctx.create<nm::linear_elasticity<policy>>(p);
    ctx.finalize();
    param_type q;
    q.insert<std::vector<std::string>>("inputs", {"strain", "temperature"});
    point = std::make_unique<nm::material_point<policy>>(ctx, q);
  }

  void evaluate(tensor2 const &strain, T temperature) {
    point->reset();
    point->input<nm::external_strain_source<policy>>("strain").bind(tensor2{}, strain);
    point->input<nm::external_scalar_source<policy>>("temperature").bind(T_ref, temperature);
    point->evaluate();
  }
};

TEST(PartialDerivatives, StressWithRespectToTemperatureMatchesFiniteDifferences) {
  thermoelastic m;
  auto const &ddsddt{m.point->derivative<tensor2>("elastic::stress", "temperature")};
  tensor2 const eps{sym(1e-3, -2e-4, 5e-4, 1e-4, -3e-4, 2e-4)};
  T const theta{80}, h{1e-3};
  m.evaluate(eps, theta + h);
  tensor2 const plus{m.point->output<tensor2>("elastic::stress")};
  m.evaluate(eps, theta - h);
  tensor2 const minus{m.point->output<tensor2>("elastic::stress")};
  m.evaluate(eps, theta);
  tensor2 const fd{(plus - minus) / (2 * h)};
  EXPECT_LT(tmech::norm(tensor2{ddsddt - fd}), 1e-8 * tmech::norm(fd));
  EXPECT_LT(tmech::norm(tensor2{ddsddt + 3 * K * alpha * I}), 1e-12) << "-C : alpha I";
}

TEST(PartialDerivatives, StressWithRespectToStrainEqualsTheTangent) {
  thermoelastic m;
  auto const &ddsdde{m.point->derivative<tensor4>("elastic::stress", "strain")};
  m.evaluate(sym(1e-3, -2e-4, 5e-4, 1e-4, -3e-4, 2e-4), 60);
  auto const &tangent{m.point->output<tensor4>("elastic::tangent")};
  EXPECT_LT(tmech::norm(tensor4{ddsdde - tangent}), 1e-12 * tmech::norm(tangent));
}

TEST(PartialDerivatives, TheMaterialPointNamesUnknownInputs) {
  thermoelastic m;
  EXPECT_THROW(m.point->derivative<tensor2>("elastic::stress", "pressure"), nm::umat::fatal_error);
  EXPECT_THROW(m.point->derivative<tensor2>("elastic::nothing", "temperature"), nm::umat::fatal_error);
}

TEST(PartialDerivatives, ThermoelasticPartialsPassTheFiniteDifferenceCheck) {
  thermoelastic m;
  m.evaluate(sym(1e-3, -2e-4, 5e-4, 1e-4, -3e-4, 2e-4), 60);
  for (auto const *name : {"thermal", "elastic_strain", "elastic"})
    for (auto const &mismatch : nm::check_partials(m.ctx, name))
      ADD_FAILURE() << mismatch;
}

// --- a declared partial is filled only when a requested derivative uses it ---

// Largest component of the packed partial d `output` / d `read` that `material` declares for read_name.
T filled(thermoelastic &m, std::string const &read_name, std::string const &material = "elastic") {
  auto *mat{m.ctx.find(material)};
  for (auto const &p : mat->partials()) {
    bool const own{p.own && p.own->traits().id.name == read_name};
    bool const in{p.input && p.input->source_name() == read_name};
    if (!own && !in)
      continue;
    std::vector<T> packed(p.block->rows() * p.block->cols());
    p.block->pack(packed.data());
    T largest{0};
    for (auto const v : packed)
      largest = std::max(largest, std::abs(v));
    return largest;
  }
  throw std::logic_error("no partial for " + read_name);
}

TEST(PartialDerivatives, APartialNoRequestedDerivativeUsesIsNotFilled) {
  thermoelastic m;
  auto const &ddsdde{m.point->derivative<tensor4>("elastic::stress", "strain")};
  tensor2 const eps{sym(1e-3, -2e-4, 5e-4, 1e-4, -3e-4, 2e-4)};
  m.evaluate(eps, T_ref);
  auto const &tangent{m.point->output<tensor4>("elastic::tangent")};
  EXPECT_LT(tmech::norm(tensor4{ddsdde - tangent}), 1e-12 * tmech::norm(tangent));
  EXPECT_GT(filled(m, "strain"), 0);
  EXPECT_EQ(filled(m, "tangent"), 0) << "d stress / d C is not on a requested path";
  EXPECT_GT(filled(m, "strain", "elastic_strain"), 0);
  EXPECT_EQ(filled(m, "eigenstrain", "elastic_strain"), 0) << "nor d elastic strain / d eigenstrain";
}

TEST(PartialDerivatives, ADerivativeThroughTheTangentFillsItsPartial) {
  thermoelastic m;
  auto const &ddsdde{m.point->derivative<tensor4>("elastic::stress", "strain")};
  auto const &dsdc{m.point->derivative<tensor6>("elastic::stress", "elastic::tangent")};
  tensor2 const eps{sym(1e-3, -2e-4, 5e-4, 1e-4, -3e-4, 2e-4)};
  m.evaluate(eps, T_ref);
  auto const &tangent{m.point->output<tensor4>("elastic::tangent")};
  EXPECT_LT(tmech::norm(tensor4{ddsdde - tangent}), 1e-12 * tmech::norm(tangent));
  EXPECT_LT(tmech::norm(tensor6{dsdc - tensor6{tmech::otimes(Is, eps)}}), 1e-12 * tmech::norm(dsdc));
  EXPECT_GT(filled(m, "tangent"), 0);
}

// --- pack and unpack against the index-loop definition ---

// The definition pack and unpack replaced: each index pair projected onto its Mandel component.
template <std::size_t Dim, std::size_t Rank>
tmech::tensor<T, Dim, Rank> minor_symmetric(unsigned seed) {
  tmech::tensor<T, Dim, Rank> t;
  std::array<std::size_t, Rank> idx{};
  auto value{[&](std::array<std::size_t, Rank> i) {
    unsigned h{seed};
    for (std::size_t k{0}; k < Rank; k += 2) {
      auto const lo{std::min(i[k], i[k + 1])}, hi{std::max(i[k], i[k + 1])};
      h = h * 31u + static_cast<unsigned>(lo * 7 + hi * 3 + 1);
    }
    return T(h % 1000u) / 100 - 5;
  }};
  std::size_t total{1};
  for (std::size_t k{0}; k < Rank; ++k)
    total *= Dim;
  for (std::size_t flat{0}; flat < total; ++flat) {
    for (std::size_t k{Rank}, rest{flat}; k-- > 0; rest /= Dim)
      idx[k] = rest % Dim;
    [&]<std::size_t... I>(std::index_sequence<I...>) { t(idx[I]...) = value(idx); }(std::make_index_sequence<Rank>{});
  }
  return t;
}

template <std::size_t Dim, std::size_t Rank, std::size_t OutRank>
void expect_round_trip() {
  namespace dv = numsim::materials::derivatives;
  auto const t{minor_symmetric<Dim, Rank>(Rank + Dim)};
  constexpr std::size_t rows{dv::width<Dim>(OutRank)}, cols{dv::width<Dim>(Rank - OutRank)};
  std::vector<T> block(rows * cols);
  dv::pack<Dim, OutRank>(t, block.data());
  // Mandel components: a sqrt(2) per off-diagonal pair makes A : A the sum of squares of the block.
  T squares{0};
  for (auto const v : block)
    squares += v * v;
  T full{0};
  std::size_t total{1};
  for (std::size_t k{0}; k < Rank; ++k)
    total *= Dim;
  for (std::size_t k{0}; k < total; ++k)
    full += t.raw_data()[k] * t.raw_data()[k];
  EXPECT_NEAR(squares, full, 1e-10 * full);
  tmech::tensor<T, Dim, Rank> back;
  dv::unpack<Dim, OutRank>(block.data(), back);
  for (std::size_t k{0}; k < total; ++k)
    EXPECT_NEAR(back.raw_data()[k], t.raw_data()[k], 1e-12) << "rank " << Rank << " dim " << Dim << " flat " << k;
}

TEST(PartialDerivatives, PackAndUnpackRoundTripMinorSymmetricTensors) {
  expect_round_trip<3, 2, 0>();
  expect_round_trip<3, 4, 2>();
  expect_round_trip<3, 4, 0>();
  expect_round_trip<3, 6, 2>();
  expect_round_trip<3, 6, 4>();
  expect_round_trip<3, 8, 4>();
  expect_round_trip<2, 2, 0>();
  expect_round_trip<2, 4, 2>();
  expect_round_trip<2, 6, 2>();
  expect_round_trip<2, 6, 4>();
}

} // namespace

namespace {
/// Entries of a value of rank r in full notation: Dim^r.
template <std::size_t Dim>
std::size_t entries(std::size_t rank) {
  std::size_t n{1};
  for (std::size_t k{0}; k < rank; ++k)
    n *= Dim;
  return n;
}

/// The tmech product of a kernel is the chain rule in full notation: out (Dim^a x Dim^s) [+]= P (Dim^a x Dim^b) *
/// in (Dim^b x Dim^s), all row-major, tmech's storage order.
template <std::size_t Dim>
void expect_kernels_match_the_row_major_product() {
  using matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  auto const random{[](std::size_t rank) {
    auto v{nm::derivatives::make_value<T, Dim>(rank)};
    Eigen::Map<matrix>(v->data(), 1, static_cast<Eigen::Index>(entries<Dim>(rank))) = matrix::Random(1, static_cast<Eigen::Index>(entries<Dim>(rank)));
    return v;
  }};
  for (std::size_t const a : {0, 2, 4})
    for (std::size_t const b : {0, 2, 4})
      for (std::size_t const s : {0, 2, 4})
        for (bool const assign : {false, true}) {
          auto const rows{static_cast<Eigen::Index>(entries<Dim>(a))}, inner{static_cast<Eigen::Index>(entries<Dim>(b))},
              cols{static_cast<Eigen::Index>(entries<Dim>(s))};
          auto P{random(a + b)}, in{random(b + s)}, out{random(a + s)};
          Eigen::Map<matrix const> const p{P->data(), rows, inner}, d{in->data(), inner, cols};
          Eigen::Map<matrix> o{out->data(), rows, cols};
          matrix const expected{assign ? matrix{p * d} : matrix{o + p * d}};
          auto scratch{nm::derivatives::make_value<T, Dim>(a + s)};
          nm::derivatives::step_kernel_for<T, Dim>(a, b, s, assign)(P->object(), in->object(), out->object(), scratch->object());
          EXPECT_LT((o - expected).cwiseAbs().maxCoeff(), 1e-12) << a << " " << b << " " << s << " " << assign;
          if (b != s)
            continue;
          auto seeded{random(a + s)};
          Eigen::Map<matrix> q{seeded->data(), rows, cols};
          matrix const added{assign ? matrix{p} : matrix{q + p}};
          nm::derivatives::seed_kernel_for<T, Dim>(a, s, assign)(P->object(), nullptr, seeded->object(), nullptr);
          EXPECT_EQ(q, added) << a << " " << s << " " << assign;
        }
}
} // namespace

TEST(PartialDerivatives, TheTmechKernelsAreTheRowMajorProduct) {
  expect_kernels_match_the_row_major_product<2>();
  expect_kernels_match_the_row_major_product<3>();
}

TEST(PartialDerivatives, TheSeedIdentityMapsASymmetricValueToItself) {
  tensor4 I2;
  nm::derivatives::identity<T, 3>(2, &I2);
  tensor2 const S2{0.5 * (N2 + tmech::trans(N2))};
  EXPECT_LT(distance(tensor2{tmech::dcontract(I2, N2)}, S2), 1e-12) << "projects onto the symmetric part";
  EXPECT_LT(distance(tensor2{tmech::dcontract(N2, I2)}, S2), 1e-12);
  tensor8 I4;
  nm::derivatives::identity<T, 3>(4, &I4);
  tensor4 X;
  X.randn();
  tensor4 const S4{0.25 * (X + tmech::basis_change<tmech::sequence<2, 1, 3, 4>>(X) + tmech::basis_change<tmech::sequence<1, 2, 4, 3>>(X) +
                           tmech::basis_change<tmech::sequence<2, 1, 4, 3>>(X))};
  EXPECT_LT(distance(tensor4{tmech::ddcontract(I4, X)}, S4), 1e-12) << "projects onto the minor-symmetric part";
  EXPECT_LT(distance(tensor4{tmech::ddcontract(X, I4)}, S4), 1e-12);
  T one{};
  nm::derivatives::identity<T, 3>(0, &one);
  EXPECT_EQ(one, 1.0);
}

#if defined(__SANITIZE_ADDRESS__)
#define NM_TEST_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define NM_TEST_ASAN 1
#endif
#endif

TEST(PartialDerivatives, AnAccumulatingRankEightProductFitsASmallThreadStack) {
#if !defined(__OPTIMIZE__) || defined(NM_TEST_ASAN)
  GTEST_SKIP() << "unoptimised or ASan builds keep tmech's contraction temporaries in the frame";
#endif
  auto P{nm::derivatives::make_value<T, 3>(8)}, in{nm::derivatives::make_value<T, 3>(8)}, out{nm::derivatives::make_value<T, 3>(8)},
      scratch{nm::derivatives::make_value<T, 3>(8)};
  struct job {
    void const *P, *in;
    void *out, *scratch;
  } j{P->object(), in->object(), out->object(), scratch->object()};
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 128 * 1024); // small thread stacks, as HPX's
  pthread_t thread;
  void *(*const run)(void *){[](void *p) -> void * {
    auto const &k{*static_cast<job *>(p)};
    nm::derivatives::step_kernel_for<T, 3>(4, 4, 4, false)(k.P, k.in, k.out, k.scratch);
    return nullptr;
  }};
  ASSERT_EQ(pthread_create(&thread, &attr, run, &j), 0);
  pthread_join(thread, nullptr);
  pthread_attr_destroy(&attr);
}

TEST(PartialDerivatives, TheScalarFoldScalesThroughTmech) {
  tensor4 out;
  nm::derivatives::scale_kernel_for<T, 3>(4)(-0.5, &A4, &out);
  EXPECT_LT(distance(out, tensor4{-0.5 * A4}), 1e-12);
}

TEST(PartialDerivatives, TheLuBoundaryPacksOnlySystemRowsOfRankTwoAtMost) {
  std::vector<T> full(6561), block(36 * 36);
  EXPECT_NO_THROW(nm::derivatives::pack_full<3>(2, 4, full.data(), block.data()));
  EXPECT_THROW(nm::derivatives::pack_full<3>(4, 4, full.data(), block.data()), std::logic_error);
  EXPECT_THROW(nm::derivatives::unpack_full<3>(4, 2, block.data(), full.data()), std::logic_error);
}

TEST(PartialDerivatives, ARankOfNoNodeValueHasNoKernel) {
  EXPECT_THROW((void)(nm::derivatives::step_kernel_for<T, 3>(1, 2, 2, false)), std::logic_error);
  EXPECT_THROW((void)(nm::derivatives::step_kernel_for<T, 3>(2, 6, 2, false)), std::logic_error);
  EXPECT_THROW((void)(nm::derivatives::seed_kernel_for<T, 3>(2, 3, false)), std::logic_error);
}

// --- unsymmetric tensors (#234) ---

namespace {

tensor2 const F0{1.1, 0.2, -0.1, 0.05, 0.9, 0.3, -0.2, 0.15, 1.2}; // a deformation gradient

struct inverse_law { // F^-1: unsymmetric of unsymmetric
  using in = tensor2;
  using out = tensor2;
  static tensor2 value(tensor2 const &x) { return tmech::inv(x); }
  static tensor4 partial(tensor2 const &x) {
    tensor2 const xi{tmech::inv(x)};
    return tensor4{-tmech::otimesu(xi, tmech::trans(xi))};
  }
};
struct left_cauchy_green { // x x^T: symmetric of unsymmetric
  using in = tensor2;
  using out = tensor2;
  static tensor2 value(tensor2 const &x) { return tensor2{x * tmech::trans(x)}; }
  static tensor4 partial(tensor2 const &x) { return tensor4{tmech::otimesu(I, x) + tmech::otimesl(x, I)}; }
};
struct symmetrised_left_cauchy_green : left_cauchy_green { // its partial projected onto the symmetric part of dx
  static tensor4 partial(tensor2 const &x) { return tensor4{tmech::dcontract(left_cauchy_green::partial(x), Is)}; }
};
struct push_forward { // N x: unsymmetric of symmetric
  using in = tensor2;
  using out = tensor2;
  static tensor2 value(tensor2 const &x) { return tensor2{N2 * x}; }
  static tensor4 partial(tensor2 const &) { return tensor4{0.5 * (tmech::otimesu(N2, I) + tmech::otimesl(N2, I))}; }
};
struct skew_square { // |skew(x)|^2: blind to a symmetric perturbation
  using in = tensor2;
  using out = T;
  static T value(tensor2 const &x) {
    tensor2 const w{0.5 * (x - tmech::trans(x))};
    return tmech::dcontract(w, w);
  }
  static tensor2 partial(tensor2 const &x) { return tensor2{x - tmech::trans(x)}; }
};

/// The chain with an unsymmetric source "F".
struct f_chain : chain {
  f_chain() { add<nm::test::unsymmetric_source<policy>>(ctx, "F"); }

  template <typename Law>
  law_node<Law, policy> &unsymmetric_node(std::string const &name, std::string const &x_source) {
    return add<law_node<Law, policy>>(ctx, name, {{"x_source", x_source}, {"unsymmetric", true}});
  }

  void load_F(tensor2 const &F) {
    dynamic_cast<nm::test::unsymmetric_source<policy> *>(ctx.find("F"))->bind(F);
    ctx.update();
  }
};

/// d material::property / dF at F by central differences of the context, one entry of F at a time.
template <typename V>
auto central_differences(f_chain &c, std::string const &material, std::string const &property, tensor2 const &F) {
  std::conditional_t<std::is_arithmetic_v<V>, tensor2, tensor4> d{};
  constexpr T h{1e-6};
  for (std::size_t k{0}; k < 3; ++k)
    for (std::size_t l{0}; l < 3; ++l) {
      std::array<V, 2> v{};
      for (std::size_t const side : {0u, 1u}) {
        tensor2 Fh{F};
        Fh(k, l) += side == 0 ? h : -h;
        c.load_F(Fh);
        v[side] = c.ctx.get<V>(material, property);
      }
      if constexpr (std::is_arithmetic_v<V>)
        d(k, l) = (v[0] - v[1]) / (2 * h);
      else
        for (std::size_t i{0}; i < 3; ++i)
          for (std::size_t j{0}; j < 3; ++j)
            d(i, j, k, l) = (v[0](i, j) - v[1](i, j)) / (2 * h);
    }
  c.load_F(F);
  return d;
}

TEST(UnsymmetricDerivatives, TheSeedIdentityOfAnUnsymmetricValueIsTheFullIdentity) {
  tensor4 I2;
  nm::derivatives::identity<T, 3>(2, &I2, true);
  EXPECT_LT(distance(tensor2{tmech::dcontract(I2, N2)}, N2), 1e-15);
  EXPECT_LT(distance(tensor2{tmech::dcontract(N2, I2)}, N2), 1e-15);
  tensor8 I4;
  nm::derivatives::identity<T, 3>(4, &I4, true);
  tensor4 X;
  X.randn();
  EXPECT_LT(distance(tensor4{tmech::ddcontract(I4, X)}, X), 1e-14);
  EXPECT_LT(distance(tensor4{tmech::ddcontract(X, I4)}, X), 1e-14);
}

TEST(UnsymmetricDerivatives, AnUnsymmetricSeedIsItsOwnFullIdentity) {
  f_chain c;
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor4>("F", "F", "F", "F")};
  c.load_F(F0);
  EXPECT_LT(distance(d, tensor4{tmech::otimesu(I, I)}), 1e-15) << "dF/dF = d_ik d_jl, not its symmetric part";
}

TEST(UnsymmetricDerivatives, ASymmetricSeedKeepsTheSymmetricIdentity) {
  f_chain c;
  c.ctx.finalize();
  auto const &d{c.ctx.derivative<tensor4>("strain", "strain", "strain", "strain")};
  c.load(x0, s0);
  EXPECT_LT(distance(d, Is), 1e-15) << "DDSDDE stays minor-symmetric";
}

TEST(UnsymmetricDerivatives, ASeedFThroughANonSymmetricChainMatchesCentralDifferences) {
  f_chain c;
  c.unsymmetric_node<inverse_law>("inverse", "F::F"); // unsymmetric of unsymmetric
  c.node<left_cauchy_green>("b", "inverse::y");      // symmetric of unsymmetric
  c.unsymmetric_node<push_forward>("pushed", "b::y"); // unsymmetric of symmetric
  c.node<project_c>("s", "b::y");
  c.ctx.finalize();
  auto const &d_inverse{c.ctx.derivative<tensor4>("inverse", "y", "F", "F")};
  auto const &d_b{c.ctx.derivative<tensor4>("b", "y", "F", "F")};
  auto const &d_pushed{c.ctx.derivative<tensor4>("pushed", "y", "F", "F")};
  auto const &d_s{c.ctx.derivative<tensor2>("s", "y", "F", "F")};
  c.load_F(F0);
  EXPECT_LT(distance(d_inverse, inverse_law::partial(F0)), 1e-12);
  EXPECT_LT(distance(d_inverse, central_differences<tensor2>(c, "inverse", "y", F0)), 1e-8);
  EXPECT_LT(distance(d_b, central_differences<tensor2>(c, "b", "y", F0)), 1e-8);
  EXPECT_LT(distance(d_pushed, central_differences<tensor2>(c, "pushed", "y", F0)), 1e-8);
  EXPECT_LT(distance(d_s, central_differences<T>(c, "s", "y", F0)), 1e-8);
}

TEST(UnsymmetricDerivatives, MixedPartialsPassTheFiniteDifferenceCheck) {
  f_chain c;
  c.unsymmetric_node<inverse_law>("inverse", "F::F");
  c.node<left_cauchy_green>("b", "inverse::y");
  c.unsymmetric_node<push_forward>("pushed", "b::y");
  c.node<skew_square>("w", "pushed::y");
  c.ctx.finalize();
  c.load_F(F0);
  for (auto const *name : {"inverse", "b", "pushed", "w"}) {
    auto const mismatches{nm::check_partials(c.ctx, name)};
    EXPECT_TRUE(mismatches.empty()) << name << ": " << mismatches.front();
  }
}

TEST(UnsymmetricDerivatives, ASymmetrisedPartialWithRespectToFFailsTheFiniteDifferenceCheck) {
  f_chain c;
  c.node<symmetrised_left_cauchy_green>("b", "F::F");
  c.ctx.finalize();
  c.load_F(F0);
  auto const mismatches{nm::check_partials(c.ctx, "b")};
  ASSERT_FALSE(mismatches.empty()) << "a symmetric perturbation of F cannot see the symmetrised partial";
  EXPECT_TRUE(mismatches.front().contains("full-notation column")) << mismatches.front();
}

TEST(UnsymmetricDerivatives, AWrongIndependentOfOnTheSkewPartOfFFailsTheFiniteDifferenceCheck) {
  f_chain c;
  c.node<skew_square>("w", "F::F", "independent");
  c.ctx.finalize();
  c.load_F(F0);
  auto const mismatches{nm::check_partials(c.ctx, "w")};
  ASSERT_FALSE(mismatches.empty());
  EXPECT_TRUE(mismatches.front().contains("independent_of")) << mismatches.front();
}

TEST(UnsymmetricDerivatives, PackLayoutKeepsTheEntriesOfAFullPairAndInvertsOnTheMandelOnes) {
  tensor4 X;
  X.randn();
  tensor4 const symmetric_out{0.5 * (X + tmech::basis_change<tmech::sequence<2, 1, 3, 4>>(X))};
  std::array<T, 6 * 9> block{};
  nm::derivatives::pack_layout<3>(X.raw_data(), 4, 2, false, true, block.data());
  tensor4 back;
  nm::derivatives::unpack_layout<3>(block.data(), 4, 2, false, true, back.raw_data());
  EXPECT_LT(distance(back, symmetric_out), 1e-14) << "the output pair symmetrised, the read pair kept";
  std::array<T, 36> mandel{}, layout{};
  nm::derivatives::pack<3, 2>(Is, mandel.data());
  nm::derivatives::pack_layout<3>(Is.raw_data(), 4, 2, false, false, layout.data());
  for (std::size_t q{0}; q < mandel.size(); ++q)
    EXPECT_NEAR(layout[q], mandel[q], 1e-15) << q;
}

} // namespace
