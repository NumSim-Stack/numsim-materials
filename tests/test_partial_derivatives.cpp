#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/partial_check.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/driver/material_point.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/thermal_eigenstrain.h"
#include "numsim-materials/umat/external_state_source.h"

// Partial derivatives per callback and their forward accumulation (#217).

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using tensor6 = tmech::tensor<T, 3, 6>;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

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

/// y = Law::value(x) for x read from "x_source"; "partials" is "declared" (the default), "independent" or "none".
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
    if (m_mode == "declared")
      handle.partials(m_y, m_x);
    else if (m_mode == "independent")
      handle.independent_of(m_x);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<std::string>(para, "x_source");
    nm::optional<std::string>(para, "partials", std::string{"declared"});
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
    param_type p;
    p.insert<std::string>("name", "strain");
    ctx.create<nm::external_strain_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "temperature");
    ctx.create<nm::external_scalar_source<policy>>(p);
  }

  template <typename Law>
  law_node<Law, policy> &node(std::string const &name, std::string const &x_source, std::string const &partials = "declared") {
    param_type p;
    p.insert<std::string>("name", name);
    p.insert<std::string>("x_source", x_source);
    p.insert<std::string>("partials", partials);
    return ctx.create<law_node<Law, policy>>(p);
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
  param_type p;
  p.insert<std::string>("name", "s");
  c.ctx.create<sum_node<policy>>(p);
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
  for (auto const *mistake : {"unread", "twice", "both", "other_output", "before_reads"}) {
    chain c;
    param_type p;
    p.insert<std::string>("name", "m");
    p.insert<std::string>("mistake", mistake);
    EXPECT_THROW(c.ctx.create<misdeclared<policy>>(p), std::logic_error) << mistake;
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
  param_type p;
  p.insert<std::string>("name", "q");
  c.ctx.create<product_node<policy>>(p);
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

// --- thermal eigenstrain with linear elasticity: the first DDSDDT ---

constexpr T K{100}, G{40}, alpha{1e-5}, T_ref{20};

struct thermoelastic {
  std::unique_ptr<ctx_type> owner{std::make_unique<ctx_type>()};
  ctx_type &ctx{*owner};
  std::unique_ptr<nm::material_point<policy>> point;

  thermoelastic() {
    param_type p;
    p.insert<std::string>("name", "strain");
    ctx.create<nm::external_strain_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "temperature");
    ctx.create<nm::external_scalar_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "alpha");
    p.insert<T>("value", alpha);
    ctx.create<nm::constant_scalar<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "thermal");
    p.insert<std::string>("alpha_source", "alpha::value");
    p.insert<T>("reference_temperature", T_ref);
    ctx.create<nm::thermal_eigenstrain<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "strain");
    p.insert<std::string>("eigenstrain_source", "thermal");
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
  for (auto const *name : {"thermal", "elastic"})
    for (auto const &mismatch : nm::check_partials(m.ctx, name))
      ADD_FAILURE() << mismatch;
}

} // namespace
