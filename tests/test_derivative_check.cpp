#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/materials/j2_plasticity.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/solvers/local_newton.h"
#include "test_support.h"

/// check_derivative(): name the output and the input; the graph perturbs the input along the directions of its
/// codec, reverting and updating as a host does.

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using numsim::materials::test::add;

param_type check(std::string const &output, std::string const &input, std::string const &analytical) {
  param_type p;
  p.insert<std::string>("output", output);
  p.insert<std::string>("input", input);
  p.insert<std::string>("analytical", analytical);
  return p;
}

template <class F>
std::string thrown(F &&f) {
  try {
    f();
  } catch (std::exception const &e) {
    return e.what();
  }
  return "no throw";
}

/// J2 with linear hardening behind a strain named "stepper::strain", bound by the host or (stepper) a callback.
struct j2_point {
  ctx_type ctx;
  nm::external_strain_source<policy> *strain{nullptr};

  explicit j2_point(bool stepper = false) {
    constexpr T K{166.67}, G{76.92};
    param_type p;
    p.insert<std::string>("name", "stepper");
    if (stepper) {
      p.insert<T>("increment", T{0.05});
      p.insert<std::vector<std::size_t>>("indices", {0, 0});
      ctx.create<nm::tensor_component_stepper<2, policy>>(p);
    } else {
      strain = &ctx.create<nm::external_strain_source<policy>>(p);
    }
    p.clear();
    add<nm::linear_elasticity<policy>>(ctx, "elastic", {{"strain_producer_name", "stepper"}, {"K", K}, {"G", G}});
    add<nm::local_newton<policy>>(ctx, "solver");
    add<nm::linear_isotropic_hardening<policy>>(ctx, "hardening", {{"source", "j2"}, {"K", T{1000.0}}});
    add<nm::j2_plasticity<policy>>(ctx, "j2",
        {{"hardening_source", "hardening"}, {"strain_source", "stepper"}, {"solver_source", "solver"}, {"K", K},
         {"G", G}, {"sigma_0", T{50.0}}});
    ctx.finalize();
  }

  /// One host increment along eps_00.
  void load(T eps_00) {
    tensor2 e{strain->strain().old_value()};
    e(0, 0) = eps_00;
    strain->bind(strain->strain().old_value(), e);
    ctx.update();
  }

  auto tangent() { return ctx.check_derivative(check("j2::stress", "stepper::strain", "j2::tangent")); }
};

/// "value" = x², "slope" = 2x from "x::state"; throws above x = 1e6.
template <typename Traits>
class square final : public nm::material_base<square<Traits>, Traits> {
public:
  using base = nm::material_base<square<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit square(Args &&...args)
      : base(std::forward<Args>(args)...), m_value(base::template add_output<value_type>("value")),
        m_slope(base::template add_output<value_type>("slope")), m_x(base::template add_input<value_type>("x", "state")) {
    base::add_callback(&square::compute, m_value, m_slope).reads(m_x);
  }

  static input_parameter_controller parameters() { return base::parameters(); }

  void compute() {
    if (m_x.get() > 1e6)
      throw std::runtime_error("square: above 1e6");
    m_value = m_x.get() * m_x.get();
    m_slope = 2 * m_x.get();
  }

private:
  value_type &m_value;
  value_type &m_slope;
  nm::input_property<value_type, nm::property_traits> const &m_x;
};

struct scalar_point {
  ctx_type ctx;
  nm::external_scalar_source<policy> *x{nullptr};

  explicit scalar_point(T value) {
    param_type p;
    x = &add<nm::external_scalar_source<policy>>(ctx, "x");
    p.insert<std::string>("name", "square");
    ctx.create<square<policy>>(p);
    ctx.finalize();
    x->bind(T{0}, value);
    ctx.update();
  }

  auto slope(param_type p) { return ctx.check_derivative(std::move(p)); }
};

/// "determinant" = det(F) and "cofactor" = d det(F)/dF = det(F) F^-T, from "F::deformation_gradient".
template <typename Traits>
class cofactor final : public nm::material_base<cofactor<Traits>, Traits> {
public:
  using base = nm::material_base<cofactor<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using tensor = tmech::tensor<value_type, 3, 2>;

  template <typename... Args>
  explicit cofactor(Args &&...args)
      : base(std::forward<Args>(args)...), m_J(base::template add_output<value_type>("determinant")),
        m_cofactor(base::template add_output<tensor>("cofactor")),
        m_F(base::template add_input<tensor>("F", "deformation_gradient")) {
    base::add_callback(&cofactor::compute, m_J, m_cofactor).reads(m_F);
  }

  static input_parameter_controller parameters() { return base::parameters(); }

  void compute() {
    m_J = tmech::det(m_F.get());
    m_cofactor = m_J * tmech::trans(tmech::inv(m_F.get()));
  }

private:
  value_type &m_J;
  tensor &m_cofactor;
  nm::input_property<tensor, nm::property_traits> const &m_F;
};

/// "value": a tensor no callback fills and no codec describes.
template <typename Traits>
class frozen final : public nm::material_base<frozen<Traits>, Traits> {
public:
  using base = nm::material_base<frozen<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit frozen(Args &&...args) : base(std::forward<Args>(args)...) {
    base::template add_output<tmech::tensor<typename base::value_type, 3, 2>>("value");
  }

  static input_parameter_controller parameters() { return base::parameters(); }
};

/// "value" = x dt, "slope" = dt: x from "x::state", dt from the host-bound time "t::state".
template <typename Traits>
class rate final : public nm::material_base<rate<Traits>, Traits> {
public:
  using base = nm::material_base<rate<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit rate(Args &&...args)
      : base(std::forward<Args>(args)...), m_value(base::template add_output<value_type>("value")),
        m_slope(base::template add_output<value_type>("slope")), m_x(base::template add_input<value_type>("x", "state")),
        m_t(base::template add_input_history<value_type>(nm::connection_source{"t", "state"})) {
    base::add_callback(&rate::compute, m_value, m_slope).reads(m_x, m_t);
  }

  static input_parameter_controller parameters() { return base::parameters(); }

  void compute() {
    m_slope = m_t.new_value() - m_t.old_value();
    m_value = m_x.get() * m_slope;
  }

private:
  value_type &m_value;
  value_type &m_slope;
  nm::input_property<value_type, nm::property_traits> const &m_x;
  nm::input_history<value_type, nm::property_traits> const &m_t;
};

/// "copy" by a callback from "x::state", and "bare": no callback of its own, filled by the same one.
template <typename Traits>
class bare final : public nm::material_base<bare<Traits>, Traits> {
public:
  using base = nm::material_base<bare<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit bare(Args &&...args)
      : base(std::forward<Args>(args)...), m_copy(base::template add_output<value_type>("copy")),
        m_bare(base::template add_output<value_type>("bare")), m_x(base::template add_input<value_type>("x", "state")) {
    base::add_callback(&bare::compute, m_copy).reads(m_x);
  }

  static input_parameter_controller parameters() { return base::parameters(); }

  void compute() { m_copy = m_bare = m_x.get(); }

private:
  value_type &m_copy;
  value_type &m_bare;
  nm::input_property<value_type, nm::property_traits> const &m_x;
};

/// "value" = 2 e_01 of the strain "e::strain"; "symmetric" and "skew" are both d value/d e along symmetric
/// directions, but only "symmetric" is minor symmetric.
template <typename Traits>
class shear final : public nm::material_base<shear<Traits>, Traits> {
public:
  using base = nm::material_base<shear<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using tensor = tmech::tensor<value_type, 3, 2>;

  template <typename... Args>
  explicit shear(Args &&...args)
      : base(std::forward<Args>(args)...), m_value(base::template add_output<value_type>("value")),
        m_symmetric(base::template add_output<tensor>("symmetric")), m_skew(base::template add_output<tensor>("skew")),
        m_e(base::template add_input<tensor>("e", "strain")) {
    base::add_callback(&shear::compute, m_value, m_symmetric, m_skew).reads(m_e);
  }

  static input_parameter_controller parameters() { return base::parameters(); }

  void compute() {
    m_value = 2 * m_e.get()(0, 1);
    m_symmetric = tensor{};
    m_symmetric(0, 1) = m_symmetric(1, 0) = 1;
    m_skew = tensor{};
    m_skew(0, 1) = 2;
  }

private:
  value_type &m_value;
  tensor &m_symmetric;
  tensor &m_skew;
  nm::input_property<tensor, nm::property_traits> const &m_e;
};

/// "value": a host-bound vector history (codecs::vector), no callback.
template <typename Traits>
class vector_source final : public nm::material_base<vector_source<Traits>, Traits> {
public:
  using base = nm::material_base<vector_source<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit vector_source(Args &&...args) : base(std::forward<Args>(args)...) {
    base::template add_history_output<tmech::tensor<typename base::value_type, 3, 1>>("value", nm::codecs::vector{});
  }

  static input_parameter_controller parameters() { return base::parameters(); }
};

/// "value" = v·v and "gradient" = 2v of "v::value".
template <typename Traits>
class dot final : public nm::material_base<dot<Traits>, Traits> {
public:
  using base = nm::material_base<dot<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using vector = tmech::tensor<value_type, 3, 1>;

  template <typename... Args>
  explicit dot(Args &&...args)
      : base(std::forward<Args>(args)...), m_value(base::template add_output<value_type>("value")),
        m_gradient(base::template add_output<vector>("gradient")), m_v(base::template add_input<vector>("v", "value")) {
    base::add_callback(&dot::compute, m_value, m_gradient).reads(m_v);
  }

  static input_parameter_controller parameters() { return base::parameters(); }

  void compute() {
    m_value = tmech::dot(m_v.get(), m_v.get());
    m_gradient = 2 * m_v.get();
  }

private:
  value_type &m_value;
  vector &m_gradient;
  nm::input_property<vector, nm::property_traits> const &m_v;
};

TEST(DerivativeCheck, J2TangentMatchesOnPlasticStepsWithoutNamingHistories) {
  j2_point point;
  T worst_plastic{0}, alpha_before{0};
  int plastic_steps{0};
  for (int i{1}; i <= 20; ++i) {
    point.load(T{0.05} * i);
    auto const result{point.tangent()};
    EXPECT_EQ(result.directions.size(), 6u) << "a strain perturbs its six symmetric components";
    auto const alpha{point.ctx.get<T>("j2", "equivalent_plastic_strain")};
    if (alpha_before > T{1e-10}) {
      worst_plastic = std::max(worst_plastic, result.rel_error);
      ++plastic_steps;
    }
    alpha_before = alpha;
    point.ctx.commit();
  }
  ASSERT_GT(plastic_steps, 5);
  EXPECT_LT(worst_plastic, 1e-8) << "every history is reverted between perturbations";
}

TEST(DerivativeCheck, LeavesTheStateEvaluatedAtTheInput) {
  j2_point point;
  for (int i{1}; i <= 6; ++i) { // into the plastic range
    point.load(T{0.05} * i);
    point.ctx.commit();
  }
  point.load(T{0.35});
  tensor2 const strain{point.strain->strain().new_value()};
  tensor2 const stress{point.ctx.get<tensor2>("j2", "stress")};
  tensor2 const plastic{point.ctx.get<tensor2>("j2", "plastic_strain")};
  point.tangent();
  EXPECT_LT(tmech::norm(tensor2{point.ctx.get<tensor2>("j2", "stress") - stress}), 1e-12);
  EXPECT_LT(tmech::norm(tensor2{point.ctx.get<tensor2>("j2", "plastic_strain") - plastic}), 1e-14);
  EXPECT_LT(tmech::norm(tensor2{point.strain->strain().new_value() - strain}), 1e-15) << "the input is back at x";
  EXPECT_NO_THROW(point.ctx.commit()) << "the host commits the state at x as usual";
}

TEST(DerivativeCheck, AnInputFilledByACallbackIsRejected) {
  j2_point point{true};
  point.ctx.update();
  auto const message{thrown([&] { point.tangent(); })};
  EXPECT_NE(message.find("stepper::strain"), std::string::npos) << message;
  EXPECT_NE(message.find("callback"), std::string::npos) << message;
}

TEST(DerivativeCheck, AScalarDerivative) {
  scalar_point point{0.5};
  auto const result{point.slope(check("square::value", "x::state", "square::slope"))};
  ASSERT_EQ(result.directions.size(), 1u);
  EXPECT_NEAR(result.directions[0].analytical[0], 1.0, 1e-15) << "read at x";
  EXPECT_NEAR(result.directions[0].numerical[0], 1.0, 1e-8);
  EXPECT_LT(result.rel_error, 1e-8);
}

TEST(DerivativeCheck, ForwardDifferencesLeanByHalfTheSecondDerivativeTimesTheStep) {
  scalar_point point{0.5};
  auto p{check("square::value", "x::state", "square::slope")};
  p.insert<std::string>("scheme", "forward");
  p.insert<T>("step", T{1e-3});
  auto const result{point.slope(std::move(p))};
  EXPECT_NEAR(result.directions[0].numerical[0], 1.0 + 1e-3, 1e-9) << "(f(x+h) - f(x))/h = 2x + h";
}

TEST(DerivativeCheck, ARelativeStepScalesWithTheInput) {
  scalar_point point{1e3};
  auto p{check("square::value", "x::state", "square::slope")};
  p.insert<std::string>("scheme", "forward");
  p.insert<std::string>("step_kind", "relative");
  p.insert<T>("step", T{1e-6});
  auto const result{point.slope(std::move(p))};
  EXPECT_NEAR(result.directions[0].numerical[0] - 2e3, 1e-3, 1e-6) << "h = 1e-6 * |x|";
}

TEST(DerivativeCheck, AnUnknownSchemeIsRejected) {
  scalar_point point{0.5};
  auto p{check("square::value", "x::state", "square::slope")};
  p.insert<std::string>("scheme", "backward");
  auto const message{thrown([&] { point.slope(std::move(p)); })};
  EXPECT_NE(message.find("backward"), std::string::npos) << message;
}

TEST(DerivativeCheck, AThrowingEvaluationRethrowsWithTheInputBackAtX) {
  scalar_point point{1e6};
  auto p{check("square::value", "x::state", "square::slope")};
  p.insert<T>("step", T{1});
  EXPECT_THROW(point.slope(std::move(p)), std::runtime_error);
  EXPECT_EQ(point.x->state().new_value(), T{1e6});
  EXPECT_NO_THROW(point.ctx.update()) << "reverted: the context takes the next update";
}

TEST(DerivativeCheck, ComponentsSelectTheDirections) {
  j2_point point;
  point.load(T{0.1});
  auto p{check("j2::stress", "stepper::strain", "j2::tangent")};
  p.insert<std::vector<std::size_t>>("components", {0, 0, 1, 0});
  auto const result{point.ctx.check_derivative(std::move(p))};
  ASSERT_EQ(result.directions.size(), 2u);
  EXPECT_EQ(result.directions[0].label, "(0,0)");
  EXPECT_EQ(result.directions[1].label, "(0,1)") << "a symmetric input: (1,0) is the direction (0,1)";
  EXPECT_LT(result.rel_error, 1e-8);
}

TEST(DerivativeCheck, AnInputWithoutACodecIsRejected) {
  ctx_type ctx;
  add<frozen<policy>>(ctx, "frozen");
  ctx.finalize();
  auto const message{thrown([&] { ctx.check_derivative(check("frozen::value", "frozen::value", "frozen::value")); })};
  EXPECT_NE(message.find("frozen::value"), std::string::npos) << message;
  EXPECT_NE(message.find("codec"), std::string::npos) << message;
}

/// F is a two-point tensor: all nine components are perturbed.
TEST(DerivativeCheck, TheDeterminantOfANonSymmetricDeformationGradient) {
  ctx_type ctx;
  auto &F{add<nm::external_deformation_gradient_source<policy>>(ctx, "F")};
  add<cofactor<policy>>(ctx, "cof");
  ctx.finalize();
  tensor2 f{tmech::eye<T, 3, 2>()};
  f(0, 1) = 0.3;
  f(2, 0) = -0.2;
  f(1, 1) = 1.1;
  F.bind(tmech::eye<T, 3, 2>(), f);
  ctx.update();
  auto const result{ctx.check_derivative(check("cof::determinant", "F::deformation_gradient", "cof::cofactor"))};
  EXPECT_EQ(result.directions.size(), 9u);
  EXPECT_LT(result.rel_error, 1e-8);
}

TEST(DerivativeCheck, AnotherHostBoundInputKeepsItsValue) {
  ctx_type ctx;
  auto &t{add<nm::external_scalar_source<policy>>(ctx, "t")};
  auto &x{add<nm::external_scalar_source<policy>>(ctx, "x")};
  add<rate<policy>>(ctx, "rate");
  ctx.finalize();
  t.bind(T{1}, T{1.5});
  x.bind(T{0}, T{2});
  ctx.update();
  auto const result{ctx.check_derivative(check("rate::value", "x::state", "rate::slope"))};
  EXPECT_NEAR(result.directions[0].analytical[0], 0.5, 1e-15) << "dt = 0.5, not the reverted 0";
  EXPECT_NEAR(result.directions[0].numerical[0], 0.5, 1e-8);
  EXPECT_EQ(t.state().new_value(), T{1.5}) << "only the histories the callbacks fill are reverted";
  EXPECT_EQ(ctx.get<T>("rate", "value"), T{1});
}

TEST(DerivativeCheck, AnOutputAnotherCallbackFillsIsRejected) {
  ctx_type ctx;
  add<nm::external_scalar_source<policy>>(ctx, "x");
  add<bare<policy>>(ctx, "bare");
  add<square<policy>>(ctx, "square"); // reads x::state, a host-bound input
  ctx.finalize();
  ctx.update();
  auto const message{thrown([&] { ctx.check_derivative(check("bare::copy", "bare::bare", "bare::copy")); })};
  EXPECT_NE(message.find("bare::bare"), std::string::npos) << message;
  EXPECT_NE(message.find("callback"), std::string::npos) << message;
}

TEST(DerivativeCheck, ATangentThatIsNotMinorSymmetricFails) {
  ctx_type ctx;
  add<nm::external_strain_source<policy>>(ctx, "e");
  add<shear<policy>>(ctx, "shear");
  ctx.finalize();
  ctx.update();
  auto const symmetric{ctx.check_derivative(check("shear::value", "e::strain", "shear::symmetric"))};
  EXPECT_EQ(symmetric.asymmetry, 0.0);
  EXPECT_LT(symmetric.rel_error, 1e-8);
  auto const skew{ctx.check_derivative(check("shear::value", "e::strain", "shear::skew"))};
  EXPECT_LT(skew.directions[1].analytical[0] - skew.directions[1].numerical[0], 1e-8)
      << "(0,1): both see 2, the directions alone cannot tell";
  EXPECT_EQ(skew.asymmetry, 2.0) << "|A_01 - A_10|";
  EXPECT_GT(skew.rel_error, 0.5) << "the skew part counts as error";
}

TEST(DerivativeCheck, AWrongAnalyticalSizeLeavesTheStateAlone) {
  j2_point point;
  for (int i{1}; i <= 6; ++i) {
    point.load(T{0.05} * i);
    point.ctx.commit();
  }
  point.load(T{0.35});
  tensor2 const plastic{point.ctx.get<tensor2>("j2", "plastic_strain")};
  auto const message{thrown([&] { point.ctx.check_derivative(check("j2::stress", "stepper::strain", "j2::stress")); })};
  EXPECT_NE(message.find("components"), std::string::npos) << message;
  EXPECT_EQ(tmech::norm(tensor2{point.ctx.get<tensor2>("j2", "plastic_strain") - plastic}), 0.0)
      << "checked before anything is evaluated";
}

TEST(DerivativeCheck, AnInputTheOutputDoesNotDependOnIsRejected) {
  ctx_type ctx;
  add<nm::external_scalar_source<policy>>(ctx, "x");
  add<nm::external_scalar_source<policy>>(ctx, "y");
  add<square<policy>>(ctx, "square");
  ctx.finalize();
  ctx.update();
  auto const message{thrown([&] { ctx.check_derivative(check("square::value", "y::state", "square::slope")); })};
  EXPECT_NE(message.find("does not depend on"), std::string::npos) << message;
}

TEST(DerivativeCheck, AVectorInputHasThreeDirections) {
  ctx_type ctx;
  param_type p;
  add<vector_source<policy>>(ctx, "v");
  p.insert<std::string>("name", "dot");
  ctx.create<dot<policy>>(p);
  ctx.finalize();
  ctx.get_mutable<tmech::tensor<T, 3, 1>>("v", "value") = tmech::tensor<T, 3, 1>{0.5, -1.0, 2.0};
  ctx.update();
  auto const all{ctx.check_derivative(check("dot::value", "v::value", "dot::gradient"))};
  ASSERT_EQ(all.directions.size(), 3u);
  EXPECT_EQ(all.directions[2].label, "(2)");
  EXPECT_NEAR(all.directions[2].numerical[0], 4.0, 1e-7);
  EXPECT_LT(all.rel_error, 1e-8);
  auto q{check("dot::value", "v::value", "dot::gradient")};
  q.insert<std::vector<std::size_t>>("components", {1, 1});
  EXPECT_EQ(ctx.check_derivative(std::move(q)).directions.size(), 1u) << "a repeated component counts once";
}

TEST(DerivativeCheck, AnInfiniteStepOrANaNThresholdIsRejected) {
  scalar_point point{0.5};
  auto p{check("square::value", "x::state", "square::slope")};
  p.insert<T>("step", std::numeric_limits<T>::infinity());
  EXPECT_THROW(point.slope(std::move(p)), std::invalid_argument);
  auto q{check("square::value", "x::state", "square::slope")};
  q.insert<T>("report_threshold", std::numeric_limits<T>::quiet_NaN());
  EXPECT_THROW(point.slope(std::move(q)), std::invalid_argument);
}

} // namespace
