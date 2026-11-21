#include <gtest/gtest.h>
#include <cmath>
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

/// check_derivative(): name the output and the input; the graph perturbs the input along the directions of its
/// codec, reverting and updating as a host does.

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;

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
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "stepper");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    ctx.create<nm::linear_elasticity<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "solver");
    ctx.create<nm::local_newton<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "hardening");
    p.insert<std::string>("source", "j2");
    p.insert<T>("K", T{1000.0});
    ctx.create<nm::linear_isotropic_hardening<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "j2");
    p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<std::string>("solver_source", "solver");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    p.insert<T>("sigma_0", T{50.0});
    ctx.create<nm::j2_plasticity<policy>>(p);
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

/// "value" = x², "slope" = 2x from "x::state"; throws above x = 1.
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
    p.insert<std::string>("name", "x");
    x = &ctx.create<nm::external_scalar_source<policy>>(p);
    p.clear();
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
  param_type p;
  p.insert<std::string>("name", "frozen");
  ctx.create<frozen<policy>>(p);
  ctx.finalize();
  auto const message{thrown([&] { ctx.check_derivative(check("frozen::value", "frozen::value", "frozen::value")); })};
  EXPECT_NE(message.find("frozen::value"), std::string::npos) << message;
  EXPECT_NE(message.find("codec"), std::string::npos) << message;
}

/// F is a two-point tensor: all nine components are perturbed.
TEST(DerivativeCheck, TheDeterminantOfANonSymmetricDeformationGradient) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "F");
  auto &F{ctx.create<nm::external_deformation_gradient_source<policy>>(p)};
  p.clear();
  p.insert<std::string>("name", "cof");
  ctx.create<cofactor<policy>>(p);
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

} // namespace
