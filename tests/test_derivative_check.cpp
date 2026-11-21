#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/driver/derivative_check.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/materials/j2_plasticity.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/solvers/local_newton.h"

/// check_derivative() differentiates along the host path: revert(), set the input, update(), read the output.

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;

constexpr T K{166.67}, G{76.92}, sigma_0{50.0}, H_mod{1000.0};

/// J2 with linear hardening behind a host-bound strain "stepper::strain".
struct j2_point {
  ctx_type ctx;
  nm::external_strain_source<policy> *strain{nullptr};
  tensor2 old_strain{tmech::zeros<T, 3, 2>()};

  j2_point() {
    param_type p;
    p.insert<std::string>("name", "stepper");
    strain = &ctx.create<nm::external_strain_source<policy>>(p);
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
    p.insert<T>("K", H_mod);
    ctx.create<nm::linear_isotropic_hardening<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "j2");
    p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<std::string>("solver_source", "solver");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    p.insert<T>("sigma_0", sigma_0);
    ctx.create<nm::j2_plasticity<policy>>(p);
    ctx.finalize();
  }

  /// The host binds the strain at the end of the increment.
  auto set_strain() {
    return [this](tensor2 const &e) { strain->bind(old_strain, e); };
  }

  auto check(tensor2 const &e) { return nm::check_derivative<tensor2>(ctx, "j2::stress", "j2::tangent", e, set_strain()); }
};

tensor2 uniaxial(T value) {
  tensor2 e{tmech::zeros<T, 3, 2>()};
  e(0, 0) = value;
  return e;
}

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
    if (m_x.get() > 1)
      throw std::runtime_error("square: above 1");
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

  scalar_point() {
    param_type p;
    p.insert<std::string>("name", "x");
    x = &ctx.create<nm::external_scalar_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "square");
    ctx.create<square<policy>>(p);
    ctx.finalize();
  }

  auto set_x() {
    return [this](T v) { x->bind(T{0}, v); };
  }
};

TEST(DerivativeCheck, J2TangentMatchesOnPlasticStepsWithoutNamingHistories) {
  j2_point point;
  T worst_plastic{0}, alpha_before{0};
  int plastic_steps{0};
  for (int i{1}; i <= 20; ++i) {
    auto const e{uniaxial(T{0.05} * i)};
    point.set_strain()(e);
    point.ctx.update();
    auto const check{point.check(e)};
    auto const alpha{point.ctx.get<T>("j2", "equivalent_plastic_strain")};
    if (alpha_before > T{1e-10}) {
      worst_plastic = std::max(worst_plastic, check.rel_error);
      ++plastic_steps;
    }
    alpha_before = alpha;
    point.ctx.commit();
    point.old_strain = e;
  }
  ASSERT_GT(plastic_steps, 5);
  EXPECT_LT(worst_plastic, 1e-8) << "every history is reverted between perturbations";
}

TEST(DerivativeCheck, LeavesTheStateEvaluatedAtTheInput) {
  j2_point point;
  for (int i{1}; i <= 6; ++i) { // into the plastic range
    point.set_strain()(uniaxial(T{0.05} * i));
    point.ctx.update();
    point.ctx.commit();
    point.old_strain = uniaxial(T{0.05} * i);
  }
  auto const e{uniaxial(T{0.35})};
  point.set_strain()(e);
  point.ctx.update();
  tensor2 const stress{point.ctx.get<tensor2>("j2", "stress")};
  tensor2 const plastic{point.ctx.get<tensor2>("j2", "plastic_strain")};
  point.check(e);
  EXPECT_LT(tmech::norm(point.ctx.get<tensor2>("j2", "stress") - stress), 1e-12);
  EXPECT_LT(tmech::norm(point.ctx.get<tensor2>("j2", "plastic_strain") - plastic), 1e-14);
  EXPECT_LT(tmech::norm(point.strain->strain().new_value() - e), 1e-15) << "the input is bound at x again";
  EXPECT_NO_THROW(point.ctx.commit()) << "the host commits the state at x as usual";
}

TEST(DerivativeCheck, AScalarDerivative) {
  scalar_point point;
  auto const check{nm::check_derivative<T>(point.ctx, "square::value", "square::slope", T{0.5}, point.set_x())};
  EXPECT_NEAR(check.analytical, 1.0, 1e-15) << "read at x, evaluated by the check";
  EXPECT_NEAR(check.numerical, 1.0, 1e-8);
  EXPECT_LT(check.rel_error, 1e-8);
}

TEST(DerivativeCheck, AThrowingEvaluationRethrowsWithTheInputBoundAtX) {
  scalar_point point;
  EXPECT_THROW(nm::check_derivative<T>(point.ctx, "square::value", "square::slope", T{1}, point.set_x(), T{1e-3}),
               std::runtime_error);
  EXPECT_EQ(point.x->state().new_value(), T{1});
  EXPECT_NO_THROW(point.ctx.update()) << "reverted: the context takes the next update";
}

} // namespace
