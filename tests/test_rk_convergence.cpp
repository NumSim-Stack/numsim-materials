/// Non-convergence must be reported, not absorbed.
///
/// Two solvers in the Runge-Kutta family used to run out of iterations
/// silently. rk_integrator wrote its unconverged stage value into the state
/// update with no way for a caller to find out, and j2_rk_plasticity's
/// implicit stage left the PREVIOUS call's flow direction in a member and used
/// it as if it belonged to this step.
#include <gtest/gtest.h>
#include <cmath>
#include "numsim-materials/core/history_property.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/default_materials.h"
#include "numsim-materials/materials/j2_rk_plasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/solvers/rk_integrator.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using RK = nm::rk_integrator<policy>;
using stepper2 = nm::tensor_component_stepper<2, policy>;

/// dy/dt = -lambda*y, with a deliberately wrong derivative so the implicit
/// stage Newton cannot converge when we want it not to.
template <typename Traits>
class decay_with_scaled_derivative final
    : public nm::material_base<decay_with_scaled_derivative<Traits>, Traits> {
public:
  using base = nm::material_base<decay_with_scaled_derivative<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit decay_with_scaled_derivative(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_rate(base::template add_output<value_type>(
            "rate", &decay_with_scaled_derivative::compute)),
        m_drate(base::template add_output<value_type>("rate_derivative")),
        m_lambda(base::template get_parameter<value_type>("lambda")),
        m_dscale(base::template get_parameter<value_type>("derivative_scale")),
        m_source(base::template get_parameter<std::string>("source")),
        m_y(base::template add_input<value_type>(m_source, "state",
                                                 nm::EdgeKind::Local)) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("source")
        .template add<numsim_core::is_required>();
    para.template insert<value_type>("lambda")
        .template add<numsim_core::is_required>();
    // 1.0 is the true derivative. Anything else is a wrong Jacobian, which
    // slows or prevents convergence without changing the solution the
    // residual defines.
    para.template insert<value_type>("derivative_scale")
        .template add<numsim_core::set_default>(value_type{1});
    return para;
  }

  void compute() {
    m_rate = -m_lambda * m_y.get();
    m_drate = -m_lambda * m_dscale;
  }

private:
  value_type& m_rate;
  value_type& m_drate;
  const value_type& m_lambda;
  const value_type& m_dscale;
  const std::string& m_source;
  const nm::input_property<value_type, nm::property_traits>& m_y;
};

/// One step of the given tableau; returns the integrator so the caller can
/// ask it whether it converged.
RK& run_one_step(ctx_type& ctx, param_type& p, const std::string& tableau,
                 int max_iter, T derivative_scale = T{1}) {
  p.clear();
  p.insert<std::string>("name", "integrator");
  p.insert<std::string>("function", "decay");
  p.insert<T>("step_size", T{0.5});
  p.insert<T>("tolerance", T{1e-12});
  p.insert<int>("max_iter", max_iter);
  p.insert<std::string>("tableau", tableau);
  auto& integ = ctx.create<RK>(p);

  p.clear();
  p.insert<std::string>("name", "decay");
  p.insert<std::string>("source", "integrator");
  p.insert<T>("lambda", T{1});
  p.insert<T>("derivative_scale", derivative_scale);
  ctx.create<decay_with_scaled_derivative<policy>>(p);

  ctx.finalize();
  auto* hist = dynamic_cast<numsim_core::history_property<T, nm::property_traits>*>(
      ctx.find_property("integrator", "state"));
  hist->old_value() = T{1};
  hist->new_value() = T{1};
  ctx.update();
  return integ;
}

// --- rk_integrator::converged() (#61) ---

TEST(RkConvergence, AnExplicitTableauIsAlwaysConverged) {
  ctx_type ctx;
  param_type p;
  // max_iter is irrelevant here: an explicit tableau never iterates, so
  // "converged" must not depend on it.
  auto& integ = run_one_step(ctx, p, "forward_euler", 1);
  EXPECT_TRUE(integ.converged());
}

TEST(RkConvergence, AConvergedImplicitStepReportsSuccess) {
  ctx_type ctx;
  param_type p;
  auto& integ = run_one_step(ctx, p, "implicit_euler", 50);
  EXPECT_TRUE(integ.converged());
}

/// The discriminating case. One iteration is not enough to reach 1e-12, and
/// the old code reported nothing at all -- the caller could not distinguish
/// this from the converged step above.
TEST(RkConvergence, AnImplicitStageThatRunsOutOfIterationsReportsFailure) {
  ctx_type ctx;
  param_type p;
  auto& integ = run_one_step(ctx, p, "implicit_euler", 1);
  EXPECT_FALSE(integ.converged())
      << "one Newton iteration cannot reach a 1e-12 tolerance, but the "
         "integrator reported success";
}

TEST(RkConvergence, AFullyImplicitTableauAlsoReportsFailure) {
  ctx_type ctx;
  param_type p;
  auto& integ = run_one_step(ctx, p, "gauss_legendre_4", 1);
  EXPECT_FALSE(integ.converged());
}

/// converged() must describe the LAST step, not latch on the first failure.
///
/// The same solver instance is driven both ways: max_iter is bound by
/// reference, so raising it through set_parameter changes the next solve
/// without rebuilding the graph. A flag that were only ever set to false, or
/// only ever set once, fails here while passing every test above.
TEST(RkConvergence, TheFlagIsRecomputedEveryStep) {
  ctx_type ctx;
  param_type p;
  auto& integ = run_one_step(ctx, p, "implicit_euler", 1);
  ASSERT_FALSE(integ.converged()) << "expected the one-iteration solve to fail";

  // The residual is checked at the top of the loop, so a linear problem needs
  // two passes: one to take the (exact) Newton step, one to see it converged.
  integ.template set_parameter<int>("max_iter", 2);
  ctx.update();
  EXPECT_TRUE(integ.converged())
      << "converged() stayed false after a solve that did converge";

  // ...and back again, so it is not latching true either.
  integ.template set_parameter<int>("max_iter", 1);
  ctx.update();
  EXPECT_FALSE(integ.converged())
      << "converged() stayed true after a solve that did not converge";
}

// --- rk_integrator reachable from a document (#61) ---

TEST(RkConvergence, RkIntegratorIsRegistered) {
  nm::register_default_materials<policy>();
  const auto types = nm::object_store<policy>::factory_type::instance()
                         .registered_types();
  EXPECT_NE(std::find(types.begin(), types.end(), "rk_integrator"), types.end())
      << "rk_integrator derives material_base but cannot be named in a "
         "document";
}

// --- j2_rk_plasticity: a stage that fails must throw (#58) ---

/// Builds a J2 Runge-Kutta model on a plastic step, with max_iter as given.
void run_j2_rk(int max_iter, const std::string& tableau) {
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.02});  // well past yield
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<stepper2>(p);

  p.clear();
  p.insert<std::string>("name", "hard");
  p.insert<std::string>("source", "j2");
  p.insert<T>("K", T{1000.0});
  ctx.create<nm::linear_isotropic_hardening<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "j2");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<std::string>("hardening_source", "hard");
  p.insert<T>("K", T{166666.7});
  p.insert<T>("G", T{76923.1});
  p.insert<T>("sigma_0", T{200.0});
  p.insert<T>("tolerance", T{1e-12});
  p.insert<int>("max_iter", max_iter);
  p.insert<std::string>("tableau", tableau);
  ctx.create<nm::j2_rk_plasticity<policy>>(p);

  ctx.finalize();
  ctx.update();
}

TEST(RkConvergence, AJ2RkPlasticStepConvergesWithEnoughIterations) {
  EXPECT_NO_THROW(run_j2_rk(50, "implicit_euler"));
}

/// Before this, the stage kept whatever m_N_stage held from the previous
/// call -- a different material point's flow direction -- and reported
/// nothing. j2_plasticity and drucker_prager_plasticity both throw here.
TEST(RkConvergence, AJ2RkStageThatRunsOutOfIterationsThrows) {
  EXPECT_THROW(run_j2_rk(1, "implicit_euler"), std::runtime_error);
}

TEST(RkConvergence, TheJ2RkFailureNamesTheMaterialAndTheStage) {
  try {
    run_j2_rk(1, "implicit_euler");
    FAIL() << "a non-converged implicit stage was accepted";
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("j2_rk_plasticity"), std::string::npos) << msg;
    EXPECT_NE(msg.find("'j2'"), std::string::npos)
        << "the message does not name the material instance: " << msg;
    EXPECT_NE(msg.find("stage"), std::string::npos) << msg;
  }
}

} // namespace
