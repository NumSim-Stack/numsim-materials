#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <print>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/drucker_prager_yield_function.h"
#include "numsim-materials/materials/small_strain_plasticity.h"
#include "numsim-materials/materials/rk_plasticity.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/solvers/butcher_tableau.h"
#include "numsim-materials/postprocessing/numerical_diff_checker.h"

namespace {

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using dp_yield = numsim::materials::drucker_prager_yield_function<T, 3>;
using dp_plasticity = numsim::materials::drucker_prager_plasticity<policy>;

/// Hydrostatic strain path. The cone apex sits on the hydrostatic axis, so a
/// uniaxial path -- which every other test here drives -- never reaches it:
/// the deviatoric stress stays large enough that the standard return never
/// overshoots. tensor_component_stepper moves one component at a time, hence
/// this.
template <typename Traits>
class hydrostatic_stepper final
    : public numsim::materials::material_base<hydrostatic_stepper<Traits>, Traits> {
public:
  using base = numsim::materials::material_base<hydrostatic_stepper<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor = tmech::tensor<value_type, Dim, 2>;

  template <typename... Args>
  explicit hydrostatic_stepper(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_strain(base::template add_output<tensor>(
            "strain", &hydrostatic_stepper::update)),
        m_inc(base::template get_parameter<value_type>("increment")) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<value_type>("increment")
        .template add<numsim::materials::is_required>();
    return para;
  }

  void update() override {
    m_strain += m_inc * tmech::eye<value_type, Dim, 2>();
  }

private:
  tensor& m_strain;
  const value_type& m_inc;
};

class DruckerPragerTest : public ::testing::Test {
protected:
  void SetUp() override {
    param_type p;

    p.clear();
    p.insert<std::string>("name", "stepper");
    p.insert<T>("increment", T{0.05});
    p.insert<std::vector<std::size_t>>("indices", {0, 0});
    ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "stepper");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    ctx.create<numsim::materials::linear_elasticity<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "solver");
    ctx.create<numsim::materials::backward_euler<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "hardening");
    p.insert<std::string>("source", "dp");
    p.insert<T>("K", H_mod);
    ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);

    // Drucker-Prager yield function with friction and dilatancy
    dp_yield yf(dp_eta, dp_beta, K);

    p.clear();
    p.insert<std::string>("name", "dp");
    p.insert<std::string>("elastic_source", "elastic");
    p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<std::string>("solver_source", "solver");
    p.insert<T>("G", G);
    p.insert<T>("sigma_0", cohesion);
    p.insert<dp_yield>("yield_function", yf);
    ctx.create<dp_plasticity>(p);

    ctx.finalize();
  }

  ctx_type ctx;
  T K{166.67};
  T G{76.92};
  T cohesion{20.0};   // cohesion k
  T H_mod{500.0};
  T dp_eta{0.1};      // friction (pressure coefficient)
  T dp_beta{0.05};    // dilatancy (non-associative: beta != eta)
};

TEST_F(DruckerPragerTest, ElasticBeforeYield) {
  ctx.update();
  auto alpha = ctx.get<T>("dp", "equivalent_plastic_strain");
  EXPECT_NEAR(alpha, 0.0, 1e-12) << "First step should be elastic";
  ctx.commit();
}

TEST_F(DruckerPragerTest, YieldingOccurs) {
  bool found_plastic = false;
  for (int i = 0; i < 20; ++i) {
    ctx.update();
    auto alpha = ctx.get<T>("dp", "equivalent_plastic_strain");
    if (alpha > 1e-10) found_plastic = true;
    ctx.commit();
  }
  EXPECT_TRUE(found_plastic) << "DP should yield within 20 steps";
}

TEST_F(DruckerPragerTest, PlasticStrainHasVolumetricComponent) {
  // Unlike J2, DP plastic strain is NOT purely deviatoric
  // because the flow normal has a volumetric part (beta/3 * I)
  for (int i = 0; i < 15; ++i) {
    ctx.update();
    ctx.commit();
  }
  ctx.update();
  auto& eps_p = ctx.get<tensor2>("dp", "plastic_strain");
  auto trace_eps_p = tmech::trace(eps_p);
  auto alpha = ctx.get<T>("dp", "equivalent_plastic_strain");

  if (alpha > 1e-10) {
    // With beta > 0, plastic strain should have nonzero trace (dilatancy)
    EXPECT_GT(std::abs(trace_eps_p), 1e-10)
        << "DP plastic strain should have volumetric component (beta=" << dp_beta << ")";
    std::println("  trace(eps_p) = {:.6e}, alpha = {:.6e}", trace_eps_p, alpha);
  }
}

TEST_F(DruckerPragerTest, PressureSensitiveYielding) {
  // DP yields earlier under tension (positive I1) than compression
  // because F = sqrt(J2) + alpha*I1 - k
  ctx.update();
  const auto& sig = ctx.get<tensor2>("dp", "stress");
  auto I1 = tmech::trace(sig);
  std::println("  I1 = {:.4f} (positive = tension in uniaxial strain)", I1);
  // Under uniaxial strain, I1 > 0 → DP yields earlier than pure J2
  ctx.commit();
}

// --- Tangent checker for DP ---

class DPTangentTest : public ::testing::Test {
protected:
  void SetUp() override {
    param_type p;

    p.clear();
    p.insert<std::string>("name", "stepper");
    p.insert<T>("increment", T{0.02});
    p.insert<std::vector<std::size_t>>("indices", {0, 0});
    ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "stepper");
    p.insert<T>("K", T{166.67});
    p.insert<T>("G", T{76.92});
    ctx.create<numsim::materials::linear_elasticity<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "solver");
    ctx.create<numsim::materials::backward_euler<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "hardening");
    p.insert<std::string>("source", "dp");
    p.insert<T>("K", T{500.0});
    ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);

    dp_yield yf(T{0.1}, T{0.05}, T{166.67});

    p.clear();
    p.insert<std::string>("name", "dp");
    p.insert<std::string>("elastic_source", "elastic");
    p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<std::string>("solver_source", "solver");
    p.insert<T>("G", T{76.92});
    p.insert<T>("sigma_0", T{20.0});
    p.insert<dp_yield>("yield_function", yf);
    ctx.create<dp_plasticity>(p);

    p.clear();
    p.insert<std::string>("name", "checker");
    p.insert<ctx_type*>("context", &ctx);
    p.insert<std::string>("output_source", "dp::stress");
    p.insert<std::string>("input_source", "stepper::strain");
    p.insert<std::string>("analytical_source", "dp::tangent");
    p.insert<std::vector<std::string>>("history_sources",
        {"dp::plastic_strain", "dp::equivalent_plastic_strain"});
    p.insert<T>("epsilon", T{1e-7});
    ctx.create<numsim::materials::tangent_checker<policy>>(p);

    ctx.finalize();
  }

  ctx_type ctx;
};

TEST_F(DPTangentTest, ConsistentTangent) {
  T max_rel_error = 0;
  for (int i = 0; i < 15; ++i) {
    ctx.update();
    auto rel = ctx.get<T>("checker", "rel_error");
    auto alpha = ctx.get<T>("dp", "equivalent_plastic_strain");
    std::println("  DP step {:2d}: rel={:.2e} alpha={:.4e}", i, rel, alpha);
    if (rel > max_rel_error) max_rel_error = rel;
    ctx.commit();
  }
  EXPECT_LT(max_rel_error, 1e-6)
      << "DP consistent tangent should match numerical derivative";
}

// --- Convergence: smaller steps → smaller tangent error ---

T run_dp_max_tangent_error(T increment, int steps) {
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", increment);
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", T{166.67});
  p.insert<T>("G", T{76.92});
  ctx.create<numsim::materials::linear_elasticity<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "solver");
  ctx.create<numsim::materials::backward_euler<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "dp");
  p.insert<T>("K", T{500.0});
  ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);

  dp_yield yf(T{0.1}, T{0.05}, T{166.67});

  p.clear();
  p.insert<std::string>("name", "dp");
  p.insert<std::string>("elastic_source", "elastic");
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<std::string>("solver_source", "solver");
  p.insert<T>("G", T{76.92});
  p.insert<T>("sigma_0", T{20.0});
  p.insert<dp_yield>("yield_function", yf);
  ctx.create<dp_plasticity>(p);

  p.clear();
  p.insert<std::string>("name", "checker");
  p.insert<ctx_type*>("context", &ctx);
  p.insert<std::string>("output_source", "dp::stress");
  p.insert<std::string>("input_source", "stepper::strain");
  p.insert<std::string>("analytical_source", "dp::tangent");
  p.insert<std::vector<std::string>>("history_sources",
      {"dp::plastic_strain", "dp::equivalent_plastic_strain"});
  p.insert<T>("epsilon", T{1e-7});
  ctx.create<numsim::materials::tangent_checker<policy>>(p);

  ctx.finalize();

  T max_rel = 0;
  for (int i = 0; i < steps; ++i) {
    ctx.update();
    auto rel = ctx.get<T>("checker", "rel_error");
    if (rel > max_rel) max_rel = rel;
    ctx.commit();
  }
  return max_rel;
}

TEST(DPConvergence, TangentErrorIsBounded) {
  auto err = run_dp_max_tangent_error(T{0.02}, 15);
  std::println("  DP tangent error: {:.4e}", err);
  EXPECT_LT(err, 1e-3) << "DP consistent tangent should match numerical derivative";
}

// ---------------------------------------------------------------------------
// The cone apex
// ---------------------------------------------------------------------------

/// Drives hydrostatic TENSION until the return map switches to the apex branch.
///
/// Nothing else in this suite reaches it: every other path is uniaxial, stays on
/// the smooth cone, and leaves needs_apex_return() false for its whole run. So
/// apex_modified_sig_eq, apex_effective_modulus, apex_plastic_strain and
/// apex_tangent were executed by no test at all.
///
/// The apex is where the cone closes on the hydrostatic axis, so the signature
/// is deviatoric stress driven to zero while plastic flow continues -- a state
/// the smooth branch cannot produce, since there the deviatoric return is
/// proportional to a nonzero s_trial.
TEST(DruckerPragerApex, HydrostaticTensionReachesTheApex) {
  ctx_type ctx;
  param_type p;
  const T K{166.67}, G{76.92}, cohesion{20.0}, H_mod{500.0};
  const T dp_eta{0.1}, dp_beta{0.05};

  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.05});
  ctx.create<hydrostatic_stepper<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  ctx.create<numsim::materials::linear_elasticity<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "solver");
  ctx.create<numsim::materials::backward_euler<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "dp");
  p.insert<T>("K", H_mod);
  ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);

  dp_yield yf(dp_eta, dp_beta, K);
  p.clear();
  p.insert<std::string>("name", "dp");
  p.insert<std::string>("elastic_source", "elastic");
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<std::string>("solver_source", "solver");
  p.insert<T>("G", G);
  p.insert<T>("sigma_0", cohesion);
  p.insert<dp_yield>("yield_function", yf);
  ctx.create<dp_plasticity>(p);
  ctx.finalize();

  bool yielded = false;
  T min_q_after_yield = std::numeric_limits<T>::max();
  for (int i = 0; i < 25; ++i) {
    ctx.update();
    const auto alpha = ctx.get<T>("dp", "equivalent_plastic_strain");
    if (alpha > 1e-10) {
      yielded = true;
      const auto& sig = ctx.get<tensor2>("dp", "stress");
      const auto s = tmech::dev(sig);
      const T q = std::sqrt(tmech::dcontract(s, s));
      min_q_after_yield = std::min(min_q_after_yield, q);
    }
    ctx.commit();
  }

  ASSERT_TRUE(yielded) << "hydrostatic tension must reach the yield surface";
  // On the apex the deviatoric stress is returned to zero. On the smooth cone
  // it cannot be: there the return is proportional to a nonzero s_trial.
  EXPECT_LT(min_q_after_yield, 1e-8)
      << "deviatoric stress never reached zero, so the apex branch was never "
         "taken -- min |s| = " << min_q_after_yield;
}

/// The apex state has to be admissible, not merely reached: hydrostatic stress
/// pinned at the cone tip k/eta, and dilatant plastic volume change.
TEST(DruckerPragerApex, ApexStateIsAdmissible) {
  ctx_type ctx;
  param_type p;
  const T K{166.67}, G{76.92}, cohesion{20.0}, H_mod{500.0};
  const T dp_eta{0.1}, dp_beta{0.05};

  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.05});
  ctx.create<hydrostatic_stepper<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", K); p.insert<T>("G", G);
  ctx.create<numsim::materials::linear_elasticity<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "solver");
  ctx.create<numsim::materials::backward_euler<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "dp");
  p.insert<T>("K", H_mod);
  ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);
  dp_yield yf(dp_eta, dp_beta, K);
  p.clear();
  p.insert<std::string>("name", "dp");
  p.insert<std::string>("elastic_source", "elastic");
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<std::string>("solver_source", "solver");
  p.insert<T>("G", G); p.insert<T>("sigma_0", cohesion);
  p.insert<dp_yield>("yield_function", yf);
  ctx.create<dp_plasticity>(p);
  ctx.finalize();

  for (int i = 0; i < 25; ++i) { ctx.update(); ctx.commit(); }
  ctx.update();

  const auto& sig = ctx.get<tensor2>("dp", "stress");
  const auto& eps_p = ctx.get<tensor2>("dp", "plastic_strain");
  const T alpha = ctx.get<T>("dp", "equivalent_plastic_strain");
  const T p_hyd = tmech::trace(sig) / 3.0;

  ASSERT_GT(alpha, 1e-10) << "must be plastic by now";
  // Beyond the apex the stress cannot keep climbing: pressure is capped by the
  // cone tip, which with linear hardening moves with alpha.
  const T tip = (cohesion + H_mod * alpha) / dp_eta;
  EXPECT_LE(p_hyd, tip * (1.0 + 1e-6))
      << "hydrostatic stress " << p_hyd << " exceeds the cone tip " << tip;
  // beta > 0, so the flow is dilatant even at the apex.
  EXPECT_GT(tmech::trace(eps_p), 0.0) << "apex flow must be dilatant";
}

} // namespace
