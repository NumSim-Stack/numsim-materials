#include <gtest/gtest.h>
#include <algorithm>
#include <print>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/drucker_prager_plasticity.h"
#include "numsim-materials/materials/j2_plasticity.h"
#include "numsim-materials/materials/j2_rk_plasticity.h"
#include "numsim-materials/solvers/local_newton.h"
#include "numsim-materials/solvers/butcher_tableau.h"
#include "numsim-materials/postprocessing/numerical_diff_checker.h"

namespace {

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;

class J2PlasticityTest : public ::testing::Test {
protected:
  void SetUp() override {
    param_type p;

    // Strain stepper: uniaxial loading
    p.clear();
    p.insert<std::string>("name", "stepper");
    p.insert<T>("increment", T{0.05});
    p.insert<std::vector<std::size_t>>("indices", {0, 0});
    ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

    // Linear elasticity (trial stress provider)
    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "stepper");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    ctx.create<numsim::materials::linear_elasticity<policy>>(p);

    // Newton-Raphson solver
    p.clear();
    p.insert<std::string>("name", "solver");
    ctx.create<numsim::materials::local_newton<policy>>(p);

    // Linear isotropic hardening (Local edge — called in inner loop)
    p.clear();
    p.insert<std::string>("name", "hardening");
    p.insert<std::string>("source", "j2");
    p.insert<T>("K", H_mod);
    ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);

    // J2 plasticity — solver passed as pointer
    p.clear();
    p.insert<std::string>("name", "j2");
      p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<std::string>("solver_source", "solver");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    p.insert<T>("sigma_0", sigma_0);
    ctx.create<numsim::materials::j2_plasticity<policy>>(p);

    ctx.finalize();
  }

  ctx_type ctx;
  T K{166.67};      // Bulk modulus
  T G{76.92};       // Shear modulus
  T sigma_0{50.0};  // Initial yield stress [MPa]
  T H_mod{1000.0};  // Hardening modulus [MPa]
};

TEST_F(J2PlasticityTest, ElasticBeforeYield) {
  for (int i = 0; i < 3; ++i) {
    ctx.update();
    auto alpha = ctx.get<T>("j2", "equivalent_plastic_strain");
    EXPECT_NEAR(alpha, 0.0, 1e-12) << "Step " << i << " should be elastic";
    ctx.commit();
  }
}

TEST_F(J2PlasticityTest, YieldingOccurs) {
  bool found_plastic = false;
  for (int i = 0; i < 20; ++i) {
    ctx.update();
    auto alpha = ctx.get<T>("j2", "equivalent_plastic_strain");
    if (alpha > 1e-10) found_plastic = true;
    ctx.commit();
  }
  EXPECT_TRUE(found_plastic) << "Plasticity should activate within 20 steps";
}

TEST_F(J2PlasticityTest, StressDoesNotExceedYieldSurface) {
  auto I = tmech::eye<T, 3, 2>();
  for (int i = 0; i < 30; ++i) {
    ctx.update();
    auto& sig = ctx.get<tensor2>("j2", "stress");
    auto trace_sig = tmech::trace(sig);
    auto sig_dev = sig - (trace_sig / T{3}) * I;
    auto sig_eq = std::sqrt(T{1.5} * tmech::dcontract(sig_dev, sig_dev));
    auto alpha = ctx.get<T>("j2", "equivalent_plastic_strain");
    auto yield_stress = sigma_0 + H_mod * alpha;

    // σ_eq should not exceed σ_0 + H(α) (within tolerance)
    EXPECT_LE(sig_eq, yield_stress + T{1.0})
        << "Step " << i << ": σ_eq=" << sig_eq << " > σ_y=" << yield_stress;
    ctx.commit();
  }
}

TEST_F(J2PlasticityTest, PlasticStrainIsDeviatoric) {
  for (int i = 0; i < 20; ++i) {
    ctx.update();
    ctx.commit();
  }
  ctx.update();
  auto& eps_p = ctx.get<tensor2>("j2", "plastic_strain");
  auto trace_eps_p = tmech::trace(eps_p);
  EXPECT_NEAR(trace_eps_p, 0.0, 1e-10)
      << "Plastic strain must be deviatoric (trace = 0)";
}

TEST_F(J2PlasticityTest, HardeningIncreasesYieldStress) {
  T prev_alpha = 0;
  for (int i = 0; i < 30; ++i) {
    ctx.update();
    auto alpha = ctx.get<T>("j2", "equivalent_plastic_strain");
    EXPECT_GE(alpha, prev_alpha) << "α must be monotonically increasing";
    prev_alpha = alpha;
    ctx.commit();
  }
  EXPECT_GT(prev_alpha, 0.0) << "Should have accumulated plastic strain";
}

// --- Tangent checker ---

class J2TangentTest : public ::testing::Test {
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
    p.insert<T>("K", T{166.67});
    p.insert<T>("G", T{76.92});
    ctx.create<numsim::materials::linear_elasticity<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "solver");
    ctx.create<numsim::materials::local_newton<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "hardening");
    p.insert<std::string>("source", "j2");
    p.insert<T>("K", T{1000.0});
    ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "j2");
      p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<std::string>("solver_source", "solver");
    p.insert<T>("K", T{166.67});
    p.insert<T>("G", T{76.92});
    p.insert<T>("sigma_0", T{50.0});
    ctx.create<numsim::materials::j2_plasticity<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "checker");
    p.insert<ctx_type*>("context", &ctx);
    p.insert<std::string>("output_source", "j2::stress");
    p.insert<std::string>("input_source", "stepper::strain");
    p.insert<std::string>("analytical_source", "j2::tangent");
    p.insert<std::vector<std::string>>("history_sources",
        {"j2::plastic_strain", "j2::equivalent_plastic_strain"});
    p.insert<T>("epsilon", T{1e-7});
    ctx.create<numsim::materials::tangent_checker<policy>>(p);

    ctx.finalize();
  }

  ctx_type ctx;
};

TEST_F(J2TangentTest, ConsistentTangentAllSteps) {
  // The elastic->plastic transition step is genuinely inexact: the tangent is
  // discontinuous across the yield surface, so a central difference straddling
  // it averages two different tangents. That step gets a loose bound.
  //
  // Every OTHER step is fully plastic and matches at ~1e-10. Bounding the whole
  // run by the transition step's 10% made the check blind: a 0.5% error in the
  // closed-form tangent still landed at 2.4e-4 and passed. The two regimes are
  // now bounded separately.
  T worst_plastic = 0, worst_transition = 0;
  T alpha_before = 0;
  int plastic_steps = 0;
  for (int i = 0; i < 20; ++i) {
    ctx.update();
    const auto rel = ctx.get<T>("checker", "rel_error");
    const auto alpha = ctx.get<T>("j2", "equivalent_plastic_strain");
    const bool was_plastic = alpha_before > T{1e-10};
    const bool is_plastic = alpha > T{1e-10};
    if (was_plastic && is_plastic) {          // fully inside the plastic regime
      worst_plastic = std::max(worst_plastic, rel);
      ++plastic_steps;
    } else if (is_plastic) {                  // the crossing step
      worst_transition = std::max(worst_transition, rel);
    }
    alpha_before = alpha;
    ctx.commit();
  }

  ASSERT_GT(plastic_steps, 5) << "the path must spend real time yielding";
  EXPECT_LT(worst_plastic, 1e-8)
      << "the consistent tangent must match the numerical derivative on fully "
         "plastic steps (worst " << worst_plastic << ")";
  EXPECT_LT(worst_transition, 0.1)
      << "the yield-crossing step is inexact but should stay bounded";
}

// --- RK plasticity: multi-stage return mapping via tableau parameter ---

class RKPlasticityTest : public ::testing::Test {
protected:
  void setup_with_tableau(const std::string& name) {
    m_tableau_name = name;
    param_type p;

    p.clear();
    p.insert<std::string>("name", "stepper");
    p.insert<T>("increment", T{0.05});
    p.insert<std::vector<std::size_t>>("indices", {0, 0});
    ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "stepper");
    p.insert<T>("K", T{166.67});
    p.insert<T>("G", T{76.92});
    ctx.create<numsim::materials::linear_elasticity<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "hardening");
    p.insert<std::string>("source", "j2");
    p.insert<T>("K", T{1000.0});
    ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "j2");
    p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<T>("K", T{166.67});
    p.insert<T>("G", T{76.92});
    p.insert<T>("sigma_0", T{50.0});
    p.insert<std::string>("tableau", m_tableau_name);
    ctx.create<numsim::materials::j2_rk_plasticity<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "checker");
    p.insert<ctx_type*>("context", &ctx);
    p.insert<std::string>("output_source", "j2::stress");
    p.insert<std::string>("input_source", "stepper::strain");
    p.insert<std::string>("analytical_source", "j2::tangent");
    p.insert<std::vector<std::string>>("history_sources",
        {"j2::plastic_strain", "j2::equivalent_plastic_strain"});
    p.insert<T>("epsilon", T{1e-7});
    ctx.create<numsim::materials::tangent_checker<policy>>(p);

    ctx.finalize();
  }

  ctx_type ctx;
  std::string m_tableau_name;
};

TEST_F(RKPlasticityTest, ImplicitEulerMatchesMonolithic) {
  setup_with_tableau("implicit_euler");

  // Split by regime, for the reason recorded on
  // J2TangentTest.ConsistentTangentAllSteps: the yield-crossing step is
  // genuinely inexact because a central difference straddles the tangent's
  // discontinuity, and bounding the whole run by that step makes the check
  // blind to everything else.
  //
  // The bound this test used to carry was 0.1 against a measured worst of
  // 3.0e-4 -- 336x of headroom. Dropping the ENTIRE plastic-softening
  // correction from the tangent lands at about 1.2e-2 and still passed.
  //
  // Unlike the monolithic J2 tangent, which is exact to ~1e-10, this one is a
  // genuine approximation: the stage-wise return map's tangent is assembled
  // from the b-weighted multiplier rather than differentiated through the
  // stages. 1e-3 is a little over 3x the measured worst and an order of
  // magnitude below a dropped softening term.
  T worst_plastic = 0, worst_transition = 0, alpha_before = 0;
  int plastic_steps = 0;
  for (int i = 0; i < 20; ++i) {
    ctx.update();
    const auto rel = ctx.get<T>("checker", "rel_error");
    const auto alpha = ctx.get<T>("j2", "equivalent_plastic_strain");
    const bool was_plastic = alpha_before > T{1e-10};
    const bool is_plastic = alpha > T{1e-10};
    if (was_plastic && is_plastic) {
      worst_plastic = std::max(worst_plastic, rel);
      ++plastic_steps;
    } else if (is_plastic) {
      worst_transition = std::max(worst_transition, rel);
    }
    alpha_before = alpha;
    ctx.commit();
  }

  ASSERT_GT(plastic_steps, 5) << "the path must spend real time yielding";
  EXPECT_LT(worst_plastic, 1e-3)
      << "implicit Euler RK tangent disagrees with the numerical derivative on fully "
         "plastic steps (worst " << worst_plastic << ")";
  EXPECT_LT(worst_transition, 0.1)
      << "the yield-crossing step is inexact but should stay bounded";
}

TEST_F(RKPlasticityTest, SDIRK3TangentCheck) {
  setup_with_tableau("sdirk3");

  // Split by regime, for the reason recorded on
  // J2TangentTest.ConsistentTangentAllSteps: the yield-crossing step is
  // genuinely inexact because a central difference straddles the tangent's
  // discontinuity, and bounding the whole run by that step makes the check
  // blind to everything else.
  //
  // The bound this test used to carry was 0.1 against a measured worst of
  // 3.0e-4 -- 336x of headroom. Dropping the ENTIRE plastic-softening
  // correction from the tangent lands at about 1.2e-2 and still passed.
  //
  // Unlike the monolithic J2 tangent, which is exact to ~1e-10, this one is a
  // genuine approximation: the stage-wise return map's tangent is assembled
  // from the b-weighted multiplier rather than differentiated through the
  // stages. 1e-3 is a little over 3x the measured worst and an order of
  // magnitude below a dropped softening term.
  T worst_plastic = 0, worst_transition = 0, alpha_before = 0;
  int plastic_steps = 0;
  for (int i = 0; i < 20; ++i) {
    ctx.update();
    const auto rel = ctx.get<T>("checker", "rel_error");
    const auto alpha = ctx.get<T>("j2", "equivalent_plastic_strain");
    const bool was_plastic = alpha_before > T{1e-10};
    const bool is_plastic = alpha > T{1e-10};
    if (was_plastic && is_plastic) {
      worst_plastic = std::max(worst_plastic, rel);
      ++plastic_steps;
    } else if (is_plastic) {
      worst_transition = std::max(worst_transition, rel);
    }
    alpha_before = alpha;
    ctx.commit();
  }

  ASSERT_GT(plastic_steps, 5) << "the path must spend real time yielding";
  EXPECT_LT(worst_plastic, 1e-3)
      << "SDIRK3 tangent disagrees with the numerical derivative on fully "
         "plastic steps (worst " << worst_plastic << ")";
  EXPECT_LT(worst_transition, 0.1)
      << "the yield-crossing step is inexact but should stay bounded";
}

/// Equivalent plastic strain after a uniaxial path, for one tableau and one
/// hardening modulus. No tangent checker -- only the return map is of
/// interest here.
T rk_alpha(const std::string& tableau, T hardening) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.05});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "j2");
  p.insert<T>("K", hardening);
  ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "j2");
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<T>("K", T{166.67});
  p.insert<T>("G", T{76.92});
  p.insert<T>("sigma_0", T{50.0});
  p.insert<std::string>("tableau", tableau);
  ctx.create<numsim::materials::j2_rk_plasticity<policy>>(p);
  ctx.finalize();

  for (int i = 0; i < 20; ++i) {
    ctx.update();
    ctx.commit();
  }
  return ctx.get<T>("j2", "equivalent_plastic_strain");
}

/// An explicit stage must solve the same scalar equation as an implicit one.
///
/// For linear hardening the yield residual is linear in dlambda,
/// r = F_trial - (3G + H')*dlambda, so the exact root is F_trial/(3G + H') and
/// forward Euler, implicit Euler and the monolithic return map must all land on
/// it -- IDENTICALLY, not merely close. That makes this an exact test rather
/// than a tolerance test.
///
/// The explicit stage used to pass the raw G into jacobian(), which takes
/// G_eff = 3G, and so solved F/(G + H'). Hardening masks it: at H' = 1000
/// against 3G = 231 the two denominators differ by 14% and the incremental
/// process recovers most of that, which is why the error looked like 0.5%. The
/// low-hardening case is where it shows.
TEST(J2RKStage, ExplicitAndImplicitStagesSolveTheSameEquation) {
  for (const T hardening : {T{1000.0}, T{100.0}, T{10.0}}) {
    const T explicit_alpha = rk_alpha("forward_euler", hardening);
    const T implicit_alpha = rk_alpha("implicit_euler", hardening);
    ASSERT_GT(implicit_alpha, T{1e-6}) << "the path must yield";
    EXPECT_NEAR(explicit_alpha, implicit_alpha, T{1e-12} * implicit_alpha)
        << "forward vs implicit Euler disagree at H' = " << hardening << ": "
        << explicit_alpha << " vs " << implicit_alpha;
  }
}

TEST_F(RKPlasticityTest, PlasticStrainAccumulates) {
  setup_with_tableau("implicit_euler");
  T prev_alpha = 0;
  for (int i = 0; i < 20; ++i) {
    ctx.update();
    auto alpha = ctx.get<T>("j2", "equivalent_plastic_strain");
    EXPECT_GE(alpha, prev_alpha);
    prev_alpha = alpha;
    ctx.commit();
  }
  EXPECT_GT(prev_alpha, 0.0) << "Should accumulate plastic strain";
}

} // namespace

namespace {
/// A fully implicit tableau must be refused, not silently mis-integrated.
///
/// The stage loop sums only a(i,j) for j < i plus the diagonal, so coupling
/// above the diagonal is dropped. gauss_legendre_4 has a(0,1) != 0; before the
/// guard it produced an equivalent plastic strain of -8.8 -- NEGATIVE -- and a
/// yield residual of +10880, with no error. It became reachable from a deck
/// when the tableau turned into a named parameter.
TEST(J2RKScheme, RefusesAFullyImplicitTableau) {
  using policy = numsim::materials::material_policy_default;
  numsim::materials::material_context<policy> ctx;
  policy::ParameterHandler p;
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.01});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);
  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "m");
  p.insert<T>("K", T{1000.0});
  ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "m");
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<T>("K", T{166.67});
  p.insert<T>("G", T{76.92});
  p.insert<T>("sigma_0", T{50.0});
  p.insert<std::string>("tableau", std::string("gauss_legendre_4"));
  EXPECT_THROW(ctx.create<numsim::materials::j2_rk_plasticity<policy>>(p),
               std::invalid_argument);
}

/// The DIRK and explicit schemes it CAN integrate must still be accepted, so
/// the guard cannot pass by refusing everything.
TEST(J2RKScheme, AcceptsEveryDirkAndExplicitScheme) {
  using policy = numsim::materials::material_policy_default;
  for (const char* scheme : {"forward_euler", "explicit_midpoint", "rk4",
                             "implicit_euler", "implicit_midpoint",
                             "crank_nicolson", "sdirk3"}) {
    numsim::materials::material_context<policy> ctx;
    policy::ParameterHandler p;
    p.insert<std::string>("name", "stepper");
    p.insert<T>("increment", T{0.01});
    p.insert<std::vector<std::size_t>>("indices", {0, 0});
    ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);
    p.clear();
    p.insert<std::string>("name", "hardening");
    p.insert<std::string>("source", "m");
    p.insert<T>("K", T{1000.0});
    ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "m");
    p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<T>("K", T{166.67});
    p.insert<T>("G", T{76.92});
    p.insert<T>("sigma_0", T{50.0});
    p.insert<std::string>("tableau", std::string(scheme));
    EXPECT_NO_THROW(ctx.create<numsim::materials::j2_rk_plasticity<policy>>(p))
        << scheme;
  }
}

/// The elastic tangent must agree with the closed forms the same materials use
/// as shortcuts, at every dimension -- not only at Dim = 3.
///
/// do_smooth_return computes C:N as 2G*dev(N) + K*tr(N)*I instead of a rank-4
/// contraction, and drucker_prager_yield_function::apex_tangent returns
/// K*H'/(K*eta*beta + H')*I⊗I. Both assume C = K*I⊗I + 2G*IIdev. All three
/// plasticity materials built 3K*IIvol instead, which is the same thing ONLY
/// at Dim = 3, since IIvol = I⊗I/Dim. At Dim = 2 the old form was off by 30%
/// against the shortcut in the same class.
///
/// material_policy_2d is not registered in the factory and no graph
/// instantiates it, so this was latent -- but it sat in exactly the code the
/// isotropy argument in docs/plasticity.md rests on.
template <std::size_t Dim>
void expect_tangent_matches_shortcut(const char* label) {
  using tensor2 = tmech::tensor<T, Dim, 2>;
  constexpr T K{166.67}, G{76.92};
  const auto C = numsim::materials::plasticity_detail::
      make_isotropic_tangent<T, Dim>(K, G);
  const auto I = tmech::eye<T, Dim, 2>();

  tensor2 x;
  x.fill(T{0});
  x(0, 0) = T{0.03};
  x(1, 1) = T{-0.01};
  x(0, 1) = T{0.02};
  x(1, 0) = T{0.02};
  if constexpr (Dim == 3) x(2, 2) = T{0.005};

  const tensor2 shortcut{T{2} * G * tmech::dev(x) + K * tmech::trace(x) * I};
  const tensor2 contracted{tmech::dcontract(C, x)};
  const tensor2 diff{contracted - shortcut};
  const T err = std::sqrt(tmech::dcontract(diff, diff));
  const T scale = std::sqrt(tmech::dcontract(shortcut, shortcut));
  EXPECT_LT(err, T{1e-12} * scale)
      << label << ": C:x disagrees with 2G dev(x) + K tr(x) I by " << err
      << " against a scale of " << scale;
}

TEST(ElasticTangent, MatchesTheShortcutIn3D) {
  expect_tangent_matches_shortcut<3>("Dim=3");
}

TEST(ElasticTangent, MatchesTheShortcutIn2D) {
  expect_tangent_matches_shortcut<2>("Dim=2");
}

}  // namespace
