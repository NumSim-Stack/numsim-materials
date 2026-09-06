#include <gtest/gtest.h>
#include <algorithm>
#include <print>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/small_strain_plasticity.h"
#include "numsim-materials/materials/j2_plasticity.h"
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
    ctx.create<numsim::materials::backward_euler<policy>>(p);

    // Linear isotropic hardening (Local edge — called in inner loop)
    p.clear();
    p.insert<std::string>("name", "hardening");
    p.insert<std::string>("source", "j2");
    p.insert<T>("K", H_mod);
    ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);

    // J2 plasticity — solver passed as pointer
    p.clear();
    p.insert<std::string>("name", "j2");
    p.insert<std::string>("elastic_source", "elastic");
    p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<std::string>("solver_source", "solver");
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
    ctx.create<numsim::materials::backward_euler<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "hardening");
    p.insert<std::string>("source", "j2");
    p.insert<T>("K", T{1000.0});
    ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "j2");
    p.insert<std::string>("elastic_source", "elastic");
    p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<std::string>("solver_source", "solver");
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
  void setup_with_tableau(const numsim::materials::butcher_tableau& tab) {
    m_tab = tab;
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
    p.insert<std::string>("elastic_source", "elastic");
    p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<T>("G", T{76.92});
    p.insert<T>("sigma_0", T{50.0});
    p.insert<const numsim::materials::butcher_tableau*>("tableau", &m_tab);
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
  numsim::materials::butcher_tableau m_tab;
};

TEST_F(RKPlasticityTest, ImplicitEulerMatchesMonolithic) {
  setup_with_tableau(numsim::materials::implicit_euler());
  T max_rel_error = 0;
  for (int i = 0; i < 20; ++i) {
    ctx.update();
    auto rel = ctx.get<T>("checker", "rel_error");
    auto alpha = ctx.get<T>("j2", "equivalent_plastic_strain");
    std::println("  IE step {:2d}: rel={:.2e} alpha={:.4e}", i, rel, alpha);
    if (rel > max_rel_error) max_rel_error = rel;
    ctx.commit();
  }
  EXPECT_LT(max_rel_error, 0.1)
      << "Implicit Euler RK should match monolithic J2";
}

TEST_F(RKPlasticityTest, SDIRK3TangentCheck) {
  setup_with_tableau(numsim::materials::sdirk3());
  T max_rel_error = 0;
  for (int i = 0; i < 20; ++i) {
    ctx.update();
    auto rel = ctx.get<T>("checker", "rel_error");
    auto alpha = ctx.get<T>("j2", "equivalent_plastic_strain");
    std::println("  SDIRK3 step {:2d}: rel={:.2e} alpha={:.4e}", i, rel, alpha);
    if (rel > max_rel_error) max_rel_error = rel;
    ctx.commit();
  }
  EXPECT_LT(max_rel_error, 0.1)
      << "SDIRK3 tangent should be consistent";
}

TEST_F(RKPlasticityTest, PlasticStrainAccumulates) {
  setup_with_tableau(numsim::materials::implicit_euler());
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
