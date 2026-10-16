#include <gtest/gtest.h>
#include <cmath>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/exponential_isotropic_hardening.h"
#include "numsim-materials/materials/j2_plasticity.h"
#include "numsim-materials/solvers/local_newton.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/material_point_evaluator.h"
#include "numsim-materials/umat/material_point_driver.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using evaluator = u::material_point_evaluator<policy>;
using driver = u::material_point_driver<policy>;
using tensor2 = driver::tensor2;
using control = u::load_control<T>;

constexpr T K{166.67};
constexpr T G{76.92};
constexpr T E{9 * K * G / (3 * K + G)};
constexpr T nu{(3 * K - 2 * G) / (2 * (3 * K + G))};
constexpr T sigma_0{0.5}; // yields at about 2.5e-3
constexpr T H{1000.0};

void build_elastic(ctx_type& ctx) {
  param_type p;
  p.insert<std::string>("name", "strain");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "strain");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  ctx.create<nm::linear_elasticity<policy>>(p);
  ctx.finalize();
}

void build_j2(ctx_type& ctx, bool exponential = false) {
  param_type p;
  p.insert<std::string>("name", "strain");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "solver");
  ctx.create<nm::local_newton<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "j2");
  if (exponential) {
    p.insert<T>("K_inf", 1.0);
    p.insert<T>("delta", 200.0);
    ctx.create<nm::exponential_isotropic_hardening<policy>>(p);
  } else {
    p.insert<T>("K", H);
    ctx.create<nm::linear_isotropic_hardening<policy>>(p);
  }
  p.clear();
  p.insert<std::string>("name", "j2");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  p.insert<T>("sigma_0", sigma_0);
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "strain");
  p.insert<std::string>("solver_source", "solver");
  ctx.create<nm::j2_plasticity<policy>>(p);
  ctx.finalize();
}

evaluator::config config_for(std::string stress_source) {
  evaluator::config cfg;
  cfg.strain_source = "strain";
  cfg.stress_source = std::move(stress_source);
  return cfg;
}

tensor2 axial(T value) {
  tensor2 v;
  v(0, 0) = value;
  return v;
}

TEST(MaterialPointDriver, StrainControlNeedsNoIteration) {
  ctx_type ctx;
  build_elastic(ctx);
  evaluator ev(ctx, config_for("elastic"));
  driver d(ev);
  auto const res{d.step(axial(1e-3), tensor2{}, control::strain(), 0, 1)};
  EXPECT_TRUE(res.converged);
  EXPECT_EQ(res.residuals.size(), 1u);
  EXPECT_NEAR(res.stress(0, 0), (K + 4 * G / 3) * 1e-3, 1e-12);
  EXPECT_NEAR(res.stress(1, 1), (K - 2 * G / 3) * 1e-3, 1e-12);
}

TEST(MaterialPointDriver, UniaxialStressOfLinearElasticityConvergesInOneCorrection) {
  ctx_type ctx;
  build_elastic(ctx);
  evaluator ev(ctx, config_for("elastic"));
  driver d(ev);
  auto const res{d.step(axial(1e-3), tensor2{}, control::uniaxial_stress(), 0, 1)};
  ASSERT_TRUE(res.converged);
  EXPECT_EQ(res.residuals.size(), 2u);
  EXPECT_NEAR(res.stress(0, 0), E * 1e-3, 1e-10);
  EXPECT_NEAR(res.strain(1, 1), -nu * 1e-3, 1e-14);
  EXPECT_NEAR(res.strain(2, 2), -nu * 1e-3, 1e-14);
  EXPECT_NEAR(tmech::norm(tensor2{res.stress - axial(res.stress(0, 0))}), 0, 1e-10);
}

TEST(MaterialPointDriver, UniaxialStressOfJ2FollowsTheBilinearCurve) {
  ctx_type ctx;
  build_j2(ctx);
  evaluator ev(ctx, config_for("j2"));
  driver d(ev);
  driver::step_result res;
  for (int n{1}; n <= 20; ++n) {
    res = d.step(axial(n * 5e-4), tensor2{}, control::uniaxial_stress(), 0, 1);
    ASSERT_TRUE(res.converged) << n;
  }
  T const eps{20 * 5e-4};
  EXPECT_NEAR(res.stress(0, 0), (eps + sigma_0 / H) / (1 / E + 1 / H), 1e-8);
  EXPECT_NEAR(tmech::norm(tensor2{res.stress - axial(res.stress(0, 0))}), 0, 1e-8);
  EXPECT_LE(res.residuals.size(), 6u);
}

// Linear hardening is linear once yielded; exponential hardening is not.
TEST(MaterialPointDriver, J2UniaxialStressConvergesQuadratically) {
  ctx_type ctx;
  build_j2(ctx, true);
  evaluator ev(ctx, config_for("j2"));
  driver d(ev);
  auto const res{d.step(axial(5e-3), tensor2{}, control::uniaxial_stress(), 0, 1)};
  ASSERT_TRUE(res.converged);
  ASSERT_GE(res.residuals.size(), 3u);
  for (std::size_t k{2}; k + 1 < res.residuals.size(); ++k)
    EXPECT_LT(res.residuals[k], 10 * res.residuals[k - 1] * res.residuals[k - 1] / res.residuals[0]) << k;
}

TEST(MaterialPointDriver, ElasticUnloadingAfterYieldHasSlopeE) {
  ctx_type ctx;
  build_j2(ctx);
  evaluator ev(ctx, config_for("j2"));
  driver d(ev);
  auto const loaded{d.step(axial(5e-3), tensor2{}, control::uniaxial_stress(), 0, 1)};
  auto const unloaded{d.step(axial(4e-3), tensor2{}, control::uniaxial_stress(), 1, 1)};
  EXPECT_NEAR(loaded.stress(0, 0) - unloaded.stress(0, 0), E * 1e-3, 1e-8);
}

TEST(MaterialPointDriver, ReportsTheTangentErrorAgainstFiniteDifferences) {
  ctx_type ctx;
  build_j2(ctx);
  evaluator ev(ctx, config_for("j2"));
  driver d(ev);
  auto const res{d.step(axial(5e-3), tensor2{}, control::uniaxial_stress(), 0, 1)};
  EXPECT_LT(d.tangent_error(res), 1e-5);
}

// Shear stress control: gamma_12 prescribed, everything else stress free.
TEST(MaterialPointDriver, ShearSlotUnderStressControlGivesPureShear) {
  ctx_type ctx;
  build_elastic(ctx);
  evaluator ev(ctx, config_for("elastic"));
  driver d(ev);
  tensor2 eps;
  eps(0, 1) = eps(1, 0) = 1e-3;
  auto const res{d.step(eps, tensor2{}, control::mixed({"11", "22", "33", "13", "23"}), 0, 1)};
  ASSERT_TRUE(res.converged);
  EXPECT_NEAR(res.stress(0, 1), 2 * G * 1e-3, 1e-10);
  EXPECT_NEAR(tmech::norm(tensor2{res.strain - eps}), 0, 1e-14);
}

// The check uses the start state stored in the result, not the driver's latest state.
TEST(MaterialPointDriver, TangentErrorOfAnEarlierStepIsUnaffectedByLaterSteps) {
  ctx_type ctx;
  build_j2(ctx, true);
  evaluator ev(ctx, config_for("j2"));
  driver d(ev);
  auto const first{d.step(axial(4e-3), tensor2{}, control::uniaxial_stress(), 0, 1)};
  auto const before{d.tangent_error(first)};
  (void)d.step(axial(8e-3), tensor2{}, control::uniaxial_stress(), 1, 1);
  EXPECT_DOUBLE_EQ(d.tangent_error(first), before);
  EXPECT_LT(before, 1e-5);
}

// Central differences: a coarser step gives a larger error on a nonlinear path.
TEST(MaterialPointDriver, FiniteDifferenceStepIsAnOption) {
  ctx_type ctx;
  build_j2(ctx, true);
  evaluator ev(ctx, config_for("j2"));
  driver fine(ev), coarse(ev, {.fd_step = 1e-3});
  auto const res{fine.step(axial(5e-3), tensor2{}, control::uniaxial_stress(), 0, 1)};
  EXPECT_GT(coarse.tangent_error(res), 10 * fine.tangent_error(res));
}

} // namespace
