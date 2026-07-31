#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/small_strain_plasticity.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/material_point_evaluator.h"
#include "numsim-materials/umat/plane_stress_evaluator.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using ps_evaluator = u::plane_stress_evaluator<policy>;
using evaluator = u::material_point_evaluator<policy>;

constexpr T K = 166.67;
constexpr T G = 76.92;
constexpr T sigma_0 = 50.0;
constexpr T H_mod = 1000.0;

const T E = 9.0 * K * G / (3.0 * K + G);
const T nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G));

void add_strain_source(ctx_type& ctx) {
  param_type p;
  p.insert<std::string>("name", "stepper");
  ctx.create<nm::external_strain_source<policy>>(p);
}

void build_elastic(ctx_type& ctx) {
  add_strain_source(ctx);
  param_type p;
  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  ctx.create<nm::linear_elasticity<policy>>(p);
  ctx.finalize();
}

void build_j2(ctx_type& ctx) {
  add_strain_source(ctx);
  param_type p;

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  ctx.create<nm::linear_elasticity<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "solver");
  ctx.create<nm::backward_euler<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "j2");
  p.insert<T>("K", H_mod);
  ctx.create<nm::linear_isotropic_hardening<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "j2");
  p.insert<std::string>("elastic_source", "elastic");
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<std::string>("solver_source", "solver");
  p.insert<T>("G", G);
  p.insert<T>("sigma_0", sigma_0);
  ctx.create<nm::j2_plasticity<policy>>(p);

  ctx.finalize();
}

evaluator::config cfg_for(const std::string& stress_source) {
  evaluator::config cfg;
  cfg.strain_source = "stepper";
  cfg.stress_source = stress_source;
  return cfg;
}

// ---------------------------------------------------------------------------
// Elastic: closed-form reference
// ---------------------------------------------------------------------------

TEST(PlaneStressEvaluator, ElasticStressMatchesAnalyticPlaneStress) {
  ctx_type ctx;
  build_elastic(ctx);
  ps_evaluator eval(ctx, cfg_for("elastic"));

  // Pure elasticity has no history, so the only STATEV slot is eps_33.
  ASSERT_EQ(eval.nstatv(), 1u);
  std::vector<T> statev(eval.nstatv(), 0.0);

  const T stran[3] = {0, 0, 0};
  const T dstran[3] = {0.003, -0.001, 0.002};  // e11, e22, gamma12
  T stress[3], ddsdde[9];

  eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                 .ddsdde = ddsdde, .statev = statev});

  const T f = E / (1.0 - nu * nu);
  EXPECT_NEAR(stress[0], f * (dstran[0] + nu * dstran[1]), 1e-9);
  EXPECT_NEAR(stress[1], f * (nu * dstran[0] + dstran[1]), 1e-9);
  EXPECT_NEAR(stress[2], G * dstran[2], 1e-9);

  // eps_33 = -nu/(1-nu) * (e11 + e22) for isotropic plane stress.
  EXPECT_NEAR(statev[0], -nu / (1.0 - nu) * (dstran[0] + dstran[1]), 1e-12);
}

TEST(PlaneStressEvaluator, ElasticTangentMatchesAnalyticPlaneStress) {
  ctx_type ctx;
  build_elastic(ctx);
  ps_evaluator eval(ctx, cfg_for("elastic"));

  std::vector<T> statev(eval.nstatv(), 0.0);
  const T stran[3] = {0, 0, 0};
  const T dstran[3] = {0.003, -0.001, 0.002};
  T stress[3], ddsdde[9];

  eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                 .ddsdde = ddsdde, .statev = statev});

  const T f = E / (1.0 - nu * nu);
  EXPECT_NEAR(ddsdde[0 + 0 * 3], f, 1e-9);
  EXPECT_NEAR(ddsdde[1 + 1 * 3], f, 1e-9);
  EXPECT_NEAR(ddsdde[0 + 1 * 3], nu * f, 1e-9);
  EXPECT_NEAR(ddsdde[1 + 0 * 3], nu * f, 1e-9);
  EXPECT_NEAR(ddsdde[2 + 2 * 3], G, 1e-9);
  EXPECT_NEAR(ddsdde[0 + 2 * 3], 0.0, 1e-9);
  EXPECT_NEAR(ddsdde[2 + 1 * 3], 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// The defining constraint
// ---------------------------------------------------------------------------

/// The point of the whole wrapper: feed the recovered eps_33 back through a
/// full 3D evaluation and sigma_33 must vanish. Checked in the plastic regime,
/// where no closed form exists and the solve is doing real work.
TEST(PlaneStressEvaluator, DrivesOutOfPlaneStressToZeroUnderPlasticity) {
  ctx_type ps_ctx;
  build_j2(ps_ctx);
  ps_evaluator eval(ps_ctx, cfg_for("j2"));

  // A 3D evaluator over an independent context, used only to audit the result.
  ctx_type audit_ctx;
  build_j2(audit_ctx);
  evaluator audit(audit_ctx, cfg_for("j2"));

  std::vector<T> statev(eval.nstatv(), 0.0);
  std::vector<T> audit_statev(audit.nstatv(), 0.0);

  T stran[3] = {0, 0, 0};
  const T dstran[3] = {0.01, -0.002, 0.004};
  T eps33_prev = 0.0;

  bool went_plastic = false;

  for (int step = 0; step < 30; ++step) {
    T stress[3], ddsdde[9];
    eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                   .ddsdde = ddsdde, .statev = statev});

    const T eps33 = statev[eval.nstatv() - 1];

    // Replay the same increment in 3D with the recovered out-of-plane strain.
    const T a_stran[6] = {stran[0], stran[1], eps33_prev, stran[2], 0, 0};
    const T a_dstran[6] = {dstran[0], dstran[1], eps33 - eps33_prev,
                           dstran[2], 0, 0};
    T a_stress[6], a_ddsdde[36];
    audit.evaluate({.stran = a_stran, .dstran = a_dstran, .stress = a_stress,
                    .ddsdde = a_ddsdde, .statev = audit_statev});

    EXPECT_NEAR(a_stress[2], 0.0, 1e-8) << "sigma_33 at step " << step;
    EXPECT_NEAR(a_stress[0], stress[0], 1e-8) << "sigma_11 at step " << step;
    EXPECT_NEAR(a_stress[1], stress[1], 1e-8) << "sigma_22 at step " << step;
    EXPECT_NEAR(a_stress[3], stress[2], 1e-8) << "sigma_12 at step " << step;

    if (audit_statev[0] > 1e-8) went_plastic = true;

    for (std::size_t i = 0; i < 3; ++i) stran[i] += dstran[i];
    eps33_prev = eps33;
  }

  EXPECT_TRUE(went_plastic) << "the path must enter the plastic regime for "
                               "this test to mean anything";
}

// ---------------------------------------------------------------------------
// Condensed tangent
// ---------------------------------------------------------------------------

/// Finite-difference the condensed 3x3 against the CONDENSED response — i.e.
/// re-solving eps_33 for each perturbed in-plane strain. A check against the
/// uncondensed 6x6 would not detect a condensation error at all.
TEST(PlaneStressEvaluator, CondensedTangentMatchesFiniteDifference) {
  ctx_type ctx;
  build_j2(ctx);
  ps_evaluator eval(ctx, cfg_for("j2"));

  std::vector<T> statev(eval.nstatv(), 0.0);
  T stran[3] = {0, 0, 0};
  const T dstran[3] = {0.01, -0.002, 0.004};

  for (int step = 0; step < 40; ++step) {
    T stress[3], ddsdde[9];
    eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                   .ddsdde = ddsdde, .statev = statev});
    for (std::size_t i = 0; i < 3; ++i) stran[i] += dstran[i];
  }
  ASSERT_GT(statev[0], 1e-8) << "test must reach the plastic regime";

  const std::vector<T> statev_n = statev;

  T stress0[3], ddsdde[9];
  {
    std::vector<T> sv = statev_n;
    eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress0,
                   .ddsdde = ddsdde, .statev = sv});
  }

  const T h = 1e-8;
  for (std::size_t j = 0; j < 3; ++j) {
    T dp[3], dm[3];
    for (std::size_t i = 0; i < 3; ++i) { dp[i] = dstran[i]; dm[i] = dstran[i]; }
    dp[j] += h;
    dm[j] -= h;

    T sp[3], sm[3], dummy[9];
    std::vector<T> svp = statev_n, svm = statev_n;
    eval.evaluate({.stran = stran, .dstran = dp, .stress = sp, .ddsdde = dummy,
                   .statev = svp});
    eval.evaluate({.stran = stran, .dstran = dm, .stress = sm, .ddsdde = dummy,
                   .statev = svm});

    for (std::size_t i = 0; i < 3; ++i) {
      const T fd = (sp[i] - sm[i]) / (2 * h);
      EXPECT_NEAR(ddsdde[i + j * 3], fd, 5e-4)
          << "DDSDDE(" << i << "," << j << ")";
    }
  }
}

// ---------------------------------------------------------------------------
// Solve behaviour
// ---------------------------------------------------------------------------

/// A linear material has a linear residual in eps_33, so Newton lands on the
/// answer in a single update: one evaluation to measure, one to confirm.
TEST(PlaneStressEvaluator, LinearMaterialConvergesInOneNewtonUpdate) {
  ctx_type ctx;
  build_elastic(ctx);
  ps_evaluator eval(ctx, cfg_for("elastic"));

  std::vector<T> statev(eval.nstatv(), 0.0);
  T stran[3] = {0, 0, 0};
  const T dstran[3] = {0.001, 0.0005, 0.0};
  T stress[3], ddsdde[9];

  for (int step = 0; step < 3; ++step) {
    eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                   .ddsdde = ddsdde, .statev = statev});
    EXPECT_EQ(eval.last_iterations(), 2) << "step " << step;
    for (std::size_t i = 0; i < 3; ++i) stran[i] += dstran[i];
  }
}

/// The carried eps_33 earns its STATEV slot: under plasticity, warm-starting
/// from the converged out-of-plane strain costs no more graph evaluations than
/// restarting from zero, and generally fewer. Each saved iteration is a full
/// re-evaluation of the material graph at every Gauss point.
TEST(PlaneStressEvaluator, WarmStartCostsNoMoreIterationsThanColdStart) {
  auto run = [](bool warm) {
    ctx_type ctx;
    build_j2(ctx);
    ps_evaluator eval(ctx, cfg_for("j2"));

    std::vector<T> statev(eval.nstatv(), 0.0);
    T stran[3] = {0, 0, 0};
    const T dstran[3] = {0.01, -0.002, 0.004};
    int total = 0;

    for (int step = 0; step < 40; ++step) {
      if (!warm) statev[eval.nstatv() - 1] = 0.0;  // discard the carried guess
      T stress[3], ddsdde[9];
      eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                     .ddsdde = ddsdde, .statev = statev});
      total += eval.last_iterations();
      for (std::size_t i = 0; i < 3; ++i) stran[i] += dstran[i];
    }
    return total;
  };

  const int warm = run(true);
  const int cold = run(false);
  EXPECT_LE(warm, cold) << "warm=" << warm << " cold=" << cold;
}

/// Statelessness must survive the extra outer loop.
TEST(PlaneStressEvaluator, RepeatedCallOnSameStateIsIdempotent) {
  ctx_type ctx;
  build_j2(ctx);
  ps_evaluator eval(ctx, cfg_for("j2"));

  std::vector<T> statev(eval.nstatv(), 0.0);
  T stran[3] = {0, 0, 0};
  const T dstran[3] = {0.01, -0.002, 0.004};

  for (int step = 0; step < 40; ++step) {
    T stress[3], ddsdde[9];
    eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                   .ddsdde = ddsdde, .statev = statev});
    for (std::size_t i = 0; i < 3; ++i) stran[i] += dstran[i];
  }
  ASSERT_GT(statev[0], 1e-8);

  const std::vector<T> statev_n = statev;
  T s1[3], c1[9], s2[3], c2[9];
  std::vector<T> sv1 = statev_n, sv2 = statev_n;

  eval.evaluate({.stran = stran, .dstran = dstran, .stress = s1, .ddsdde = c1,
                 .statev = sv1});
  eval.evaluate({.stran = stran, .dstran = dstran, .stress = s2, .ddsdde = c2,
                 .statev = sv2});

  for (std::size_t i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(s2[i], s1[i]) << i;
  for (std::size_t i = 0; i < 9; ++i) EXPECT_DOUBLE_EQ(c2[i], c1[i]) << i;
  for (std::size_t i = 0; i < sv1.size(); ++i)
    EXPECT_DOUBLE_EQ(sv2[i], sv1[i]) << i;
}

TEST(PlaneStressEvaluator, RequiresTheExtraStatevSlot) {
  ctx_type ctx;
  build_j2(ctx);
  ps_evaluator eval(ctx, cfg_for("j2"));

  // One slot short: enough for the material's history, not for eps_33.
  std::vector<T> statev(eval.nstatv() - 1, 0.0);
  const T stran[3] = {0, 0, 0};
  const T dstran[3] = {0.001, 0.0, 0.0};
  T stress[3], ddsdde[9];

  EXPECT_THROW(eval.evaluate({.stran = stran, .dstran = dstran,
                              .stress = stress, .ddsdde = ddsdde,
                              .statev = statev}),
               std::runtime_error);
}

TEST(PlaneStressEvaluator, ReportsStatevLayoutIncludingOutOfPlaneSlot) {
  ctx_type ctx;
  build_j2(ctx);
  ps_evaluator eval(ctx, cfg_for("j2"));

  EXPECT_EQ(eval.nstatv(), 8u);  // 7 for J2 + 1 for eps_33
  const auto desc = eval.describe_statev();
  ASSERT_EQ(desc.size(), 3u);
  EXPECT_EQ(desc[2], "7..7  <plane_stress>::out_of_plane_strain");
}


/// Energies must also work through the plane-stress wrapper, where they are
/// computed from the CONVERGED canonical state — so the elastic split uses the
/// genuine 3D strain including the solved-for eps_33, not the in-plane part.
TEST(PlaneStressEvaluator, ReportsEnergiesFromTheConvergedState) {
  ctx_type ctx;
  build_j2(ctx);
  auto cfg = cfg_for("j2");
  cfg.plastic_strain_property = "j2::plastic_strain";
  ps_evaluator eval(ctx, cfg);

  std::vector<T> statev(eval.nstatv(), 0.0);
  T stran[3] = {0, 0, 0};
  const T dstran[3] = {0.01, -0.002, 0.004};
  T stress[3] = {0, 0, 0}, ddsdde[9];
  T sse = 0, spd = 0, scd = 0;

  T work = 0;
  for (int step = 0; step < 40; ++step) {
    T sig_old[3];
    for (std::size_t i = 0; i < 3; ++i) sig_old[i] = stress[i];

    eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                   .ddsdde = ddsdde, .statev = statev, .sse = &sse,
                   .spd = &spd, .scd = &scd});

    // In-plane work only: sigma_33 is zero by construction, so the out-of-plane
    // strain does no work and the 2D and 3D work increments agree.
    T w6_old[6], w6_new[6], de6[6];
    u::widen_vector<T>(sig_old, u::element_case::plane_stress, w6_old);
    u::widen_vector<T>(stress, u::element_case::plane_stress, w6_new);
    u::widen_vector<T>(dstran, u::element_case::plane_stress, de6);
    const auto s_old = u::stress_from_buffer<T>(w6_old);
    const auto s_new = u::stress_from_buffer<T>(w6_new);
    const auto de = u::strain_from_buffer<T>(de6);
    tensor2 s_mid;
    s_mid = 0.5 * (s_old + s_new);
    work += tmech::dcontract(s_mid, de);

    EXPECT_NEAR(sse + spd, work, 1e-8) << "energy balance at step " << step;
    for (std::size_t i = 0; i < 3; ++i) stran[i] += dstran[i];
  }

  EXPECT_GT(statev[0], 1e-8) << "the path must yield";
  EXPECT_GT(spd, 0.0);
  EXPECT_GT(sse, 0.0);
}

}  // namespace
