#include <gtest/gtest.h>
#include <cmath>
#include <span>
#include <vector>
#include "numsim-materials/umat/tensor_conversion.h"
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/small_strain_plasticity.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/material_point_evaluator.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using evaluator = u::material_point_evaluator<policy>;

constexpr T K = 166.67;
constexpr T G = 76.92;
constexpr T sigma_0 = 50.0;
constexpr T H_mod = 1000.0;

/// The J2 chain, with a host-driven strain source named "stepper".
void build_j2(ctx_type& ctx) {
  param_type p;

  p.clear();
  p.insert<std::string>("name", "stepper");
  ctx.create<nm::external_strain_source<policy>>(p);

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

evaluator::config j2_config() {
  evaluator::config cfg;
  cfg.strain_source = "stepper";
  cfg.stress_source = "j2";
  return cfg;
}

// ---------------------------------------------------------------------------
// Equivalence with the self-driven reference
// ---------------------------------------------------------------------------

/// The evaluator, driven through raw pointers, must reproduce the trajectory
/// the stepper-driven context produces. This is the end-to-end check that the
/// pointer boundary changes nothing.
TEST(MaterialPointEvaluator, ReproducesStepperDrivenTrajectory) {
  constexpr T d_eps = 0.05;

  ctx_type ref;
  {
    param_type p;
    p.insert<std::string>("name", "stepper");
    p.insert<T>("increment", d_eps);
    p.insert<std::vector<std::size_t>>("indices", {0, 0});
    ref.create<nm::tensor_component_stepper<2, policy>>(p);
  }
  {
    param_type p;
    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "stepper");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    ref.create<nm::linear_elasticity<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "solver");
    ref.create<nm::backward_euler<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "hardening");
    p.insert<std::string>("source", "j2");
    p.insert<T>("K", H_mod);
    ref.create<nm::linear_isotropic_hardening<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "j2");
    p.insert<std::string>("elastic_source", "elastic");
    p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<std::string>("solver_source", "solver");
    p.insert<T>("G", G);
    p.insert<T>("sigma_0", sigma_0);
    ref.create<nm::j2_plasticity<policy>>(p);
  }
  ref.finalize();

  ctx_type ctx;
  build_j2(ctx);
  evaluator eval(ctx, j2_config());

  std::vector<T> statev(eval.nstatv(), 0.0);
  T stran[6] = {0, 0, 0, 0, 0, 0};

  for (int step = 0; step < 30; ++step) {
    ref.update();
    const auto& ref_strain = ref.get<tensor2>("stepper", "strain");

    // Take the increment from the reference so the input is bit-identical.
    T dstran[6] = {0, 0, 0, 0, 0, 0};
    dstran[0] = ref_strain(0, 0) - stran[0];

    T stress[6], ddsdde[36];
    eval.evaluate({.stran = stran,
                   .dstran = dstran,
                   .stress = stress,
                   .ddsdde = ddsdde,
                   .statev = statev});

    const auto& ref_sig = ref.get<tensor2>("j2", "stress");
    EXPECT_DOUBLE_EQ(stress[0], ref_sig(0, 0)) << "step " << step;
    EXPECT_DOUBLE_EQ(stress[1], ref_sig(1, 1)) << "step " << step;
    EXPECT_DOUBLE_EQ(stress[2], ref_sig(2, 2)) << "step " << step;

    EXPECT_DOUBLE_EQ(statev[0], ref.get<T>("j2", "equivalent_plastic_strain"))
        << "step " << step;

    ref.commit();
    for (std::size_t i = 0; i < 6; ++i) stran[i] += dstran[i];
  }
}

// ---------------------------------------------------------------------------
// Statelessness
// ---------------------------------------------------------------------------

/// A UMAT is re-called on unconverged global iterates. Calling twice with the
/// same STATEV must give the same answer — if any state leaked across the call
/// boundary, the second result would drift.
TEST(MaterialPointEvaluator, RepeatedCallOnSameStateIsIdempotent) {
  ctx_type ctx;
  build_j2(ctx);
  evaluator eval(ctx, j2_config());

  std::vector<T> statev(eval.nstatv(), 0.0);
  T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.01, -0.0025, 0.0, 0.005, 0.0, 0.0};

  // Walk into the plastic regime so the path actually depends on history.
  for (int step = 0; step < 40; ++step) {
    T stress[6], ddsdde[36];
    eval.evaluate({.stran = stran,
                   .dstran = dstran,
                   .stress = stress,
                   .ddsdde = ddsdde,
                   .statev = statev});
    for (std::size_t i = 0; i < 6; ++i) stran[i] += dstran[i];
  }
  ASSERT_GT(statev[0], 1e-8) << "test must reach the plastic regime";

  const std::vector<T> statev_n = statev;

  T s1[6], c1[36], s2[6], c2[36];
  std::vector<T> sv1 = statev_n, sv2 = statev_n;

  eval.evaluate({.stran = stran, .dstran = dstran, .stress = s1, .ddsdde = c1,
                 .statev = sv1});
  eval.evaluate({.stran = stran, .dstran = dstran, .stress = s2, .ddsdde = c2,
                 .statev = sv2});

  for (std::size_t i = 0; i < 6; ++i) EXPECT_DOUBLE_EQ(s2[i], s1[i]) << i;
  for (std::size_t i = 0; i < 36; ++i) EXPECT_DOUBLE_EQ(c2[i], c1[i]) << i;
  for (std::size_t i = 0; i < sv1.size(); ++i)
    EXPECT_DOUBLE_EQ(sv2[i], sv1[i]) << i;
}

// ---------------------------------------------------------------------------
// DDSDDE through the pointer boundary
// ---------------------------------------------------------------------------

/// Finite-difference the exported DDSDDE *through the raw-pointer boundary*.
///
/// This catches an engineering-shear factor of 2 and a slot permutation —
/// neither of which a tensor-level tangent check can see, because they live
/// outside the tensor algebra.
///
/// It does NOT catch a row/column-major transpose, despite the symmetry of the
/// FD loop making it look like it would: J2's consistent tangent is
/// major-symmetric, so a transpose is invisible here. That contract is covered
/// only by the deliberately major-ASYMMETRIC fixture in test_umat_conversion,
/// which makes it a single point of coverage worth keeping.
///
/// Each perturbed call starts from a fresh copy of the t_n STATEV, which is
/// only correct because the evaluator is stateless.
void check_tangent_by_finite_difference(evaluator& eval,
                                        const std::vector<T>& statev_n,
                                        std::span<const T> stran,
                                        std::span<const T> dstran, T tol) {
  T stress0[6], ddsdde[36];
  {
    std::vector<T> sv = statev_n;
    eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress0,
                   .ddsdde = ddsdde, .statev = sv});
  }

  const T h = 1e-8;
  for (std::size_t j = 0; j < 6; ++j) {
    T dp[6], dm[6];
    for (std::size_t i = 0; i < 6; ++i) { dp[i] = dstran[i]; dm[i] = dstran[i]; }
    dp[j] += h;
    dm[j] -= h;

    T sp[6], sm[6], dummy[36];
    std::vector<T> svp = statev_n, svm = statev_n;
    eval.evaluate({.stran = stran, .dstran = dp, .stress = sp, .ddsdde = dummy,
                   .statev = svp});
    eval.evaluate({.stran = stran, .dstran = dm, .stress = sm, .ddsdde = dummy,
                   .statev = svm});

    for (std::size_t i = 0; i < 6; ++i) {
      const T fd = (sp[i] - sm[i]) / (2 * h);
      // DDSDDE is column-major: entry (i,j) lives at i + j*NTENS.
      EXPECT_NEAR(ddsdde[i + j * 6], fd, tol)
          << "DDSDDE(" << i << "," << j << ")";
    }
  }
}

TEST(MaterialPointEvaluator, ElasticTangentMatchesFiniteDifference) {
  ctx_type ctx;
  build_j2(ctx);
  evaluator eval(ctx, j2_config());

  const std::vector<T> statev_n(eval.nstatv(), 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {1e-4, -5e-5, 0.0, 8e-5, 3e-5, 0.0};

  check_tangent_by_finite_difference(eval, statev_n, stran, dstran, 1e-5);
}

TEST(MaterialPointEvaluator, PlasticConsistentTangentMatchesFiniteDifference) {
  ctx_type ctx;
  build_j2(ctx);
  evaluator eval(ctx, j2_config());

  std::vector<T> statev(eval.nstatv(), 0.0);
  T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.01, -0.0025, 0.0, 0.005, 0.0, 0.0};

  for (int step = 0; step < 40; ++step) {
    T stress[6], ddsdde[36];
    eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                   .ddsdde = ddsdde, .statev = statev});
    for (std::size_t i = 0; i < 6; ++i) stran[i] += dstran[i];
  }
  ASSERT_GT(statev[0], 1e-8) << "test must reach the plastic regime";

  check_tangent_by_finite_difference(eval, statev, stran, dstran, 5e-4);
}

// ---------------------------------------------------------------------------
// Dimension adapter
// ---------------------------------------------------------------------------

/// A strain path with no out-of-plane shear is representable in both the 3D and
/// the plane-strain interface, so the two must agree exactly. This is what
/// makes the widen/narrow adapter lossless rather than merely plausible.
TEST(MaterialPointEvaluator, PlaneStrainAgreesWithSolid3dOnAPlanarPath) {
  ctx_type ctx3;
  build_j2(ctx3);
  evaluator eval3(ctx3, j2_config());

  ctx_type ctx2;
  build_j2(ctx2);
  evaluator eval2(ctx2, j2_config());

  std::vector<T> sv3(eval3.nstatv(), 0.0);
  std::vector<T> sv2(eval2.nstatv(), 0.0);

  T stran3[6] = {0, 0, 0, 0, 0, 0};
  T stran2[4] = {0, 0, 0, 0};
  // eps_33 = 0 (plane strain), no 13/23 shear.
  const T d3[6] = {0.004, -0.001, 0.0, 0.002, 0.0, 0.0};
  const T d2[4] = {0.004, -0.001, 0.0, 0.002};

  for (int step = 0; step < 20; ++step) {
    T s3[6], c3[36], s2[4], c2[16];
    eval3.evaluate({.stran = stran3, .dstran = d3, .stress = s3, .ddsdde = c3,
                    .statev = sv3});
    eval2.evaluate({.stran = stran2, .dstran = d2, .stress = s2, .ddsdde = c2,
                    .statev = sv2,
                    .ec = u::element_case::plane_strain});

    for (std::size_t i = 0; i < 4; ++i)
      EXPECT_DOUBLE_EQ(s2[i], s3[i]) << "step " << step << " slot " << i;

    // The 4x4 must be the leading sub-block of the 6x6, both column-major.
    for (std::size_t a = 0; a < 4; ++a)
      for (std::size_t b = 0; b < 4; ++b)
        EXPECT_DOUBLE_EQ(c2[a + b * 4], c3[a + b * 6])
            << "step " << step << " (" << a << "," << b << ")";

    for (std::size_t i = 0; i < sv2.size(); ++i)
      EXPECT_DOUBLE_EQ(sv2[i], sv3[i]) << "step " << step << " statev " << i;

    for (std::size_t i = 0; i < 6; ++i) stran3[i] += d3[i];
    for (std::size_t i = 0; i < 4; ++i) stran2[i] += d2[i];
  }
}

/// Axisymmetric supplies a genuinely nonzero hoop strain, unlike plane strain.
TEST(MaterialPointEvaluator, AxisymmetricCarriesHoopStrain) {
  ctx_type ctx3;
  build_j2(ctx3);
  evaluator eval3(ctx3, j2_config());

  ctx_type ctxA;
  build_j2(ctxA);
  evaluator evalA(ctxA, j2_config());

  std::vector<T> sv3(eval3.nstatv(), 0.0);
  std::vector<T> svA(evalA.nstatv(), 0.0);

  const T stran3[6] = {0, 0, 0, 0, 0, 0};
  const T stranA[4] = {0, 0, 0, 0};
  const T d3[6] = {0.003, -0.001, 0.0015, 0.002, 0.0, 0.0};
  const T dA[4] = {0.003, -0.001, 0.0015, 0.002};

  T s3[6], c3[36], sA[4], cA[16];
  eval3.evaluate({.stran = stran3, .dstran = d3, .stress = s3, .ddsdde = c3,
                  .statev = sv3});
  evalA.evaluate({.stran = stranA, .dstran = dA, .stress = sA, .ddsdde = cA,
                  .statev = svA,
                  .ec = u::element_case::axisymmetric});

  for (std::size_t i = 0; i < 4; ++i) EXPECT_DOUBLE_EQ(sA[i], s3[i]) << i;

  // The meaningful check is not that the hoop stress is large, but that the
  // supplied eps_33 is actually CARRIED: zeroing it must change the answer.
  // Without this, an implementation that silently dropped slot 2 (as plane
  // strain may, since there eps_33 is zero anyway) would still pass.
  ctx_type ctxZ;
  build_j2(ctxZ);
  evaluator evalZ(ctxZ, j2_config());
  std::vector<T> svZ(evalZ.nstatv(), 0.0);
  const T dZ[4] = {0.003, -0.001, 0.0, 0.002};
  T sZ[4], cZ[16];
  evalZ.evaluate({.stran = stranA, .dstran = dZ, .stress = sZ, .ddsdde = cZ,
                  .statev = svZ,
                  .ec = u::element_case::axisymmetric});
  // Quantitatively: this step is elastic, so removing eps_33 must change the
  // hoop stress by exactly C_3333 * eps_33 = (K + 4G/3) * eps_33. Asserting the
  // value rather than "some difference" also pins down that slot 2 is the 33
  // component and carries no shear factor.
  EXPECT_NEAR(sA[2] - sZ[2], (K + 4.0 * G / 3.0) * 0.0015, 1e-9);
}

// ---------------------------------------------------------------------------
// Configuration errors
// ---------------------------------------------------------------------------

TEST(MaterialPointEvaluator, RejectsPlaneStress) {
  ctx_type ctx;
  build_j2(ctx);
  evaluator eval(ctx, j2_config());

  std::vector<T> statev(eval.nstatv(), 0.0);
  const T stran[3] = {0, 0, 0};
  const T dstran[3] = {0.001, 0.0, 0.0};
  T stress[3], ddsdde[9];

  EXPECT_THROW(eval.evaluate({.stran = stran, .dstran = dstran,
                              .stress = stress, .ddsdde = ddsdde,
                              .statev = statev,
                              .ec = u::element_case::plane_stress}),
               u::fatal_error);
}

TEST(MaterialPointEvaluator, RejectsUndersizedStatev) {
  ctx_type ctx;
  build_j2(ctx);
  evaluator eval(ctx, j2_config());

  std::vector<T> statev(eval.nstatv() - 1, 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  T stress[6], ddsdde[36];

  EXPECT_THROW(eval.evaluate({.stran = stran, .dstran = dstran,
                              .stress = stress, .ddsdde = ddsdde,
                              .statev = statev}),
               std::runtime_error);
}

TEST(MaterialPointEvaluator, RejectsNonSourceStrainMaterial) {
  ctx_type ctx;
  build_j2(ctx);
  auto cfg = j2_config();
  cfg.strain_source = "elastic";  // exists, but is not a host-driven source
  EXPECT_THROW(evaluator(ctx, cfg), std::runtime_error);
}

TEST(MaterialPointEvaluator, ReportsStatevLayout) {
  ctx_type ctx;
  build_j2(ctx);
  evaluator eval(ctx, j2_config());
  EXPECT_EQ(eval.nstatv(), 7u);

  const auto desc = eval.describe_statev();
  ASSERT_EQ(desc.size(), 2u);
  EXPECT_EQ(desc[0], "0..0  j2::equivalent_plastic_strain");
  EXPECT_EQ(desc[1], "1..6  j2::plastic_strain  [engineering shear]");
}

// ---------------------------------------------------------------------------
// Co-rotational state update (DROT)
// ---------------------------------------------------------------------------

/// 90 degrees about z, in Abaqus's column-major DROT(3,3) layout.
void rot_z90(T out[9]) {
  const T R[3][3] = {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) out[i + 3 * j] = R[i][j];
}

/// Objectivity: rotating the strain history and handing the UMAT the same
/// rotation must rotate the stress, and nothing else.
///
/// This only holds if STATEV is rotated on the way in. Abaqus rotates STRESS
/// and STRAN itself under NLGEOM=YES but never touches user state, so an
/// implementation that ignored DROT would carry the plastic strain in a stale
/// frame — and would fail here by an amount that grows with the rotation.
TEST(MaterialPointEvaluator, CoRotationalUpdateKeepsTheResponseObjective) {
  T drot[9];
  rot_z90(drot);
  const auto R = u::rotation_from_buffer<T>(drot);

  auto rotate_strain_slots = [&R](const T in[6], T out[6]) {
    const auto t = u::strain_from_buffer<T>(in);
    u::strain_to_buffer<T>(u::rotate(t, R), out);
  };

  ctx_type ctx;
  build_j2(ctx);
  evaluator eval(ctx, j2_config());

  // Build a genuinely anisotropic plastic state at t_n.
  std::vector<T> statev(eval.nstatv(), 0.0);
  T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.01, -0.0025, 0.0, 0.005, 0.0, 0.0};
  for (int step = 0; step < 40; ++step) {
    T stress[6], ddsdde[36];
    eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                   .ddsdde = ddsdde, .statev = statev});
    for (std::size_t i = 0; i < 6; ++i) stran[i] += dstran[i];
  }
  ASSERT_GT(statev[0], 1e-8) << "the state must be plastic to be directional";

  const std::vector<T> statev_n = statev;

  // Unrotated reference.
  T sig_ref[6], c_ref[36];
  std::vector<T> sv_ref = statev_n;
  eval.evaluate({.stran = stran, .dstran = dstran, .stress = sig_ref,
                 .ddsdde = c_ref, .statev = sv_ref});

  // Same state, everything rotated, DROT supplied.
  T stran_r[6], dstran_r[6];
  rotate_strain_slots(stran, stran_r);
  rotate_strain_slots(dstran, dstran_r);

  T sig_rot[6], c_rot[36];
  std::vector<T> sv_rot = statev_n;
  eval.evaluate({.stran = stran_r, .dstran = dstran_r, .stress = sig_rot,
                 .ddsdde = c_rot, .statev = sv_rot,
                 .drot = std::span<const T>(drot, 9)});

  // sigma_rotated must equal R sigma_ref R^T.
  const auto expected = u::rotate(u::stress_from_buffer<T>(sig_ref), R);
  T expected_slots[6];
  u::stress_to_buffer<T>(expected, expected_slots);
  for (std::size_t i = 0; i < 6; ++i)
    EXPECT_NEAR(sig_rot[i], expected_slots[i], 1e-9) << "stress slot " << i;

  // The scalar state is frame-indifferent and must be untouched by rotation.
  EXPECT_NEAR(sv_rot[0], sv_ref[0], 1e-12);
}

/// Without DROT the state must NOT be rotated — otherwise every small-strain
/// analysis would silently acquire a spurious rotation.
TEST(MaterialPointEvaluator, OmittingDrotLeavesStateUnrotated) {
  ctx_type ctx;
  build_j2(ctx);
  evaluator eval(ctx, j2_config());

  std::vector<T> statev(eval.nstatv(), 0.0);
  T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.01, -0.0025, 0.0, 0.005, 0.0, 0.0};
  for (int step = 0; step < 40; ++step) {
    T stress[6], ddsdde[36];
    eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                   .ddsdde = ddsdde, .statev = statev});
    for (std::size_t i = 0; i < 6; ++i) stran[i] += dstran[i];
  }

  const std::vector<T> statev_n = statev;
  T identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  T s_none[6], c_none[36], s_ident[6], c_ident[36];
  std::vector<T> sv_none = statev_n, sv_ident = statev_n;
  eval.evaluate({.stran = stran, .dstran = dstran, .stress = s_none,
                 .ddsdde = c_none, .statev = sv_none});
  eval.evaluate({.stran = stran, .dstran = dstran, .stress = s_ident,
                 .ddsdde = c_ident, .statev = sv_ident,
                 .drot = std::span<const T>(identity, 9)});

  for (std::size_t i = 0; i < 6; ++i) EXPECT_DOUBLE_EQ(s_ident[i], s_none[i]);
  for (std::size_t i = 0; i < sv_none.size(); ++i)
    EXPECT_DOUBLE_EQ(sv_ident[i], sv_none[i]) << i;
}

// ---------------------------------------------------------------------------
// Time source
// ---------------------------------------------------------------------------

/// The time source must receive TIME(2) and TIME(2)+DTIME as the old/new pair,
/// which is what a rate-dependent material reads to form dt — and it must stay
/// out of STATEV, since the host owns it.
TEST(MaterialPointEvaluator, BindsTimeSourceAndExcludesItFromStatev) {
  ctx_type ctx;
  {
    param_type p;
    p.insert<std::string>("name", "stepper");
    ctx.create<nm::external_strain_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "clock");
    ctx.create<nm::external_scalar_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "stepper");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    ctx.create<nm::linear_elasticity<policy>>(p);
    ctx.finalize();
  }

  auto cfg = j2_config();
  cfg.stress_source = "elastic";
  cfg.time_source = "clock";
  evaluator eval(ctx, cfg);

  // "clock::state" is history, but host-owned, so it must not consume a slot.
  EXPECT_EQ(eval.nstatv(), 0u);

  std::vector<T> statev(eval.nstatv(), 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  T stress[6], ddsdde[36];

  eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                 .ddsdde = ddsdde, .statev = statev, .time = 7.5,
                 .dtime = 0.25});

  auto* prop = ctx.find_property("clock", "state");
  ASSERT_NE(prop, nullptr);
  auto* hist =
      dynamic_cast<numsim_core::history_property<T, nm::property_traits>*>(prop);
  ASSERT_NE(hist, nullptr);
  EXPECT_DOUBLE_EQ(hist->old_value(), 7.5);
  EXPECT_DOUBLE_EQ(hist->new_value(), 7.75);
}

// ---------------------------------------------------------------------------
// Energies
// ---------------------------------------------------------------------------

/// SSE and SPD must together account for the total stress work, and SPD must
/// stay at zero while the response is elastic.
TEST(MaterialPointEvaluator, ReportsElasticEnergyAndPlasticDissipation) {
  ctx_type ctx;
  build_j2(ctx);
  auto cfg = j2_config();
  cfg.plastic_strain_property = "j2::plastic_strain";
  evaluator eval(ctx, cfg);
  ASSERT_TRUE(eval.reports_energies());

  std::vector<T> statev(eval.nstatv(), 0.0);
  T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.01, -0.0025, 0.0, 0.005, 0.0, 0.0};
  T stress[6] = {0, 0, 0, 0, 0, 0}, ddsdde[36];
  T sse = 0, spd = 0, scd = 0;

  T work = 0;  // independently accumulated trapezoidal stress work
  bool checked_elastic = false;

  for (int step = 0; step < 40; ++step) {
    T sig_old[6];
    for (std::size_t i = 0; i < 6; ++i) sig_old[i] = stress[i];

    eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                   .ddsdde = ddsdde, .statev = statev, .sse = &sse,
                   .spd = &spd, .scd = &scd});

    const auto s_old = u::stress_from_buffer<T>(sig_old);
    const auto s_new = u::stress_from_buffer<T>(stress);
    const auto de = u::strain_from_buffer<T>(dstran);
    tensor2 s_mid;
    s_mid = 0.5 * (s_old + s_new);
    work += tmech::dcontract(s_mid, de);

    if (!checked_elastic && statev[0] == 0.0) {
      EXPECT_NEAR(spd, 0.0, 1e-12) << "no dissipation while elastic, step "
                                   << step;
      checked_elastic = true;
    }

    EXPECT_NEAR(sse + spd, work, 1e-9) << "energy balance at step " << step;
    for (std::size_t i = 0; i < 6; ++i) stran[i] += dstran[i];
  }

  EXPECT_TRUE(checked_elastic);
  EXPECT_GT(statev[0], 1e-8) << "the path must yield";
  EXPECT_GT(spd, 0.0) << "plastic dissipation must accumulate";
  EXPECT_GT(sse, 0.0);
}

TEST(MaterialPointEvaluator, LeavesEnergiesAloneWhenNotConfigured) {
  ctx_type ctx;
  build_j2(ctx);
  evaluator eval(ctx, j2_config());
  EXPECT_FALSE(eval.reports_energies());

  std::vector<T> statev(eval.nstatv(), 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  T stress[6] = {0, 0, 0, 0, 0, 0}, ddsdde[36];
  T sse = 42.0, spd = 17.0, scd = 3.0;

  eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                 .ddsdde = ddsdde, .statev = statev, .sse = &sse, .spd = &spd,
                 .scd = &scd});

  EXPECT_DOUBLE_EQ(sse, 42.0);
  EXPECT_DOUBLE_EQ(spd, 17.0);
  EXPECT_DOUBLE_EQ(scd, 3.0);
}

// ---------------------------------------------------------------------------
// Buffer-size validation
// ---------------------------------------------------------------------------

/// The array length is dictated by the element family. A mismatch used to be an
/// out-of-bounds read that produced plausible-looking numbers; it is now caught.
TEST(MaterialPointEvaluator, RejectsBuffersSizedForTheWrongElementType) {
  ctx_type ctx;
  build_j2(ctx);
  evaluator eval(ctx, j2_config());

  std::vector<T> statev(eval.nstatv(), 0.0);
  const T short_stran[4] = {0, 0, 0, 0};
  const T short_dstran[4] = {0.001, 0, 0, 0};
  T short_stress[4], ddsdde[36];
  T full_stran[6] = {0, 0, 0, 0, 0, 0}, full_dstran[6] = {0.001, 0, 0, 0, 0, 0};
  T full_stress[6], small_ddsdde[16];

  // 4-component arrays with the default solid3d case.
  EXPECT_THROW(eval.evaluate({.stran = short_stran, .dstran = short_dstran,
                              .stress = short_stress, .ddsdde = ddsdde,
                              .statev = statev}),
               u::fatal_error);

  // Right vectors, wrong tangent size.
  EXPECT_THROW(eval.evaluate({.stran = full_stran, .dstran = full_dstran,
                              .stress = full_stress, .ddsdde = small_ddsdde,
                              .statev = statev}),
               u::fatal_error);

  // A DROT that is not 3x3.
  T bad_rot[4] = {1, 0, 0, 1};
  EXPECT_THROW(eval.evaluate({.stran = full_stran, .dstran = full_dstran,
                              .stress = full_stress, .ddsdde = ddsdde,
                              .statev = statev,
                              .drot = std::span<const T>(bad_rot, 4)}),
               u::fatal_error);
}


/// The energy-balance assertion above is a TAUTOLOGY on its own: sse and spd are
/// updated as `sse += dSSE; spd += dW - dSSE`, so their sum telescopes to the
/// accumulated work for ANY definition of dSSE whatsoever. Defining the elastic
/// strain as the total strain (ignoring plastic strain entirely) passes it.
///
/// These two tests constrain the SPLIT, which is the only thing ALLSE/ALLPD
/// actually depend on.
TEST(MaterialPointEvaluator, ElasticStepStoresAllWorkAsStrainEnergy) {
  ctx_type ctx;
  build_j2(ctx);
  auto cfg = j2_config();
  cfg.plastic_strain_property = "j2::plastic_strain";
  evaluator eval(ctx, cfg);

  std::vector<T> statev(eval.nstatv(), 0.0);
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, -0.0004, 0.0, 0.0008, 0.0, 0.0};
  T stress[6] = {0, 0, 0, 0, 0, 0}, ddsdde[36];
  T sse = 0, spd = 0, scd = 0;

  eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                 .ddsdde = ddsdde, .statev = statev, .sse = &sse, .spd = &spd,
                 .scd = &scd});

  ASSERT_DOUBLE_EQ(statev[0], 0.0) << "this step must be elastic";

  // From rest with no plastic flow, all the work is stored: SSE = 1/2 sigma:eps.
  const auto sig = u::stress_from_buffer<T>(stress);
  const auto eps = u::strain_from_buffer<T>(dstran);
  EXPECT_NEAR(sse, 0.5 * tmech::dcontract(sig, eps), 1e-12);
  EXPECT_NEAR(spd, 0.0, 1e-15);
}

/// Regression: SSE and SPD are both accumulated incrementally, so a host that
/// starts from a pre-stressed state with SSE = 0 (which is what Abaqus does
/// under *INITIAL CONDITIONS, TYPE=STRESS) must not book the pre-existing
/// stored energy as negative dissipation.
///
/// An earlier version set SSE to the absolute 1/2 sigma:(eps - eps_p) and
/// derived the SPD increment from its change; this scenario produced
/// SPD = -1.35 on a step with no plastic flow at all, and the offset persisted
/// in ALLPD for the whole analysis.
TEST(MaterialPointEvaluator, PreStressedStartDoesNotProduceNegativeDissipation) {
  ctx_type ctx;
  build_j2(ctx);
  auto cfg = j2_config();
  cfg.plastic_strain_property = "j2::plastic_strain";
  evaluator eval(ctx, cfg);

  std::vector<T> statev(eval.nstatv(), 0.0);
  // The host reports an existing strain and the matching stress, with the
  // energies still at zero.
  const T stran[6] = {0.1, 0.0, 0.0, 0.0, 0.0, 0.0};
  T stress[6] = {26.92, 11.54, 11.54, 0.0, 0.0, 0.0};
  const T dstran[6] = {0.001, 0.0, 0.0, 0.0, 0.0, 0.0};
  T ddsdde[36];
  T sse = 0, spd = 0, scd = 0;

  eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress,
                 .ddsdde = ddsdde, .statev = statev, .sse = &sse, .spd = &spd,
                 .scd = &scd});

  ASSERT_DOUBLE_EQ(statev[0], 0.0) << "this step must be elastic";
  EXPECT_NEAR(spd, 0.0, 1e-12) << "no plastic flow, so no dissipation";
  EXPECT_GE(spd, -1e-12) << "dissipation must never be negative";
  EXPECT_GT(sse, 0.0) << "an elastic increment stores energy";
}

}  // namespace
