#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/exponential_isotropic_hardening.h"
#include "numsim-materials/materials/j2_plasticity.h"
#include "numsim-materials/solvers/local_newton.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/driver/material_point_driver.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using driver = nm::material_point_driver<policy>;
using tensor2 = driver::tensor2;
using control = nm::load_control<T>;

constexpr T K{166.67};
constexpr T G{76.92};
constexpr T E{9 * K * G / (3 * K + G)};
constexpr T nu{(3 * K - 2 * G) / (2 * (3 * K + G))};
constexpr T sigma_0{0.5}; // yields at about 2.5e-3
constexpr T H{1000.0};

/// `stuck`: a constant_scalar "stuck::value" of 1, a residual that never falls.
void build_elastic(ctx_type& ctx, bool stuck = false) {
  param_type p;
  if (stuck) {
    p.insert<std::string>("name", "stuck");
    p.insert<T>("value", 1);
    ctx.create<nm::constant_scalar<policy>>(p);
    p.clear();
  }
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

param_type parameters_for(std::string const &stress_source) {
  param_type p;
  p.insert<std::string>("strain_source", "strain");
  p.insert<std::string>("stress_source", stress_source);
  return p;
}

tensor2 axial(T value) {
  tensor2 v;
  v(0, 0) = value;
  return v;
}

TEST(MaterialPointDriver, StrainControlNeedsNoIteration) {
  ctx_type ctx;
  build_elastic(ctx);
  driver d(ctx, parameters_for("elastic"));
  auto const res{d.step(axial(1e-3), tensor2{}, control::strain(), 0, 1)};
  EXPECT_TRUE(res.converged);
  EXPECT_EQ(res.residuals.size(), 1u);
  EXPECT_NEAR(res.stress(0, 0), (K + 4 * G / 3) * 1e-3, 1e-12);
  EXPECT_NEAR(res.stress(1, 1), (K - 2 * G / 3) * 1e-3, 1e-12);
}

TEST(MaterialPointDriver, UniaxialStressOfLinearElasticityConvergesInOneCorrection) {
  ctx_type ctx;
  build_elastic(ctx);
  driver d(ctx, parameters_for("elastic"));
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
  driver d(ctx, parameters_for("j2"));
  driver::step_result res;
  for (int n{1}; n <= 20; ++n) {
    res = d.step(axial(n * 5e-4), tensor2{}, control::uniaxial_stress(), 0, 1);
    ASSERT_TRUE(res.converged) << n;
    d.accept();
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
  driver d(ctx, parameters_for("j2"));
  auto const res{d.step(axial(5e-3), tensor2{}, control::uniaxial_stress(), 0, 1)};
  ASSERT_TRUE(res.converged);
  ASSERT_GE(res.residuals.size(), 3u);
  for (std::size_t k{2}; k + 1 < res.residuals.size(); ++k)
    EXPECT_LT(res.residuals[k], 10 * res.residuals[k - 1] * res.residuals[k - 1] / res.residuals[0]) << k;
}

TEST(MaterialPointDriver, ElasticUnloadingAfterYieldHasSlopeE) {
  ctx_type ctx;
  build_j2(ctx);
  driver d(ctx, parameters_for("j2"));
  auto const loaded{d.step(axial(5e-3), tensor2{}, control::uniaxial_stress(), 0, 1)};
  d.accept();
  auto const unloaded{d.step(axial(4e-3), tensor2{}, control::uniaxial_stress(), 1, 1)};
  EXPECT_NEAR(loaded.stress(0, 0) - unloaded.stress(0, 0), E * 1e-3, 1e-8);
}

TEST(MaterialPointDriver, ReportsTheTangentErrorAgainstFiniteDifferences) {
  ctx_type ctx;
  build_j2(ctx);
  driver d(ctx, parameters_for("j2"));
  auto const res{d.step(axial(5e-3), tensor2{}, control::uniaxial_stress(), 0, 1)};
  EXPECT_LT(d.tangent_error(res), 1e-5);
}

// Shear stress control: gamma_12 prescribed, everything else stress free.
TEST(MaterialPointDriver, ShearSlotUnderStressControlGivesPureShear) {
  ctx_type ctx;
  build_elastic(ctx);
  driver d(ctx, parameters_for("elastic"));
  tensor2 eps;
  eps(0, 1) = eps(1, 0) = 1e-3;
  auto const res{d.step(eps, tensor2{}, control::mixed({"11", "22", "33", "13", "23"}), 0, 1)};
  ASSERT_TRUE(res.converged);
  EXPECT_NEAR(res.stress(0, 1), 2 * G * 1e-3, 1e-10);
  EXPECT_NEAR(tmech::norm(tensor2{res.strain - eps}), 0, 1e-14);
}

// The check perturbs the trial and evaluates it again: accepting afterwards is the same as without the check.
TEST(MaterialPointDriver, TheTangentCheckLeavesTheTrialAsItWas) {
  ctx_type checked_ctx, plain_ctx;
  build_j2(checked_ctx, true);
  build_j2(plain_ctx, true);
  driver checked(checked_ctx, parameters_for("j2")), plain(plain_ctx, parameters_for("j2"));
  for (auto *d : {&checked, &plain}) {
    auto const res{d->step(axial(4e-3), tensor2{}, control::uniaxial_stress(), 0, 1)};
    if (d == &checked) {
      EXPECT_LT(d->tangent_error(res), 1e-5);
    }
    d->accept();
  }
  auto const a{checked.step(axial(8e-3), tensor2{}, control::uniaxial_stress(), 1, 1)};
  auto const b{plain.step(axial(8e-3), tensor2{}, control::uniaxial_stress(), 1, 1)};
  EXPECT_EQ(a.stress, b.stress);
  EXPECT_EQ(checked_ctx.get<tensor2>("j2", "plastic_strain"), plain_ctx.get<tensor2>("j2", "plastic_strain"));
}

// The accepted strain is the old value of the strain source: the next step starts there.
TEST(MaterialPointDriver, AcceptMakesTheStepTheStartOfTheNext) {
  ctx_type ctx;
  build_j2(ctx);
  driver d(ctx, parameters_for("j2"));
  auto const res{d.step(axial(5e-3), tensor2{}, control::uniaxial_stress(), 0, 1)};
  EXPECT_EQ(d.accepted_strain(), tensor2{}) << "not accepted yet";
  d.accept();
  EXPECT_EQ(d.accepted_strain(), res.strain);
}

// A rejected step leaves the history as it was accepted: the run continues as if it had not been tried.
TEST(MaterialPointDriver, RejectReturnsToTheAcceptedState) {
  ctx_type reference_ctx, rejecting_ctx;
  build_j2(reference_ctx);
  build_j2(rejecting_ctx);
  driver reference(reference_ctx, parameters_for("j2")), rejecting(rejecting_ctx, parameters_for("j2"));
  for (auto *d : {&reference, &rejecting}) {
    (void)d->step(axial(4e-3), tensor2{}, control::uniaxial_stress(), 0, 1);
    d->accept();
  }
  (void)rejecting.step(axial(4e-2), tensor2{}, control::uniaxial_stress(), 1, 1);
  rejecting.reject();
  auto const a{rejecting.step(axial(6e-3), tensor2{}, control::uniaxial_stress(), 1, 1)};
  auto const b{reference.step(axial(6e-3), tensor2{}, control::uniaxial_stress(), 1, 1)};
  EXPECT_EQ(a.stress, b.stress);
  EXPECT_EQ(a.strain, b.strain);
}

// Central differences: a coarser step gives a larger error on a nonlinear path.
TEST(MaterialPointDriver, FiniteDifferenceStepIsAParameter) {
  ctx_type ctx;
  build_j2(ctx, true);
  auto coarse_parameters{parameters_for("j2")};
  coarse_parameters.insert<T>("fd_step", 1e-3);
  driver fine(ctx, parameters_for("j2")), coarse(ctx, coarse_parameters);
  auto const res{fine.step(axial(5e-3), tensor2{}, control::uniaxial_stress(), 0, 1)};
  EXPECT_GT(coarse.tangent_error(res), 10 * fine.tangent_error(res));
}

// Out of iterations: the result is the last evaluated trial, the one accept() commits.
TEST(MaterialPointDriver, UnconvergedStepReturnsTheStrainItEvaluated) {
  ctx_type ctx;
  build_elastic(ctx);
  auto parameters{parameters_for("elastic")};
  parameters.insert<int>("max_iter", 0);
  driver d(ctx, parameters);
  auto const res{d.step(axial(1e-3), tensor2{}, control::uniaxial_stress(), 0, 1)};
  ASSERT_FALSE(res.converged);
  EXPECT_EQ(res.residuals.size(), 1u);
  EXPECT_EQ(res.stress, d.point().output<tensor2>("elastic", "stress"));
  d.accept();
  EXPECT_EQ(d.accepted_strain(), res.strain);
}

TEST(MaterialPointDriver, InvalidSettingsAreFatal) {
  ctx_type ctx;
  build_elastic(ctx);
  auto with = [](auto set) {
    auto p{parameters_for("elastic")};
    set(p);
    return p;
  };
  EXPECT_THROW(static_cast<void>(driver{ctx, with([](auto &p) { p.template insert<int>("max_iter", -1); })}),
               u::fatal_error);
  EXPECT_THROW(static_cast<void>(driver{ctx, with([](auto &p) { p.template insert<T>("fd_step", 0); })}),
               u::fatal_error);
  EXPECT_THROW(static_cast<void>(driver{ctx, with([](auto &p) { p.template insert<T>("tolerance", -1); })}),
               u::fatal_error);
}

TEST(MaterialPointDriver, ParametersAreChecked) {
  ctx_type ctx;
  build_elastic(ctx);
  param_type missing;
  missing.insert<std::string>("stress_source", "elastic");
  EXPECT_ANY_THROW(static_cast<void>(driver{ctx, missing})) << "strain_source is required";
  param_type wrong;
  wrong.insert<std::string>("strain_source", "elastic");
  wrong.insert<std::string>("stress_source", "elastic");
  EXPECT_THROW(static_cast<void>(driver{ctx, wrong}), u::fatal_error) << "not a strain source";
}

TEST(MaterialPointDriver, PassesTheResidualSettingsToThePoint) {
  ctx_type ctx;
  build_elastic(ctx, true);
  auto p{parameters_for("elastic")};
  p.insert<std::string>("residual_source", "stuck::value");
  p.insert<int>("max_passes", 3);
  driver stuck{ctx, p};
  EXPECT_THROW(static_cast<void>(stuck.step(axial(1e-3), tensor2{}, control::strain(), 0, 1)), u::cutback_error);
  p.insert<T>("residual_tolerance", 2);
  driver loose{ctx, p};
  EXPECT_TRUE(loose.step(axial(1e-3), tensor2{}, control::strain(), 0, 1).converged);
}

} // namespace
