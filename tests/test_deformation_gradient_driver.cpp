#include <gtest/gtest.h>
#include <cmath>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/neo_hooke.h"
#include "numsim-materials/materials/saint_venant_kirchhoff.h"
#include "numsim-materials/umat/deformation_gradient_driver.h"
#include "numsim-materials/umat/external_state_source.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using driver = u::deformation_gradient_driver<policy>;
using tensor2 = driver::tensor2;
using control = u::deformation_control<T>;

constexpr T K{166.67};
constexpr T G{76.92};
constexpr T E{9 * K * G / (3 * K + G)};
constexpr T nu{(3 * K - 2 * G) / (2 * (3 * K + G))};

template <typename Model>
void build(ctx_type& ctx) {
  param_type p;
  p.insert<std::string>("name", "F");
  ctx.create<nm::external_deformation_gradient_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "model");
  p.insert<std::string>("deformation_gradient_source", "F");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  ctx.create<Model>(p);
  ctx.finalize();
}

param_type parameters() {
  param_type p;
  p.insert<std::string>("deformation_gradient_source", "F");
  p.insert<std::string>("stress_source", "model");
  return p;
}

tensor2 stretch(T lambda) {
  tensor2 F{tmech::eye<T, 3, 2>()};
  F(0, 0) = lambda;
  return F;
}

TEST(DeformationControl, UniaxialStressFreesEverythingButEleven) {
  auto const I{tmech::eye<T, 3, 2>()};
  tensor2 const free{tmech::dcontract(control::uniaxial_stress().stress_projector(), I)};
  EXPECT_DOUBLE_EQ(free(0, 0), 0);
  EXPECT_DOUBLE_EQ(free(1, 1), 1);
  tensor2 g;
  g(1, 0) = 1; // the 21 component is its own slot
  EXPECT_DOUBLE_EQ(tensor2{tmech::dcontract(control::mixed({"21"}).stress_projector(), g)}(1, 0), 1);
  EXPECT_DOUBLE_EQ(tensor2{tmech::dcontract(control::mixed({"12"}).stress_projector(), g)}(1, 0), 0);
}

TEST(DeformationGradientDriver, PrescribedFNeedsNoIteration) {
  ctx_type ctx;
  build<nm::neo_hooke<policy>>(ctx);
  driver d(ctx, parameters());
  auto const res{d.step(stretch(1.1), tensor2{}, control::deformation(), 0, 1)};
  EXPECT_TRUE(res.converged);
  EXPECT_EQ(res.residuals.size(), 1u);
  EXPECT_GT(res.stress(0, 0), 0);
  EXPECT_DOUBLE_EQ(res.F(0, 0), 1.1);
}

TEST(DeformationGradientDriver, SaintVenantKirchhoffUniaxialStressIsExact) {
  ctx_type ctx;
  build<nm::saint_venant_kirchhoff<policy>>(ctx);
  driver d(ctx, parameters());
  T const lambda{1.05};
  auto const res{d.step(stretch(lambda), tensor2{}, control::uniaxial_stress(), 0, 1)};
  ASSERT_TRUE(res.converged);
  T const E11{(lambda * lambda - 1) / 2};
  T const E22{-nu * E11}; // S22 = 0 with S linear in E
  EXPECT_NEAR(res.F(1, 1), std::sqrt(1 + 2 * E22), 1e-12);
  EXPECT_NEAR(res.F(2, 2), std::sqrt(1 + 2 * E22), 1e-12);
  EXPECT_NEAR(res.F(0, 1), 0, 1e-14);
  // S = F^-1 P is diagonal with S11 = E * E11
  tensor2 const S{tmech::inv(res.F) * res.stress};
  EXPECT_NEAR(S(0, 0), E * E11, 1e-9);
  EXPECT_NEAR(S(1, 1), 0, 1e-9);
  EXPECT_LT(res.residuals.size(), 6u);
}

TEST(DeformationGradientDriver, NeoHookeUniaxialStressAndTangent) {
  ctx_type ctx;
  build<nm::neo_hooke<policy>>(ctx);
  driver d(ctx, parameters());
  auto const res{d.step(stretch(1.2), tensor2{}, control::uniaxial_stress(), 0, 1)};
  ASSERT_TRUE(res.converged);
  for (std::size_t a{1}; a < 9; ++a) { // every component but 11
    auto const [i, j]{nm::full_indices[a]};
    EXPECT_NEAR(res.stress(i, j), 0, 1e-8 * res.stress(0, 0)) << i << j;
  }
  EXPECT_LT(res.F(1, 1), 1);
  EXPECT_NEAR(res.F(1, 1), res.F(2, 2), 1e-12);
  EXPECT_LT(d.tangent_error(res), 1e-5);
  // Small stretch: P11 ~ E (lambda - 1)
  driver d2(ctx, parameters());
  auto const small{d2.step(stretch(1.0001), tensor2{}, control::uniaxial_stress(), 0, 1)};
  EXPECT_NEAR(small.stress(0, 0), E * 1e-4, 1e-2 * E * 1e-4);
}

// Simple shear at finite strain: a normal Cauchy stress sigma_11 = G gamma^2 appears (J = 1).
TEST(DeformationGradientDriver, SimpleShearHasANormalStress) {
  ctx_type ctx;
  build<nm::neo_hooke<policy>>(ctx);
  driver d(ctx, parameters());
  tensor2 F{tmech::eye<T, 3, 2>()};
  F(0, 1) = 0.3;
  auto const res{d.step(F, tensor2{}, control::deformation(), 0, 1)};
  EXPECT_DOUBLE_EQ(res.F(0, 1), 0.3);
  EXPECT_DOUBLE_EQ(res.F(1, 0), 0);
  tensor2 const sigma{res.stress * tmech::trans(res.F) / tmech::det(res.F)};
  EXPECT_NEAR(sigma(0, 0), G * 0.09, 1e-9);
  EXPECT_NEAR(sigma(1, 1), 0, 1e-9);
  EXPECT_NEAR(sigma(0, 1), G * 0.3, 1e-9);
  EXPECT_NEAR(sigma(0, 1), sigma(1, 0), 1e-12);
}

// The accepted F is the old value of the F source; a rejected trial leaves it.
TEST(DeformationGradientDriver, AcceptAndRejectMoveTheStartOfTheStep) {
  ctx_type ctx;
  build<nm::neo_hooke<policy>>(ctx);
  driver d(ctx, parameters());
  auto const first{d.step(stretch(1.1), tensor2{}, control::uniaxial_stress(), 0, 1)};
  EXPECT_EQ(d.accepted_F(), stretch(1.0));
  d.accept();
  EXPECT_EQ(d.accepted_F(), first.F);
  (void)d.step(stretch(3.0), tensor2{}, control::uniaxial_stress(), 1, 1);
  d.reject();
  EXPECT_EQ(d.accepted_F(), first.F);
  auto const second{d.step(stretch(1.2), tensor2{}, control::uniaxial_stress(), 1, 1)};
  EXPECT_LT(d.tangent_error(second), 1e-5);
}

// Back to F11 = 1 under uniaxial stress: P = 0 exactly, so |P| alone is no scale for the residual.
TEST(DeformationGradientDriver, UnloadingToZeroStressConverges) {
  ctx_type ctx;
  build<nm::neo_hooke<policy>>(ctx);
  driver d(ctx, parameters());
  ASSERT_TRUE(d.step(stretch(1.1), tensor2{}, control::uniaxial_stress(), 0, 1).converged);
  d.accept();
  auto const res{d.step(stretch(1.0), tensor2{}, control::uniaxial_stress(), 1, 1)};
  EXPECT_TRUE(res.converged);
  EXPECT_NEAR(tmech::norm(tensor2{res.F - stretch(1.0)}), 0, 1e-10);
}

TEST(DeformationGradientDriver, UnconvergedStepReturnsTheFItEvaluated) {
  ctx_type ctx;
  build<nm::neo_hooke<policy>>(ctx);
  auto p{parameters()};
  p.insert<int>("max_iter", 0);
  driver d(ctx, p);
  auto const res{d.step(stretch(1.1), tensor2{}, control::uniaxial_stress(), 0, 1)};
  ASSERT_FALSE(res.converged);
  d.accept();
  EXPECT_EQ(d.accepted_F(), res.F);
}

TEST(DeformationGradientDriver, InvalidSettingsAreFatal) {
  ctx_type ctx;
  build<nm::neo_hooke<policy>>(ctx);
  auto p{parameters()};
  p.insert<int>("max_iter", -1);
  EXPECT_THROW(static_cast<void>(driver{ctx, p}), u::fatal_error);
}

} // namespace
