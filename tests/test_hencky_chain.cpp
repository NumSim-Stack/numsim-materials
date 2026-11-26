#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <string>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/partial_check.h"
#include "numsim-materials/driver/deformation_gradient_driver.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/umat/json_model.h"
#include "numsim-materials/umat/material_point_evaluator.h"
#include "test_support.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using driver = nm::deformation_gradient_driver<policy>;
using control = nm::deformation_control<T>;
using numsim::materials::test::stretch;
using tensor4 = tmech::tensor<T, 3, 4>;

/// The driver on "F" -> "hencky" (the push-forward at the end of the chain).
policy::ParameterHandler point() {
  policy::ParameterHandler p;
  p.insert<std::string>("deformation_gradient_source", "F");
  p.insert<std::string>("stress_source", "hencky");
  return p;
}

constexpr T K{166.67}, G{76.92}, alpha{1e-4}, T0{20};
constexpr T E{9 * K * G / (3 * K + G)}, nu{(3 * K - 2 * G) / (2 * (3 * K + G))};

/// The Hencky chain as a JSON deck: F -> log_strain -> elastic_strain -> stress -> kirchhoff_to_first_piola.
/// `fixed` takes linear_elasticity with K and G, otherwise a linear_stress on a produced isotropic_tangent.
std::string deck(bool eigenstrain, bool fixed = false) {
  std::string const stress{fixed ? R"({"type": "linear_elasticity", "name": "tau", "strain_producer_name": "elastic_strain", "K": 166.67, "G": 76.92})"
                                 : R"({"type": "linear_stress", "name": "tau", "tangent_source": "stiffness", "strain_source": "elastic_strain"})"};
  // a history source adds statev slots the evaluator tests do not provide: only with an eigenstrain
  std::string const thermal{eigenstrain ? R"({"type": "external_scalar_source", "name": "temperature"},
    {"type": "constant_scalar", "name": "alpha", "value": 1e-4},
    {"type": "thermal_eigenstrain", "name": "thermal", "alpha_source": "alpha::value", "reference_temperature": 20.0},)"
                                        : ""};
  return R"({"materials": [
    {"type": "external_deformation_gradient_source", "name": "F"},
    {"type": "deformation_gradient_kinematics", "name": "kinematics", "deformation_gradient_source": "F"},
    {"type": "constant_scalar", "name": "K", "value": 166.67},
    {"type": "constant_scalar", "name": "G", "value": 76.92},
    {"type": "isotropic_tangent", "name": "stiffness", "K_source": "K", "G_source": "G"},
)" + thermal + R"(
    {"type": "log_strain", "name": "log_strain", "deformation_gradient_source": "F"},
    {"type": "elastic_strain", "name": "elastic_strain", "strain_source": "log_strain",
     "eigenstrain_sources": [)" + std::string{eigenstrain ? R"("thermal")" : ""} + R"(]},
    )" + stress + R"(,
    {"type": "kirchhoff_to_first_piola", "name": "hencky", "stress_source": "tau", "tangent_source": ")" +
         std::string{fixed ? "tau" : "stiffness"} + R"(", "strain_source": "log_strain", "kinematics_source": "kinematics"}]})";
}

std::unique_ptr<ctx_type> build(bool with_eigenstrain = false, bool fixed = false) {
  auto owner{std::make_unique<ctx_type>()};
  nm::umat::make_json_builder<policy>(deck(with_eigenstrain, fixed))(*owner, {});
  return owner;
}

TEST(Hencky, DilatationIsExact) {
  auto ctx{build()};
  u::material_point_evaluator<policy> ev(*ctx, {.strain_source = {}, .stress_source = "hencky", .deformation_gradient_source = "F"});
  T const l{1.2};
  ev.evaluate_deformation_gradient(nullptr, tmech::eye<T, 3, 2>(), stretch(l, l, l), 0, 1);
  EXPECT_NEAR(ev.stress()(0, 0), 3 * K * std::log(l) / l, 1e-10); // P = tau F^-T, tau = 3 K ln(l) I
  EXPECT_NEAR(ev.stress()(0, 1), 0, 1e-14);
  EXPECT_NEAR(ctx->get<tensor2>("log_strain", "strain")(1, 1), std::log(l), 1e-14);
}

TEST(Hencky, SmallStrainLimitIsLinearElasticity) {
  auto ctx{build()};
  u::material_point_evaluator<policy> ev(*ctx, {.strain_source = {}, .stress_source = "hencky", .deformation_gradient_source = "F"});
  tensor2 F{tmech::eye<T, 3, 2>()};
  F(0, 0) = 1 + 1e-6;
  F(0, 1) = 2e-6;
  ev.evaluate_deformation_gradient(nullptr, tmech::eye<T, 3, 2>(), F, 0, 1);
  tensor2 const eps{tmech::sym(tensor2{F - tmech::eye<T, 3, 2>()})};
  tensor2 const sigma{tmech::dcontract(nm::isotropic_stiffness<T, 3>(K, G), eps)};
  EXPECT_NEAR(tmech::norm(tensor2{ev.stress() - sigma}) / tmech::norm(sigma), 0, 1e-5);
}

TEST(Hencky, UniaxialStressHasLogarithmicPoissonContractionAndTangent) {
  auto ctx{build()};
  driver d(*ctx, point());
  for (T const l : {1.1, 1.5, 0.7}) { // never accepted: every step starts from F = I
    auto const res{d.step(stretch(l), tensor2{}, control::uniaxial_stress(), 0, 1)};
    ASSERT_TRUE(res.converged) << l;
    EXPECT_NEAR(res.F(1, 1), std::pow(l, -nu), 1e-10) << l;       // ln F22 = -nu ln F11 exactly
    EXPECT_NEAR(res.F(2, 2), std::pow(l, -nu), 1e-10) << l;
    EXPECT_NEAR(res.stress(0, 0), E * std::log(l) / l, 1e-8) << l; // tau11 = E ln l, P11 = tau11 / F11
    EXPECT_LT(d.tangent_error(res), 1e-6) << l;
  }
  // non-symmetric F: simple shear, the Kirchhoff stress P F^T is symmetric
  tensor2 F{tmech::eye<T, 3, 2>()};
  F(0, 1) = 0.4;
  auto const res{d.step(F, tensor2{}, control::deformation(), 0, 1)};
  tensor2 const tau{res.stress * tmech::trans(res.F)};
  EXPECT_NEAR(tmech::norm(tensor2{tau - tmech::trans(tau)}), 0, 1e-10);
  EXPECT_LT(d.tangent_error(res), 1e-6);
}

// the symptom of tmech#37: the tangent at repeated and nearly repeated stretches
TEST(Hencky, TangentIsConsistentAtRepeatedStretches) {
  auto ctx{build()};
  driver d(*ctx, point());
  for (auto const &F : {stretch(1.15, 1.15, 1.0), stretch(1.2, 1.2, 1.2), stretch(1.15, 1.15 + 1e-9, 1.0), stretch(1.204, 0.9458, 0.9458)}) {
    auto const res{d.step(F, tensor2{}, control::deformation(), 0, 1)};
    EXPECT_LT(d.tangent_error(res), 1e-6) << F(0, 0) << " " << F(1, 1) << " " << F(2, 2);
  }
}

TEST(Hencky, IsObjective) {
  auto ctx{build()};
  u::material_point_evaluator<policy> ev(*ctx, {.strain_source = {}, .stress_source = "hencky", .deformation_gradient_source = "F"});
  tmech::tensor<T, 3, 1> axis;
  axis(0) = 1;
  axis(1) = 2;
  axis(2) = 2;
  tensor2 const R{nm::rotation(tmech::tensor<T, 3, 1>{axis / 3}, T{0.7})};
  tensor2 U{stretch(1.3, 0.9, 1.1)};
  U(0, 1) = U(1, 0) = 0.1;
  ev.evaluate_deformation_gradient(nullptr, tmech::eye<T, 3, 2>(), U, 0, 1);
  tensor2 const P_U{ev.stress()};
  ev.evaluate_deformation_gradient(nullptr, tmech::eye<T, 3, 2>(), tensor2{R * U}, 0, 1);
  EXPECT_NEAR(tmech::norm(tensor2{ev.stress() - R * P_U}) / tmech::norm(P_U), 0, 1e-12); // P(R U) = R P(U)
  ev.evaluate_deformation_gradient(nullptr, tmech::eye<T, 3, 2>(), R, 0, 1);
  EXPECT_NEAR(tmech::norm(ev.stress()), 0, 1e-10 * K);                                  // a rotation is stress free
}

TEST(Hencky, EigenstrainShiftsTheLogStrain) {
  auto ctx{build(true)};
  dynamic_cast<nm::external_deformation_gradient_source<policy> *>(ctx->find("F"))->bind(tmech::eye<T, 3, 2>(), tmech::eye<T, 3, 2>());
  dynamic_cast<nm::external_scalar_source<policy> *>(ctx->find("temperature"))->bind(T0, T0 + 100);
  ctx->update();
  EXPECT_NEAR(ctx->get<tensor2>("hencky", "stress")(0, 0), -3 * K * alpha * 100, 1e-10); // constrained heating
  T const l{std::exp(alpha * 100)};                                                    // free expansion: ln l = alpha dT
  dynamic_cast<nm::external_deformation_gradient_source<policy> *>(ctx->find("F"))->bind(tmech::eye<T, 3, 2>(), stretch(l, l, l));
  ctx->update();
  EXPECT_NEAR(tmech::norm(ctx->get<tensor2>("hencky", "stress")), 0, 1e-10);
}

TEST(Hencky, TangentMatchesFiniteDifferencesAtAGeneralFWithAnEigenstrain) {
  tmech::tensor<T, 3, 1> axis;
  axis(0) = 1, axis(1) = 2, axis(2) = 2;
  tensor2 U{stretch(1.25, 0.9, 1.1)};
  U(0, 1) = U(1, 0) = 0.15;
  tensor2 F{nm::rotation(tmech::tensor<T, 3, 1>{axis / 3}, T{0.7}) * U};
  F(0, 1) += 0.2;
  for (bool const fixed : {false, true}) {
    auto ctx{build(true, fixed)};
    dynamic_cast<nm::external_scalar_source<policy> *>(ctx->find("temperature"))->bind(T0, T0 + 40);
    driver d(*ctx, point());
    auto const res{d.step(F, tensor2{}, control::deformation(), 0, 1)};
    ASSERT_TRUE(res.converged);
    EXPECT_LT(d.tangent_error(res), 1e-6) << fixed;
  }
}

/// A general F: stretch, rotation and shear.
tensor2 general_F() {
  tmech::tensor<T, 3, 1> axis;
  axis(0) = 1, axis(1) = 2, axis(2) = 2;
  tensor2 U{stretch(1.25, 0.9, 1.1)};
  U(0, 1) = U(1, 0) = 0.15;
  tensor2 F{nm::rotation(tmech::tensor<T, 3, 1>{axis / 3}, T{0.7}) * U};
  F(0, 1) += 0.2;
  return F;
}

void bind(ctx_type &ctx, tensor2 const &F, T temperature) {
  dynamic_cast<nm::external_deformation_gradient_source<policy> *>(ctx.find("F"))->bind(tmech::eye<T, 3, 2>(), F);
  dynamic_cast<nm::external_scalar_source<policy> *>(ctx.find("temperature"))->bind(T0, temperature);
  ctx.update();
}

TEST(Hencky, TheDerivativeOfPByFIsThePublishedTangent) {
  for (bool const fixed : {false, true}) {
    auto ctx{build(true, fixed)};
    auto const &dP{ctx->derivative<tensor4>("hencky", "stress", "F", "deformation_gradient")};
    bind(*ctx, general_F(), T0 + 40);
    EXPECT_LT(nm::test::relative(dP, ctx->get<tensor4>("hencky", "tangent")), 1e-12) << fixed;
  }
}

TEST(Hencky, TheDerivativeOfPByTheTemperatureMatchesCentralDifferences) {
  for (bool const fixed : {false, true}) {
    auto ctx{build(true, fixed)};
    auto const &dP{ctx->derivative<tensor2>("hencky", "stress", "temperature", "state")};
    constexpr T h{1e-3};
    bind(*ctx, general_F(), T0 + 40 + h);
    tensor2 const plus{ctx->get<tensor2>("hencky", "stress")};
    bind(*ctx, general_F(), T0 + 40 - h);
    tensor2 const minus{ctx->get<tensor2>("hencky", "stress")};
    bind(*ctx, general_F(), T0 + 40);
    tensor2 const fd{(plus - minus) / (2 * h)};
    EXPECT_LT(nm::test::relative(dP, fd), 1e-8) << fixed;
    ASSERT_GT(tmech::norm(fd), 1e-3) << "a temperature dependence to see";
  }
}

TEST(Hencky, TheDeclaredPartialsOfTheChainMatchFiniteDifferences) {
  for (bool const fixed : {false, true}) {
    auto ctx{build(true, fixed)};
    bind(*ctx, general_F(), T0 + 40);
    for (auto const *material : ctx->materials())
      for (auto const &mismatch : nm::check_partials(*ctx, material->name()))
        ADD_FAILURE() << fixed << ": " << mismatch;
  }
}

TEST(Hencky, LinearElasticityOnTheSameConstantsGivesTheSameStress) {
  auto produced{build()};
  auto fixed{build(false, true)};
  tensor2 F{stretch(1.2, 0.95, 1.05)};
  F(0, 1) = 0.3;
  for (auto *ctx : {produced.get(), fixed.get()}) {
    dynamic_cast<nm::external_deformation_gradient_source<policy> *>(ctx->find("F"))->bind(tmech::eye<T, 3, 2>(), F);
    ctx->update();
  }
  EXPECT_LE(tmech::norm(tensor2{produced->get<tensor2>("hencky", "stress") - fixed->get<tensor2>("hencky", "stress")}), 1e-12);
}

// the documented failure mode: a push-forward on a wrong tangent_source keeps P and breaks dP/dF
TEST(Hencky, AMismatchedTangentSourceShowsInTheTangentError) {
  std::string mismatched{deck(false)};
  auto const replace = [&](std::string const &from, std::string const &to) {
    auto const at{mismatched.find(from)};
    ASSERT_NE(at, std::string::npos) << from;
    mismatched.replace(at, from.size(), to);
  };
  replace(R"("tangent_source": "stiffness", "strain_source": "log_strain")", R"("tangent_source": "other", "strain_source": "log_strain")");
  replace(R"({"type": "log_strain")", R"({"type": "constant_scalar", "name": "G2", "value": 153.84},
    {"type": "isotropic_tangent", "name": "other", "K_source": "K", "G_source": "G2"},
    {"type": "log_strain")");
  auto right{build()};
  ctx_type wrong;
  nm::umat::make_json_builder<policy>(mismatched)(wrong, {});
  tensor2 F{stretch(1.2, 0.95, 1.05)};
  F(0, 1) = 0.3;
  driver d_right(*right, point()), d_wrong(wrong, point());
  auto const r_right{d_right.step(F, tensor2{}, control::deformation(), 0, 1)};
  auto const r_wrong{d_wrong.step(F, tensor2{}, control::deformation(), 0, 1)};
  EXPECT_LE(tmech::norm(tensor2{r_right.stress - r_wrong.stress}), 1e-12);
  EXPECT_LT(d_right.tangent_error(r_right), 1e-6);
  EXPECT_GT(d_wrong.tangent_error(r_wrong), 0.1);
}

// the chain as a deck: the template a user starts from
TEST(Hencky, TheJsonDeckBuildsAndEvaluates) {
  auto ctx{build(true)};
  EXPECT_NE(ctx->find("hencky"), nullptr);
  dynamic_cast<nm::external_deformation_gradient_source<policy> *>(ctx->find("F"))->bind(tmech::eye<T, 3, 2>(), tmech::eye<T, 3, 2>());
  dynamic_cast<nm::external_scalar_source<policy> *>(ctx->find("temperature"))->bind(T0, T0 + 100);
  ctx->update();
  EXPECT_NEAR(ctx->get<tensor2>("hencky", "stress")(1, 1), -3 * K * alpha * 100, 1e-10);
}

} // namespace
