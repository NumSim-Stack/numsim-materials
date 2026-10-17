#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/hencky_elasticity.h"
#include "numsim-materials/materials/thermal_eigenstrain.h"
#include "numsim-materials/umat/deformation_gradient_driver.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/json_model.h"
#include "numsim-materials/umat/material_point_evaluator.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using driver = u::deformation_gradient_driver<policy>;
using control = u::deformation_control<T>;

constexpr T K{166.67}, G{76.92}, alpha{1e-4}, T0{20};
constexpr T E{9 * K * G / (3 * K + G)}, nu{(3 * K - 2 * G) / (2 * (3 * K + G))};

std::unique_ptr<ctx_type> build(bool with_eigenstrain = false) {
  auto owner{std::make_unique<ctx_type>()};
  auto &ctx{*owner};
  param_type p;
  p.insert<std::string>("name", "F");
  ctx.create<nm::external_deformation_gradient_source<policy>>(p);
  if (with_eigenstrain) {
    p.clear();
    p.insert<std::string>("name", "temperature");
    ctx.create<nm::external_scalar_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "thermal");
    p.insert<T>("alpha", alpha);
    p.insert<T>("reference_temperature", T0);
    ctx.create<nm::thermal_eigenstrain<policy>>(p);
  }
  p.clear();
  p.insert<std::string>("name", "hencky");
  p.insert<std::string>("deformation_gradient_source", "F");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  if (with_eigenstrain)
    p.insert<std::string>("eigenstrain_source", "thermal");
  ctx.create<nm::hencky_elasticity<policy>>(p);
  ctx.finalize();
  return owner;
}

tensor2 stretch(T l1, T l2 = 1, T l3 = 1) {
  tensor2 F;
  F(0, 0) = l1;
  F(1, 1) = l2;
  F(2, 2) = l3;
  return F;
}

TEST(Hencky, DilatationIsExact) {
  auto ctx{build()};
  u::material_point_evaluator<policy> ev(*ctx, {.strain_source = {}, .stress_source = "hencky", .deformation_gradient_source = "F"});
  T const l{1.2};
  ev.evaluate_deformation_gradient(nullptr, tmech::eye<T, 3, 2>(), stretch(l, l, l), 0, 1);
  EXPECT_NEAR(ev.stress()(0, 0), 3 * K * std::log(l) / l, 1e-10); // P = tau F^-T, tau = 3 K ln(l) I
  EXPECT_NEAR(ev.stress()(0, 1), 0, 1e-14);
  EXPECT_NEAR(ctx->get<tensor2>("hencky", "log_strain")(1, 1), std::log(l), 1e-14);
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
  u::material_point_evaluator<policy> ev(*ctx, {.strain_source = {}, .stress_source = "hencky", .deformation_gradient_source = "F"});
  driver d(ev);
  for (T const l : {1.1, 1.5, 0.7}) {
    driver d2(ev);
    auto const res{d2.step(stretch(l), tensor2{}, control::uniaxial_stress(), 0, 1)};
    ASSERT_TRUE(res.converged) << l;
    EXPECT_NEAR(res.F(1, 1), std::pow(l, -nu), 1e-10) << l;       // ln F22 = -nu ln F11 exactly
    EXPECT_NEAR(res.F(2, 2), std::pow(l, -nu), 1e-10) << l;
    EXPECT_NEAR(res.stress(0, 0), E * std::log(l) / l, 1e-8) << l; // tau11 = E ln l, P11 = tau11 / F11
    EXPECT_LT(d2.tangent_error(res), 1e-6) << l;
  }
  // non-symmetric F: simple shear, the Kirchhoff stress P F^T is symmetric
  tensor2 F{tmech::eye<T, 3, 2>()};
  F(0, 1) = 0.4;
  auto const res{d.step(F, tensor2{}, control::deformation(), 0, 1)};
  tensor2 const tau{res.stress * tmech::trans(res.F)};
  EXPECT_NEAR(tmech::norm(tensor2{tau - tmech::trans(tau)}), 0, 1e-10);
  EXPECT_LT(d.tangent_error(res), 1e-6);
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

TEST(Hencky, FromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_deformation_gradient_source", "name": "F"},
    {"type": "hencky_elasticity", "name": "h", "deformation_gradient_source": "F", "K": 166.67, "G": 76.92}]})")(ctx, {});
  EXPECT_NE(ctx.find("h"), nullptr);
}

} // namespace
