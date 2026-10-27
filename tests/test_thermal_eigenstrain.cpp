#include <gtest/gtest.h>
#include <memory>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/thermal_eigenstrain.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;

constexpr T K{100}, G{40}, alpha{1e-5}, T0{20};

std::unique_ptr<ctx_type> build() {
  auto owner{std::make_unique<ctx_type>()};
  auto &ctx{*owner};
  param_type p;
  p.insert<std::string>("name", "strain");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "temperature");
  ctx.create<nm::external_scalar_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "alpha");
  p.insert<T>("value", alpha);
  ctx.create<nm::constant_scalar<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "thermal");
  p.insert<std::string>("temperature_source", "temperature");
  p.insert<std::string>("alpha_source", "alpha::value");
  p.insert<T>("reference_temperature", T0);
  ctx.create<nm::thermal_eigenstrain<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "strain");
  p.insert<std::string>("eigenstrain_source", "thermal");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  ctx.create<nm::linear_elasticity<policy>>(p);
  ctx.finalize();
  return owner;
}

void load(ctx_type &ctx, tensor2 const &eps, T temperature) {
  dynamic_cast<nm::external_strain_source<policy> *>(ctx.find("strain"))->bind(tensor2{}, eps);
  dynamic_cast<nm::external_scalar_source<policy> *>(ctx.find("temperature"))->bind(T0, temperature);
  ctx.update();
}

TEST(ThermalEigenstrain, FreeExpansionIsStressFree) {
  auto ctx{build()};
  T const dT{50};
  load(*ctx, tensor2{alpha * dT * tmech::eye<T, 3, 2>()}, T0 + dT);
  EXPECT_NEAR(tmech::norm(ctx->get<tensor2>("elastic", "stress")), 0, 1e-12);
  EXPECT_NEAR(ctx->get<tensor2>("thermal", "eigenstrain")(1, 1), alpha * dT, 1e-18);
}

TEST(ThermalEigenstrain, ConstrainedHeatingGivesHydrostaticPressure) {
  auto ctx{build()};
  T const dT{50};
  load(*ctx, tensor2{}, T0 + dT);
  auto const &sig{ctx->get<tensor2>("elastic", "stress")};
  EXPECT_NEAR(sig(0, 0), -3 * K * alpha * dT, 1e-12);
  EXPECT_NEAR(sig(1, 1), sig(0, 0), 1e-12);
  EXPECT_NEAR(sig(0, 1), 0, 1e-15);
}

TEST(ThermalEigenstrain, SuperposesWithMechanicalStrainAndTheTangentIsUnchanged) {
  auto ctx{build()};
  tensor2 eps;
  eps(0, 1) = eps(1, 0) = 1e-3;
  eps(0, 0) = 2e-3;
  load(*ctx, eps, T0 + 10);
  tensor2 const expected{tmech::dcontract(ctx->get<tmech::tensor<T, 3, 4>>("elastic", "tangent"), tensor2{eps - alpha * 10 * tmech::eye<T, 3, 2>()})};
  EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("elastic", "stress") - expected}), 0, 1e-12);
  load(*ctx, eps, T0); // no temperature change: plain elasticity
  EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("elastic", "stress") - tmech::dcontract(ctx->get<tmech::tensor<T, 3, 4>>("elastic", "tangent"), eps)}), 0, 1e-12);
}

TEST(ThermalEigenstrain, FromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "external_scalar_source", "name": "temperature"},
    {"type": "constant_scalar", "name": "alpha", "value": 1e-5},
    {"type": "thermal_eigenstrain", "name": "thermal", "alpha_source": "alpha::value", "reference_temperature": 20.0},
    {"type": "linear_elasticity", "name": "elastic", "strain_producer_name": "strain", "eigenstrain_source": "thermal", "K": 100.0, "G": 40.0}]})")(ctx, {});
  load(ctx, tensor2{}, 120);
  EXPECT_NEAR(ctx.get<tensor2>("elastic", "stress")(2, 2), -3 * 100 * 1e-5 * 100, 1e-12);
}

} // namespace
