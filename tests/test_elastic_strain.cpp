#include <gtest/gtest.h>
#include <memory>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/partial_check.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/elastic_strain.h"
#include "numsim-materials/materials/thermal_eigenstrain.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using tensor2 = tmech::tensor<T, 3, 2>;

constexpr T alpha{1e-5}, beta{-2e-3}, T0{20};

/// Two eigenstrains: thermal (alpha, temperature) and shrinkage (beta, cure), plus the elastic strain over `sources`.
std::string deck(std::string const &sources) {
  return R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "external_scalar_source", "name": "temperature"},
    {"type": "external_scalar_source", "name": "cure"},
    {"type": "constant_scalar", "name": "alpha", "value": 1e-5},
    {"type": "constant_scalar", "name": "beta", "value": -2e-3},
    {"type": "thermal_eigenstrain", "name": "thermal", "alpha_source": "alpha::value", "reference_temperature": 20.0},
    {"type": "thermal_eigenstrain", "name": "shrinkage", "alpha_source": "beta::value", "temperature_source": "cure"},
    {"type": "elastic_strain", "name": "ee", "strain_source": "strain", "eigenstrain_sources": [)" + sources + R"(]}]})";
}

tensor2 sym(T a, T b, T c, T d, T e, T f) {
  tensor2 x;
  x(0, 0) = a, x(1, 1) = b, x(2, 2) = c;
  x(1, 2) = x(2, 1) = d, x(0, 2) = x(2, 0) = e, x(0, 1) = x(1, 0) = f;
  return x;
}
tensor2 const eps{sym(1e-3, -4e-4, 2e-4, 5e-4, -1e-4, 3e-4)};

void load(ctx_type &ctx, T temperature, T cure) {
  dynamic_cast<nm::external_strain_source<policy> *>(ctx.find("strain"))->bind(tensor2{}, eps);
  dynamic_cast<nm::external_scalar_source<policy> *>(ctx.find("temperature"))->bind(T0, temperature);
  dynamic_cast<nm::external_scalar_source<policy> *>(ctx.find("cure"))->bind(0, cure);
  ctx.update();
}

TEST(ElasticStrain, SubtractsEveryEigenstrain) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(deck(R"("thermal", "shrinkage")"))(ctx, {});
  load(ctx, 70, 0.5);
  tensor2 const expected{eps - ctx.get<tensor2>("thermal", "eigenstrain") - ctx.get<tensor2>("shrinkage", "eigenstrain")};
  EXPECT_NEAR(tmech::norm(tensor2{ctx.get<tensor2>("ee", "strain") - expected}), 0, 1e-18);
  EXPECT_NEAR(ctx.get<tensor2>("shrinkage", "eigenstrain")(0, 0), beta * 0.5, 1e-18);
  EXPECT_NEAR(ctx.get<tensor2>("ee", "strain")(1, 1), -4e-4 - alpha * 50 - beta * 0.5, 1e-15);
  EXPECT_NEAR(ctx.get<tensor2>("ee", "strain")(0, 1), 3e-4, 1e-18); // eigenstrains are spherical
}

TEST(ElasticStrain, WithoutEigenstrainsItPassesTheStrainThrough) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(deck(""))(ctx, {});
  load(ctx, 70, 0.5);
  EXPECT_EQ(tmech::norm(tensor2{ctx.get<tensor2>("ee", "strain") - eps}), 0);
}

TEST(ElasticStrain, ASourceNamesAMaterialOrAMaterialAndProperty) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(deck(R"("thermal::eigenstrain")"))(ctx, {});
  load(ctx, 70, 0.5);
  EXPECT_NEAR(ctx.get<tensor2>("ee", "strain")(2, 2), 2e-4 - alpha * 50, 1e-15);
}

TEST(ElasticStrain, TheStrainPropertyIsConfigurable) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "elastic_strain", "name": "ee", "strain_source": "strain", "strain_property": "strain"}]})")(ctx, {});
  EXPECT_NE(ctx.find("ee"), nullptr);
}

TEST(ElasticStrain, PartialsPassTheFiniteDifferenceCheck) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(deck(R"("thermal", "shrinkage")"))(ctx, {});
  load(ctx, 70, 0.5);
  EXPECT_FALSE(ctx.find("ee")->partials().empty());
  for (auto const &mismatch : nm::check_partials(ctx, "ee"))
    ADD_FAILURE() << mismatch;
}

} // namespace
