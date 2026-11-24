#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
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
    {"type": "external_scalar_source", "name": "temperature"},
    {"type": "constant_scalar", "name": "alpha", "value": 1e-5},
    {"type": "thermal_eigenstrain", "name": "thermal", "alpha_source": "alpha::value", "reference_temperature": 20.0},
    {"type": "elastic_strain", "name": "ee", "strain_source": "thermal", "strain_property": "eigenstrain"}]})")(ctx, {});
  dynamic_cast<nm::external_scalar_source<policy> *>(ctx.find("temperature"))->bind(T0, 70);
  ctx.update();
  EXPECT_NEAR(ctx.get<tensor2>("ee", "strain")(0, 0), alpha * 50, 1e-18);
}

TEST(ElasticStrain, TheSameEigenstrainTwiceIsRejected) {
  for (auto const *sources : {R"("thermal", "thermal")", R"("thermal", "thermal::eigenstrain")"}) {
    ctx_type ctx;
    try {
      nm::umat::make_json_builder<policy>(deck(sources))(ctx, {});
      ADD_FAILURE() << sources << " built";
    } catch (std::invalid_argument const &e) {
      EXPECT_NE(std::string{e.what()}.find("'thermal::eigenstrain' twice"), std::string::npos) << e.what();
    }
  }
}

// sigma = C : (eps - eps_th(T) - eps_sh(c)): d sigma / dT = -3 K alpha I and d sigma / dc = -3 K beta I, through both
// eigenstrains at once
TEST(ElasticStrain, StressDerivativesWithRespectToBothEigenstrainDriversMatchFiniteDifferences) {
  ctx_type ctx;
  std::string text{deck(R"("thermal", "shrinkage")")};
  text.insert(text.rfind("]}"), R"(,
    {"type": "linear_elasticity", "name": "elastic", "strain_producer_name": "ee", "K": 100.0, "G": 40.0})");
  nm::umat::make_json_builder<policy>(text)(ctx, {});
  auto const &dT{ctx.derivative<tensor2>("elastic", "stress", "temperature", "state")};
  auto const &dc{ctx.derivative<tensor2>("elastic", "stress", "cure", "state")};
  T const theta{70}, cure{0.5}, hT{1e-3}, hc{1e-5};
  load(ctx, theta, cure);
  tensor2 const analytic_T{dT}, analytic_c{dc};
  auto const stress_at{[&](T temperature, T c) {
    load(ctx, temperature, c);
    return tensor2{ctx.get<tensor2>("elastic", "stress")};
  }};
  tensor2 const fd_T{(stress_at(theta + hT, cure) - stress_at(theta - hT, cure)) / (2 * hT)};
  tensor2 const fd_c{(stress_at(theta, cure + hc) - stress_at(theta, cure - hc)) / (2 * hc)};
  EXPECT_LT(tmech::norm(tensor2{analytic_T - fd_T}), 1e-8 * tmech::norm(fd_T));
  EXPECT_LT(tmech::norm(tensor2{analytic_c - fd_c}), 1e-8 * tmech::norm(fd_c));
  auto const I{tmech::eye<T, 3, 2>()};
  EXPECT_LT(tmech::norm(tensor2{analytic_T + 3 * 100.0 * alpha * I}), 1e-12) << "-3 K alpha I";
  EXPECT_LT(tmech::norm(tensor2{analytic_c + 3 * 100.0 * beta * I}), 1e-12) << "-3 K beta I";
}

TEST(ElasticStrain, AnEntryWithoutMaterialOrPropertyIsRejected) {
  for (auto const *source : {"thermal::", "::eigenstrain", ""}) {
    ctx_type ctx;
    try {
      nm::umat::make_json_builder<policy>(deck("\"" + std::string{source} + "\""))(ctx, {});
      ADD_FAILURE() << "'" << source << "' built";
    } catch (std::invalid_argument const &e) {
      EXPECT_NE(std::string{e.what()}.find("eigenstrain_sources: '" + std::string{source} + "' is not"), std::string::npos) << e.what();
    }
  }
}

TEST(ParseSource, ABareNameReadsTheDefaultProperty) {
  auto const bare{nm::parse_source("thermal", "eigenstrain")}, full{nm::parse_source("shrinkage::strain", "eigenstrain")};
  EXPECT_EQ(bare.material, "thermal");
  EXPECT_EQ(bare.property, "eigenstrain");
  EXPECT_EQ(full.material, "shrinkage");
  EXPECT_EQ(full.property, "strain");
  for (auto const *source : {"thermal::", "::eigenstrain", "", "::"})
    EXPECT_THROW(nm::parse_source(source, "eigenstrain"), std::invalid_argument) << source;
}

// #240 moved the eigenstrain out of linear_elasticity: an old deck must not lose it silently
TEST(LinearElasticity, TheRemovedEigenstrainSourceIsAnError) {
  ctx_type ctx;
  try {
    nm::umat::make_json_builder<policy>(R"({"materials": [
      {"type": "external_strain_source", "name": "strain"},
      {"type": "linear_elasticity", "name": "el", "strain_producer_name": "strain", "K": 1.0, "G": 1.0,
       "eigenstrain_source": "thermal"}]})")(ctx, {});
    ADD_FAILURE() << "built";
  } catch (std::invalid_argument const &e) {
    EXPECT_NE(std::string{e.what()}.find("elastic_strain"), std::string::npos) << e.what();
  }
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
