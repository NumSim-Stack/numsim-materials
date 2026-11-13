#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/driver/deformation_gradient_driver.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/materials/elastic_strain.h"
#include "numsim-materials/materials/hencky_elasticity.h"
#include "numsim-materials/materials/kirchhoff_to_first_piola.h"
#include "numsim-materials/materials/log_strain.h"
#include "numsim-materials/umat/json_model.h"

// The Hencky chain F -> log_strain -> elastic_strain -> stress -> kirchhoff_to_first_piola against hencky_elasticity.

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using driver = nm::deformation_gradient_driver<policy>;
using control = nm::deformation_control<T>;

/// The old material and the chain side by side on one F; `fixed` takes linear_elasticity with K, G instead of a produced stiffness.
std::string deck(bool eigenstrain, bool fixed) {
  std::string const eig{eigenstrain ? R"("thermal")" : ""};
  std::string const tangent{fixed ? "elastic" : "stiffness"};
  std::string const stress{fixed ? R"({"type": "linear_elasticity", "name": "elastic", "strain_producer_name": "ee", "K": 166.67, "G": 76.92})"
                                 : R"({"type": "linear_stress", "name": "tau", "tangent_source": "stiffness", "strain_source": "ee"})"};
  return R"({"materials": [
    {"type": "external_deformation_gradient_source", "name": "F"},
    {"type": "external_scalar_source", "name": "temperature"},
    {"type": "constant_scalar", "name": "K", "value": 166.67},
    {"type": "constant_scalar", "name": "G", "value": 76.92},
    {"type": "constant_scalar", "name": "alpha", "value": 1e-4},
    {"type": "isotropic_tangent", "name": "stiffness", "K_source": "K", "G_source": "G"},
    {"type": "thermal_eigenstrain", "name": "thermal", "alpha_source": "alpha::value", "reference_temperature": 20.0},
    {"type": "hencky_elasticity", "name": "hencky", "deformation_gradient_source": "F", "tangent_source": "stiffness")" +
         std::string{eigenstrain ? R"(, "eigenstrain_source": "thermal")" : ""} + R"(},
    {"type": "log_strain", "name": "lnV", "deformation_gradient_source": "F"},
    {"type": "elastic_strain", "name": "ee", "strain_source": "lnV", "eigenstrain_sources": [)" + eig + R"(]},
    )" + stress + R"(,
    {"type": "kirchhoff_to_first_piola", "name": "pf", "stress_source": ")" + std::string{fixed ? "elastic" : "tau"} +
         R"(", "tangent_source": ")" + tangent + R"(", "strain_source": "lnV", "deformation_gradient_source": "F"}]})";
}

/// Rotation, stretch and shear.
tensor2 general_F() {
  tmech::tensor<T, 3, 1> axis;
  axis(0) = 1, axis(1) = 2, axis(2) = 2;
  tensor2 U{tmech::eye<T, 3, 2>()};
  U(0, 0) = 1.25, U(1, 1) = 0.9, U(2, 2) = 1.1, U(0, 1) = U(1, 0) = 0.15, U(1, 2) = U(2, 1) = -0.08;
  tensor2 F{nm::rotation(tmech::tensor<T, 3, 1>{axis / 3}, T{0.7}) * U};
  F(0, 1) += 0.2;
  return F;
}

T relative(tensor2 const &a, tensor2 const &b) { return tmech::norm(tensor2{a - b}) / tmech::norm(b); }
T relative(tensor4 const &a, tensor4 const &b) { return tmech::norm(tensor4{a - b}) / tmech::norm(b); }

void check_equivalence(bool eigenstrain, bool fixed) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(deck(eigenstrain, fixed))(ctx, {});
  dynamic_cast<nm::external_deformation_gradient_source<policy> *>(ctx.find("F"))->bind(tmech::eye<T, 3, 2>(), general_F());
  dynamic_cast<nm::external_scalar_source<policy> *>(ctx.find("temperature"))->bind(20, 120);
  ctx.update();
  EXPECT_LE(relative(ctx.get<tensor2>("pf", "stress"), ctx.get<tensor2>("hencky", "stress")), 1e-14);
  EXPECT_LE(relative(ctx.get<tensor4>("pf", "tangent"), ctx.get<tensor4>("hencky", "tangent")), 1e-14);
  EXPECT_LE(relative(ctx.get<tensor2>("lnV", "strain"), ctx.get<tensor2>("hencky", "log_strain")), 1e-14);
}

TEST(HenckyChain, EqualsHenckyElasticityWithAProducedStiffness) { check_equivalence(false, false); }
TEST(HenckyChain, EqualsHenckyElasticityWithAThermalEigenstrain) { check_equivalence(true, false); }
TEST(HenckyChain, LinearElasticityEqualsHenckyOnTheSameConstants) { check_equivalence(false, true); }
TEST(HenckyChain, LinearElasticityWithAThermalEigenstrainEqualsHencky) { check_equivalence(true, true); }

TEST(HenckyChain, TangentMatchesFiniteDifferencesOfTheWholeChain) {
  for (bool const fixed : {false, true}) {
    ctx_type ctx;
    nm::umat::make_json_builder<policy>(deck(true, fixed))(ctx, {});
    dynamic_cast<nm::external_scalar_source<policy> *>(ctx.find("temperature"))->bind(20, 60);
    policy::ParameterHandler p;
    p.insert<std::string>("deformation_gradient_source", "F");
    p.insert<std::string>("stress_source", "pf");
    driver d(ctx, p);
    auto const res{d.step(general_F(), tensor2{}, control::deformation(), 0, 1)};
    ASSERT_TRUE(res.converged);
    EXPECT_LT(d.tangent_error(res), 1e-6) << fixed;
  }
}

} // namespace
