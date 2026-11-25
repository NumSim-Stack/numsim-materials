#include <gtest/gtest.h>
#include <stdexcept>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/isotropic_moduli.h"
#include "numsim-materials/materials/isotropic_tangent.h"
#include "numsim-materials/materials/poisson_ratio_from_moduli.h"
#include "numsim-materials/umat/json_model.h"
#include "test_support.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using numsim::materials::test::add;

constexpr T K{100}, G{40};

void add_constant(ctx_type &ctx, std::string const &name, T value) {
  add<nm::constant_scalar<policy>>(ctx, name, {{"value", value}});
}

// a constant: the value is there from construction on and needs no update
TEST(ScalarProducers, PoissonsRatioFromTwoModuliIsFixedAtConstruction) {
  ctx_type ctx;
  add<nm::poisson_ratio_from_moduli<policy>>(ctx, "nu", {{"K", K}, {"G", G}});
  ctx.finalize();
  T const expected{(3 * K - 2 * G) / (2 * (3 * K + G))};
  EXPECT_NEAR(ctx.get<T>("nu", "value"), expected, 1e-15);
  ctx.update();
  EXPECT_NEAR(ctx.get<T>("nu", "value"), expected, 1e-15);
}

TEST(ScalarProducers, PoissonsRatioRejectsNonPositiveModuli) {
  for (auto const &[k, g] : {std::pair{0.0, G}, std::pair{K, -1.0}}) {
    ctx_type ctx;
    EXPECT_THROW((add<nm::poisson_ratio_from_moduli<policy>>(ctx, "nu", {{"K", k}, {"G", g}})), std::invalid_argument);
  }
}

// K, G -> isotropic_tangent -> isotropic_moduli round trip, all through the graph
TEST(ScalarProducers, IsotropicModuliRecoverTheModuliOfAnIsotropicTangent) {
  ctx_type ctx;
  add_constant(ctx, "K", K);
  add_constant(ctx, "G", G);
  add<nm::isotropic_tangent<policy>>(ctx, "stiffness", {{"K_source", "K"}, {"G_source", "G"}});
  add<nm::isotropic_moduli<policy>>(ctx, "moduli", {{"tangent_source", "stiffness"}});
  ctx.finalize();
  ctx.update();
  EXPECT_NEAR(ctx.get<T>("moduli", "K"), K, 1e-12);
  EXPECT_NEAR(ctx.get<T>("moduli", "G"), G, 1e-12);
  EXPECT_NEAR(ctx.get<T>("moduli", "poisson_ratio"), nm::poisson_ratio(K, G), 1e-14);
}

TEST(ScalarProducers, FromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "constant_scalar", "name": "K", "value": 100},
    {"type": "constant_scalar", "name": "G", "value": 40},
    {"type": "isotropic_tangent", "name": "stiffness", "K_source": "K", "G_source": "G"},
    {"type": "isotropic_moduli", "name": "moduli", "tangent_source": "stiffness"},
    {"type": "poisson_ratio_from_moduli", "name": "nu", "K": 100, "G": 40}]})")(ctx, {});
  ctx.update();
  EXPECT_NEAR(ctx.get<T>("nu", "value"), nm::poisson_ratio(K, G), 1e-14);
  EXPECT_NEAR(ctx.get<T>("moduli", "poisson_ratio"), ctx.get<T>("nu", "value"), 1e-14);
}

} // namespace
