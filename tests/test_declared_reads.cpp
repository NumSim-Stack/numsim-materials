#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/incremental_linear_elasticity.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/poisson_ratio_from_moduli.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/driver/external_state_source.h"
#include "test_support.h"

// The shipped materials declare what each callback reads (#168): constness follows from the reads, and a
// constant from the parameters alone is recomputed after set_parameter().

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using numsim::materials::test::add;

nm::constant_scalar<policy> &constant(ctx_type &ctx, std::string const &name, T value) {
  return add<nm::constant_scalar<policy>>(ctx, name, {{"value", value}});
}

TEST(DeclaredReadsMigration, ConstantScalarFollowsSetParameter) {
  ctx_type ctx;
  auto &c{constant(ctx, "c", 1.0)};
  ctx.finalize();
  ctx.update();
  EXPECT_TRUE(ctx.is_constant("c", "value"));
  c.set_parameter<T>("value", 2.0);
  ctx.update();
  EXPECT_EQ(ctx.get<T>("c", "value"), 2.0);
}

TEST(DeclaredReadsMigration, PoissonRatioFollowsSetParameter) {
  ctx_type ctx;
  auto &nu{add<nm::poisson_ratio_from_moduli<policy>>(ctx, "nu", {{"K", 1.0}, {"G", 1.0}})};
  ctx.finalize();
  ctx.update();
  EXPECT_TRUE(ctx.is_constant("nu", "value"));
  nu.set_parameter<T>("K", 2.0);
  ctx.update();
  EXPECT_DOUBLE_EQ(ctx.get<T>("nu", "value"), (3 * 2.0 - 2 * 1.0) / (2 * (3 * 2.0 + 1.0)));
}

TEST(DeclaredReadsMigration, AStatefulStepperIsNeverConstant) {
  ctx_type ctx;
  add<nm::tensor_component_stepper<2, policy>>(ctx, "step",
      {{"increment", 0.5}, {"indices", std::vector<std::size_t>{0, 1}}});
  ctx.finalize();
  for (int i{0}; i < 3; ++i)
    ctx.update();
  EXPECT_FALSE(ctx.is_constant("step", "strain"));
  EXPECT_EQ(ctx.get<tensor2>("step", "strain")(0, 1), 1.5);
}

TEST(DeclaredReadsMigration, IncrementalElasticityCopiesAConstantTangentOnce) {
  ctx_type ctx;
  auto &strain{add<nm::external_strain_source<policy>>(ctx, "strain")};
  add<nm::linear_elasticity<policy>>(ctx, "total", {{"strain_producer_name", "strain"}, {"K", 100.0}, {"G", 40.0}});
  add<nm::incremental_linear_elasticity<policy>>(ctx, "inc",
      {{"strain_source", "strain"}, {"tangent_source", "total"}});
  ctx.finalize();
  tensor2 eps;
  eps(0, 0) = 1e-3;
  strain.bind(tensor2{}, eps);
  ctx.update();
  EXPECT_TRUE(ctx.is_constant("total", "tangent"));
  EXPECT_TRUE(ctx.is_constant("inc", "tangent"));
  EXPECT_FALSE(ctx.is_constant("inc", "stress"));
  EXPECT_NEAR(ctx.get<tensor2>("inc", "stress")(0, 0), (100.0 + 4.0 * 40.0 / 3.0) * 1e-3, 1e-12);
}

} // namespace
