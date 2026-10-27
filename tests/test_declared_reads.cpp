#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/poisson_ratio_from_moduli.h"
#include "numsim-materials/materials/tensor_component_stepper.h"

// The shipped materials declare what each callback reads (#168): constness follows from the reads, and a
// constant from the parameters alone is recomputed after set_parameter().

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;

nm::constant_scalar<policy> &constant(ctx_type &ctx, std::string const &name, T value) {
  param_type p;
  p.insert<std::string>("name", name);
  p.insert<T>("value", value);
  return ctx.create<nm::constant_scalar<policy>>(p);
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
  param_type p;
  p.insert<std::string>("name", "nu");
  p.insert<T>("K", 1.0);
  p.insert<T>("G", 1.0);
  auto &nu{ctx.create<nm::poisson_ratio_from_moduli<policy>>(p)};
  ctx.finalize();
  ctx.update();
  EXPECT_TRUE(ctx.is_constant("nu", "value"));
  nu.set_parameter<T>("K", 2.0);
  ctx.update();
  EXPECT_DOUBLE_EQ(ctx.get<T>("nu", "value"), (3 * 2.0 - 2 * 1.0) / (2 * (3 * 2.0 + 1.0)));
}

TEST(DeclaredReadsMigration, AStatefulStepperIsNeverConstant) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "step");
  p.insert<T>("increment", 0.5);
  p.insert<std::vector<std::size_t>>("indices", {0, 1});
  ctx.create<nm::tensor_component_stepper<2, policy>>(p);
  ctx.finalize();
  for (int i{0}; i < 3; ++i)
    ctx.update();
  EXPECT_FALSE(ctx.is_constant("step", "strain"));
  EXPECT_EQ(ctx.get<tensor2>("step", "strain")(0, 1), 1.5);
}

} // namespace
