#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/scalar_split.h"
#include "test_support.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using numsim::materials::test::add;

void build(ctx_type &ctx, std::vector<T> const &weights) {
  add<nm::constant_scalar<policy>>(ctx, "c", {{"value", 2.0}});
  add<nm::scalar_split<policy>>(ctx, "split", {{"source", "c"}, {"weights", std::vector<T>(weights)}});
}

TEST(ScalarSplit, OneOutputPerWeight) {
  ctx_type ctx;
  build(ctx, {0.25, 0.75});
  ctx.finalize();
  ctx.update();
  EXPECT_EQ(ctx.get<T>("split", "value_0"), 0.5);
  EXPECT_EQ(ctx.get<T>("split", "value_1"), 1.5);
}

TEST(ScalarSplit, RejectsNoWeights) {
  ctx_type ctx;
  EXPECT_THROW(build(ctx, {}), std::invalid_argument);
}

} // namespace
