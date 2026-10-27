#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/scalar_split.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

void build(ctx_type &ctx, std::vector<T> const &weights) {
  param_type p;
  p.insert<std::string>("name", "c");
  p.insert<T>("value", 2.0);
  ctx.create<nm::constant_scalar<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "split");
  p.insert<std::string>("source", "c");
  p.insert<std::vector<T>>("weights", weights);
  ctx.create<nm::scalar_split<policy>>(p);
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
