#include <gtest/gtest.h>

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/default_materials.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "test_support.h"

#include <string>
#include <vector>

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using T = policy::value_type;
using nm::test::add;

namespace {

param_type inserted(nm::test::parameter<ctx_type> const &entry) {
  param_type p;
  entry.insert_into(p);
  return p;
}

} // namespace

TEST(TestSupport, StringsAreStoredAsStdString) {
  EXPECT_EQ(inserted({"s", "abc"}).get<std::string>("s"), "abc");
  EXPECT_EQ(inserted({"s", std::string{"def"}}).get<std::string>("s"), "def");
}

TEST(TestSupport, FloatingPointValuesAreStoredAsTheValueType) {
  EXPECT_EQ(inserted({"x", 1.5f}).get<T>("x"), T{1.5});
  EXPECT_EQ(inserted({"x", 2.0}).get<T>("x"), T{2});
}

TEST(TestSupport, OtherValuesAreStoredAsWritten) {
  EXPECT_EQ(inserted({"n", 3}).get<int>("n"), 3);
  EXPECT_TRUE(inserted({"b", true}).get<bool>("b"));
  EXPECT_EQ(inserted({"v", std::vector<std::size_t>{1, 2}}).get<std::vector<std::size_t>>("v"),
            (std::vector<std::size_t>{1, 2}));
}

TEST(TestSupport, AddCreatesTheMaterialWithItsNameAndParameters) {
  ctx_type ctx;
  auto &c{add<nm::constant_scalar<policy>>(ctx, "c", {{"value", 2.5}})};
  ctx.finalize();
  ctx.update();
  EXPECT_EQ(c.name(), "c");
  EXPECT_EQ(ctx.get<T>("c", "value"), T{2.5});
}
