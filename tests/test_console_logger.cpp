#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/postprocessing/property_console_logger.h"
#include "numsim-materials/driver/external_state_source.h"
#include "test_support.h"

// The console logger prints its inputs as a table on every update.

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using numsim::materials::test::add;

TEST(PropertyConsoleLogger, PrintsItsInputsOnEveryUpdate) {
  ctx_type ctx;
  auto &x{add<nm::external_scalar_source<policy>>(ctx, "x")};
  add<nm::property_console_logger<policy>>(ctx, "log",
      {{"input", std::unordered_map<std::string, std::vector<std::string>>{{"x", {"state"}}}}});
  ctx.finalize();
  for (double const value : {1.5, 2.5}) {
    x.bind(0.0, value);
    testing::internal::CaptureStdout();
    ctx.update();
    auto const out{testing::internal::GetCapturedStdout()};
    EXPECT_NE(out.find("state"), std::string::npos) << out;
    EXPECT_NE(out.find(std::format("{:.6g}", value)), std::string::npos) << out;
  }
}

} // namespace
