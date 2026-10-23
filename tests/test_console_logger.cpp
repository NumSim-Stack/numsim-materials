#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/postprocessing/property_console_logger.h"
#include "numsim-materials/umat/external_state_source.h"

// The console logger prints its inputs as a table on every update.

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

TEST(PropertyConsoleLogger, PrintsItsInputsOnEveryUpdate) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "x");
  auto &x{ctx.create<nm::external_scalar_source<policy>>(p)};
  p.clear();
  p.insert<std::string>("name", "log");
  p.insert<std::unordered_map<std::string, std::vector<std::string>>>("input", {{"x", {"state"}}});
  ctx.create<nm::property_console_logger<policy>>(p);
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
