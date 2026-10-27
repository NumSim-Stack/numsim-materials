#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/scalar_stepper.h"

// A checkpoint names each history property: it restores into a context built in another order, and refuses one
// whose properties differ.

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

void stepper(ctx_type &ctx, std::string const &name, T increment) {
  param_type p;
  p.insert<std::string>("name", name);
  p.insert<T>("increment", increment);
  ctx.create<nm::scalar_stepper<policy>>(p);
}

/// Steppers "a" (+1) and "b" (+10), created in the given order.
void build(ctx_type &ctx, std::vector<std::string> const &order) {
  for (auto const &name : order)
    stepper(ctx, name, name == "a" ? 1.0 : 10.0);
  ctx.finalize();
}

struct checkpoint {
  std::string path{(std::filesystem::temp_directory_path() / ("numsim_checkpoint_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
                                                             ::testing::UnitTest::GetInstance()->current_test_info()->name() + ".bin")).string()};
  ~checkpoint() { std::filesystem::remove(path); }
};

/// Saves after two committed steps.
void save_two_steps(std::vector<std::string> const &order, std::string const &path) {
  ctx_type ctx;
  build(ctx, order);
  for (int i{0}; i < 2; ++i) {
    ctx.update();
    ctx.commit();
  }
  ctx.save_state(path);
}

TEST(Checkpoint, RestoresIntoAContextBuiltInAnotherOrder) {
  checkpoint file;
  save_two_steps({"a", "b"}, file.path);
  ctx_type ctx;
  build(ctx, {"b", "a"});
  ctx.load_state(file.path);
  ctx.update();
  EXPECT_EQ(ctx.get<T>("a", "state"), 3.0);
  EXPECT_EQ(ctx.get<T>("b", "state"), 30.0);
}

TEST(Checkpoint, RestoredContinuationMatchesTheUninterruptedRun) {
  checkpoint file;
  save_two_steps({"a", "b"}, file.path);
  ctx_type restored, uninterrupted;
  build(restored, {"a", "b"});
  restored.load_state(file.path);
  build(uninterrupted, {"a", "b"});
  for (int i{0}; i < 2; ++i) {
    uninterrupted.update();
    uninterrupted.commit();
  }
  for (int i{0}; i < 3; ++i) {
    restored.update();
    restored.commit();
    uninterrupted.update();
    uninterrupted.commit();
    EXPECT_EQ(restored.get<T>("a", "state"), uninterrupted.get<T>("a", "state"));
    EXPECT_EQ(restored.get<T>("b", "state"), uninterrupted.get<T>("b", "state"));
  }
}

TEST(Checkpoint, RefusesAContextWithOtherProperties) {
  checkpoint file;
  save_two_steps({"a", "b"}, file.path);
  ctx_type ctx;
  build(ctx, {"a", "c"});
  try {
    ctx.load_state(file.path);
    FAIL() << "no throw";
  } catch (std::runtime_error const &e) {
    std::string const what{e.what()};
    EXPECT_NE(what.find("b::state"), std::string::npos) << what;
  }
}

TEST(Checkpoint, RefusesATruncatedFile) {
  checkpoint file;
  save_two_steps({"a", "b"}, file.path);
  auto const size{std::filesystem::file_size(file.path)};
  std::filesystem::resize_file(file.path, size - 3);
  ctx_type ctx;
  build(ctx, {"a", "b"});
  EXPECT_THROW(ctx.load_state(file.path), std::runtime_error);
}

} // namespace
