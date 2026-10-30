#include <gtest/gtest.h>
#include <array>
#include <string>
#include <tmech/tmech.h>
#include <vector>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/incremental_linear_elasticity.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/driver/external_state_source.h"

// Contracts of the engine: folding constants and the order materials are created in change no result.

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;

/// strain -> linear_elasticity "total" (constant tangent) -> incremental_linear_elasticity "inc", created in
/// `order`; returns the strain source.
nm::external_strain_source<policy> *build(ctx_type &ctx, std::vector<std::string> const &order) {
  nm::external_strain_source<policy> *strain{nullptr};
  for (auto const &name : order) {
    param_type p;
    p.insert<std::string>("name", name);
    if (name == "strain") {
      strain = &ctx.create<nm::external_strain_source<policy>>(p);
    } else if (name == "total") {
      p.insert<std::string>("strain_producer_name", "strain");
      p.insert<T>("K", 100.0);
      p.insert<T>("G", 40.0);
      ctx.create<nm::linear_elasticity<policy>>(p);
    } else {
      p.insert<std::string>("strain_source", "strain");
      p.insert<std::string>("tangent_source", "total");
      ctx.create<nm::incremental_linear_elasticity<policy>>(p);
    }
  }
  ctx.finalize();
  return strain;
}

/// The stresses of four steps, the stiffness changed by set_parameter() before the third.
std::vector<tensor2> run(ctx_type &ctx, nm::external_strain_source<policy> &strain) {
  std::vector<tensor2> stresses;
  tensor2 eps;
  for (int step{0}; step < 4; ++step) {
    if (step == 2)
      dynamic_cast<nm::linear_elasticity<policy> &>(*ctx.find("total")).set_parameter<T>("K", 150.0);
    tensor2 const old{eps};
    eps(0, 0) += 1e-3;
    eps(0, 1) = eps(1, 0) = 5e-4 * step;
    strain.bind(old, eps);
    ctx.update();
    ctx.commit();
    stresses.push_back(ctx.get<tensor2>("inc", "stress"));
    stresses.push_back(ctx.get<tensor2>("total", "stress"));
  }
  return stresses;
}

void expect_same(std::vector<tensor2> const &a, std::vector<tensor2> const &b) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i{0}; i < a.size(); ++i)
    EXPECT_LE(tmech::norm(a[i] - b[i]), 1e-12 * (1 + tmech::norm(a[i]))) << "result " << i;
}

std::vector<std::string> const created{"strain", "total", "inc"};

TEST(EngineContracts, FoldingConstantsChangesNoResult) {
  ctx_type folded, unfolded;
  auto *a{build(folded, created)};
  auto *b{build(unfolded, created)};
  unfolded.fold_constants(false);
  expect_same(run(folded, *a), run(unfolded, *b));
  EXPECT_TRUE(folded.is_constant("total", "tangent"));
}

TEST(EngineContracts, TheCreationOrderChangesNoResult) {
  ctx_type forward, backward;
  auto *a{build(forward, created)};
  auto *b{build(backward, {"inc", "total", "strain"})};
  expect_same(run(forward, *a), run(backward, *b));
}

TEST(EngineContracts, PointersAreNeverVerifiable) {
  // a pointer compares by address, not by what it points to
  EXPECT_FALSE(nm::detail::verifiable<double const *>);
  EXPECT_FALSE(nm::detail::verifiable<tensor2 *>);
  EXPECT_TRUE(nm::detail::verifiable<tensor2>);
}

} // namespace
