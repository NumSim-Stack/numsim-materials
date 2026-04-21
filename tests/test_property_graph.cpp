#include <gtest/gtest.h>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/scalar_stepper.h"
#include "numsim-materials/materials/scalar_identity_weight.h"
#include "numsim-materials/default_materials.h"

namespace {

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;

// --- Context and execution order ---

TEST(PropertyGraph, ExecutionOrder) {
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.1});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", T{166.67});
  p.insert<T>("G", T{76.92});
  ctx.create<numsim::materials::linear_elasticity<policy>>(p);

  ctx.finalize();

  const auto& order = ctx.execution_order();
  ASSERT_EQ(order.size(), 2u);
  EXPECT_EQ(order[0]->name(), "stepper");
  EXPECT_EQ(order[1]->name(), "elastic");
}

TEST(PropertyGraph, UpdateProducesValues) {
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.1});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", T{166.67});
  p.insert<T>("G", T{76.92});
  ctx.create<numsim::materials::linear_elasticity<policy>>(p);

  ctx.finalize();
  ctx.update();

  const auto& eps = ctx.get<tensor2>("stepper", "strain");
  EXPECT_NEAR(eps(0,0), 0.1, 1e-12);

  const auto& sig = ctx.get<tensor2>("elastic", "stress");
  EXPECT_GT(std::abs(sig(0,0)), 0.0);
}

TEST(PropertyGraph, CommitAdvancesHistory) {
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.1});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

  ctx.finalize();
  ctx.update();
  ctx.commit();
  ctx.update();

  const auto& eps = ctx.get<tensor2>("stepper", "strain");
  EXPECT_NEAR(eps(0,0), 0.2, 1e-12);
}

TEST(PropertyGraph, WireInputsThrowsOnMissing) {
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "nonexistent");
  p.insert<T>("K", T{166.67});
  p.insert<T>("G", T{76.92});
  ctx.create<numsim::materials::linear_elasticity<policy>>(p);

  EXPECT_THROW(ctx.finalize(), std::runtime_error);
}

TEST(PropertyGraph, SelectivePropertyUpdate) {
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.1});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", T{166.67});
  p.insert<T>("G", T{76.92});
  ctx.create<numsim::materials::linear_elasticity<policy>>(p);

  ctx.finalize();
  ctx.update();

  const auto sig_before = ctx.get<tensor2>("elastic", "stress");
  ctx.update_property("elastic", "stress");
  const auto sig_after = ctx.get<tensor2>("elastic", "stress");

  // Stepper runs again via upstream, so strain increments
  EXPECT_GT(std::abs(sig_after(0,0)), std::abs(sig_before(0,0)));
}

TEST(PropertyGraph, FactoryConstruction) {
  numsim::materials::register_default_materials<policy>();

  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.5});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create("tensor_component_stepper_rank2", p);

  ctx.finalize();
  ctx.update();

  const auto& eps = ctx.get<tensor2>("stepper", "strain");
  EXPECT_NEAR(eps(0,0), 0.5, 1e-12);
}

} // namespace
