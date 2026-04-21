#include <gtest/gtest.h>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/history_property.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/scalar_stepper.h"
#include "numsim-materials/materials/scalar_identity_weight.h"
#include "numsim-materials/materials/autocatalytic_reaction.h"
#include "numsim-materials/solvers/backward_euler.h"

namespace {

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

// --- Linear elasticity ---

class LinearElasticityTest : public ::testing::Test {
protected:
  void SetUp() override {
    param_type p;

    p.clear();
    p.insert<std::string>("name", "stepper");
    p.insert<T>("increment", T{0.001});
    p.insert<std::vector<std::size_t>>("indices", {0, 0});
    ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "stepper");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    ctx.create<numsim::materials::linear_elasticity<policy>>(p);

    ctx.finalize();
  }

  ctx_type ctx;
  T K{166.67};
  T G{76.92};
};

TEST_F(LinearElasticityTest, StressIsLinear) {
  ctx.update();
  const auto sig1 = ctx.get<tensor2>("elastic", "stress")(0,0);
  ctx.commit();

  ctx.update();
  const auto sig2 = ctx.get<tensor2>("elastic", "stress")(0,0);

  EXPECT_NEAR(sig2 / sig1, 2.0, 1e-10);
}

TEST_F(LinearElasticityTest, TangentSymmetry) {
  ctx.update();
  const auto& C = ctx.get<tensor4>("elastic", "tangent");

  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t l = 0; l < 3; ++l) {
          EXPECT_NEAR(C(i,j,k,l), C(k,l,i,j), 1e-12)
              << "Major symmetry violated at (" << i << "," << j << "," << k << "," << l << ")";
          EXPECT_NEAR(C(i,j,k,l), C(j,i,k,l), 1e-12)
              << "Left minor symmetry violated at (" << i << "," << j << "," << k << "," << l << ")";
        }
}

TEST_F(LinearElasticityTest, BulkModulus) {
  ctx.update();
  const auto& C = ctx.get<tensor4>("elastic", "tangent");
  // C(0,0,0,0) = K + 4G/3
  T expected = K + T{4} * G / T{3};
  EXPECT_NEAR(C(0,0,0,0), expected, 1e-2);
}

// --- Scalar weight ---

TEST(ScalarWeight, IdentityWeight) {
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "time");
  p.insert<T>("increment", T{0.5});
  ctx.create<numsim::materials::scalar_stepper<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "w_time");
  p.insert<std::string>("source", "time::state");
  ctx.create<numsim::materials::scalar_identity_weight<policy>>(p);

  ctx.finalize();
  ctx.update();

  auto weight = ctx.get<T>("w_time", "value");
  EXPECT_NEAR(weight, 0.5, 1e-12);

  ctx.commit();
  ctx.update();
  weight = ctx.get<T>("w_time", "value");
  EXPECT_NEAR(weight, 1.0, 1e-12);
}

// --- Curing simulation with backward Euler ---

// TODO: scalar_stepper uses history_property — get_mutable<T> fails because
// it finds a history_property, not a property<T>. Need a set_initial_value API.
TEST(CuringSimulation, DISABLED_ConvergesToFullCure) {
  ctx_type ctx;
  param_type p;

  // Temperature (constant)
  p.clear();
  p.insert<std::string>("name", "temperature");
  p.insert<T>("increment", T{0});
  ctx.create<numsim::materials::scalar_stepper<policy>>(p);

  // Time stepper
  p.clear();
  p.insert<std::string>("name", "time");
  p.insert<T>("increment", T{10});
  ctx.create<numsim::materials::scalar_stepper<policy>>(p);

  // Backward Euler solver
  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<std::string>("function", "curing");
  ctx.create<numsim::materials::backward_euler<policy>>(p);

  // Autocatalytic reaction
  p.clear();
  p.insert<std::string>("name", "curing");
  p.insert<T>("A", T{1e6});
  p.insert<T>("E", T{50000});
  p.insert<T>("n", T{1.2});
  p.insert<T>("m", T{0.8});
  p.insert<std::string>("temperature_name", "temperature");
  p.insert<std::string>("timer_name", "time");
  p.insert<std::string>("solver_name", "solver");
  ctx.create<numsim::materials::autocatalytic_reaction<policy>>(p);

  ctx.finalize();

  // Initialize temperature to 80°C (353 K)
  // scalar_stepper uses history_property — set initial value before first update
  auto* temp_prop = ctx.find_property("temperature", "state");
  auto* temp_hist = dynamic_cast<numsim_core::history_property<T, numsim::materials::property_traits>*>(temp_prop);
  temp_hist->old_value() = T{353};
  temp_hist->new_value() = T{353};

  T curing = 0;
  for (int i = 0; i < 20; ++i) {
    ctx.update();
    curing = ctx.get<T>("curing", "current_state");
    ctx.commit();
  }

  EXPECT_GT(curing, 0.99) << "Curing should approach 1.0 after 20 steps at 80°C";
}

} // namespace
