#include <gtest/gtest.h>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/scalar_stepper.h"
#include "numsim-materials/solvers/local_newton.h"
#include "numsim-materials/materials/j2_plasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
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

TEST(CuringSimulation, ConvergesToFullCure) {
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
  ctx.get_mutable<T>("temperature", "state") = T{353};

  T curing = 0;
  for (int i = 0; i < 20; ++i) {
    ctx.update();
    curing = ctx.get<T>("curing", "current_state");
    ctx.commit();
  }

  EXPECT_GT(curing, 0.99) << "Curing should approach 1.0 after 20 steps at 80°C";
}

} // namespace

namespace {
namespace nm_be = numsim::materials;

/// backward_euler is the GRAPH-driven solver, so "function" is required.
///
/// It used to default to empty, which silently selected a second,
/// callback-driven mode inside the same class: no inputs were created, update()
/// was never bound, and "delta" stayed at zero. A consumer reading it froze at
/// its start value for the whole analysis with no error -- measured, a cure
/// that should reach 1.0 sat at 0.01 for 30 steps. That mode is local_newton
/// now, chosen by naming a type rather than by omitting a parameter.
TEST(BackwardEulerSetup, RequiresAFunctionToSolve) {
  using policy = nm_be::material_policy_default;
  nm_be::material_context<policy> ctx;
  policy::ParameterHandler p;
  p.insert<std::string>("name", "solver");
  EXPECT_THROW(ctx.create<nm_be::backward_euler<policy>>(p),
               std::invalid_argument)
      << "omitting \"function\" must fail at setup, not produce an inert "
         "solver whose consumers silently never advance";
}

/// A function material that does not publish residual/jacobian is caught at
/// finalize, by name.
TEST(BackwardEulerSetup, RejectsAFunctionWithoutResidualOrJacobian) {
  using policy = nm_be::material_policy_default;
  using T2 = policy::value_type;
  nm_be::material_context<policy> ctx;
  policy::ParameterHandler p;
  p.insert<std::string>("name", "drv");
  p.insert<T2>("increment", T2{0.1});
  ctx.create<nm_be::scalar_stepper<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<std::string>("function", std::string("drv"));
  ctx.create<nm_be::backward_euler<policy>>(p);
  EXPECT_THROW(ctx.finalize(), std::runtime_error);
}

/// local_newton is the material-driven one: no function, no graph inputs, and
/// its result carries convergence so a caller cannot read the number without
/// the flag being at hand.
TEST(LocalNewtonSolver, SolvesAndReportsConvergence) {
  using policy = nm_be::material_policy_default;
  using T2 = policy::value_type;
  nm_be::material_context<policy> ctx;
  policy::ParameterHandler p;
  p.insert<std::string>("name", "solver");
  auto& s = ctx.create<nm_be::local_newton<policy>>(p);
  ctx.finalize();

  // x^2 - 4 = 0 from x0 = 3
  const auto ok = s.solve(
      [](T2 x) { return std::pair<T2, T2>{x * x - T2{4}, T2{2} * x}; }, T2{3});
  EXPECT_TRUE(ok.converged);
  EXPECT_NEAR(ok.x, 2.0, 1e-10);

  // No root: x^2 + 1 = 0. Must report failure rather than a plausible number.
  const auto bad = s.solve(
      [](T2 x) { return std::pair<T2, T2>{x * x + T2{1}, T2{2} * x}; }, T2{1});
  EXPECT_FALSE(bad.converged);
}
}  // namespace

namespace {
namespace nm_w = numsim::materials;

/// A wrongly-typed reference must say so, not report the material as missing.
///
/// wire_materials() wrapped look-up and type-check in one catch(...), so
/// swapping a solver for one of another type told the user their solver did
/// not exist -- sending them to look for a material that was sitting right
/// there in the document.
TEST(WireMaterials, WrongTypeIsNotReportedAsMissing) {
  using policy = nm_w::material_policy_default;
  using T2 = policy::value_type;
  nm_w::material_context<policy> ctx;
  policy::ParameterHandler p;

  p.insert<std::string>("name", "stepper");
  p.insert<T2>("increment", T2{0.01});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<nm_w::tensor_component_stepper<2, policy>>(p);

  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "j2");
  p.insert<T2>("K", T2{1000.0});
  ctx.create<nm_w::linear_isotropic_hardening<policy>>(p);

  // A backward_euler where a local_newton is required: it EXISTS.
  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<std::string>("function", "j2");
  ctx.create<nm_w::backward_euler<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "j2");
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<std::string>("solver_source", "solver");
  p.insert<T2>("K", T2{166.67});
  p.insert<T2>("G", T2{76.92});
  p.insert<T2>("sigma_0", T2{50.0});
  ctx.create<nm_w::j2_plasticity<policy>>(p);

  try {
    ctx.finalize();
    FAIL() << "wiring a wrongly-typed solver must fail";
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("wrong type"), std::string::npos) << msg;
    EXPECT_NE(msg.find("'solver'"), std::string::npos) << msg;
    EXPECT_EQ(msg.find("do not exist"), std::string::npos)
        << "the material exists; saying otherwise sends the user hunting for "
           "something that is right there: " << msg;
  }
}

/// The genuinely-absent case still reports absence.
/// A material whose construction throws must leave nothing behind.
///
/// material_base used to register itself in the material handler BEFORE
/// validating its parameters, and derived constructors can throw after that in
/// any case -- a missing required parameter, an unknown tableau name, any deck
/// typo. The handler stores std::ref and query_map has no erase, so the failed
/// object stayed reachable under a live name: wire_materials would find it and
/// dynamic_cast storage that had already been freed. Registration now happens
/// in object_store::create, after the constructor has returned.
///
/// Here "solver" fails to construct (backward_euler requires "function") and
/// is then referenced. The wiring must report it as absent. Before the fix it
/// resolved to the destroyed object and was reported as merely the wrong type.
TEST(WireMaterials, AFailedConstructionLeavesNoMaterialBehind) {
  using policy = nm_w::material_policy_default;
  using T2 = policy::value_type;
  nm_w::material_context<policy> ctx;
  policy::ParameterHandler p;

  p.insert<std::string>("name", "stepper");
  p.insert<T2>("increment", T2{0.01});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<nm_w::tensor_component_stepper<2, policy>>(p);

  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "j2");
  p.insert<T2>("K", T2{1000.0});
  ctx.create<nm_w::linear_isotropic_hardening<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "solver");
  EXPECT_ANY_THROW(ctx.create<nm_w::backward_euler<policy>>(p))
      << "backward_euler must reject a missing \"function\"";

  p.clear();
  p.insert<std::string>("name", "j2");
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<std::string>("solver_source", "solver");
  p.insert<T2>("K", T2{166.67});
  p.insert<T2>("G", T2{76.92});
  p.insert<T2>("sigma_0", T2{50.0});
  ctx.create<nm_w::j2_plasticity<policy>>(p);

  try {
    ctx.finalize();
    FAIL() << "a reference to a material that failed to construct must fail "
              "the wiring";
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("do not exist"), std::string::npos) << msg;
    EXPECT_EQ(msg.find("wrong type"), std::string::npos)
        << "the failed material is still registered: " << msg;
  }
}

TEST(WireMaterials, MissingIsStillReportedAsMissing) {
  using policy = nm_w::material_policy_default;
  using T2 = policy::value_type;
  nm_w::material_context<policy> ctx;
  policy::ParameterHandler p;

  p.insert<std::string>("name", "stepper");
  p.insert<T2>("increment", T2{0.01});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<nm_w::tensor_component_stepper<2, policy>>(p);
  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "j2");
  p.insert<T2>("K", T2{1000.0});
  ctx.create<nm_w::linear_isotropic_hardening<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "j2");
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<std::string>("solver_source", "no_such_solver");
  p.insert<T2>("K", T2{166.67});
  p.insert<T2>("G", T2{76.92});
  p.insert<T2>("sigma_0", T2{50.0});
  ctx.create<nm_w::j2_plasticity<policy>>(p);

  try {
    ctx.finalize();
    FAIL() << "a missing solver must fail";
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("do not exist"), std::string::npos) << msg;
    EXPECT_NE(msg.find("no_such_solver"), std::string::npos) << msg;
    EXPECT_EQ(msg.find("wrong type"), std::string::npos) << msg;
  }
}
}  // namespace
