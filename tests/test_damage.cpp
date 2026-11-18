#include <gtest/gtest.h>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/von_mises_state_function.h"
#include "numsim-materials/materials/strain_threshold_yield.h"
#include "numsim-materials/materials/exponential_damage_law.h"
#include "numsim-materials/materials/isotropic_damage.h"
#include "numsim-materials/materials/strain_energy_state_function.h"
#include "numsim-materials/materials/vector_strain_state_function.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/json_model.h"
#include "numsim-materials/postprocessing/numerical_diff_checker.h"

namespace {

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

T relative(tensor4 const &a, tensor4 const &b) { return tmech::norm(tensor4{a - b}) / tmech::norm(b); }

class IsotropicDamageTest : public ::testing::Test {
protected:
  void SetUp() override {
    param_type p;

    p.clear();
    p.insert<std::string>("name", "stepper");
    p.insert<T>("increment", T{0.001});
    p.insert<std::vector<std::size_t>>("indices", {0, 0});
    ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

    p.clear();
    p.insert<std::string>("name", "state_func");
    p.insert<std::string>("strain_source", "stepper");
    ctx.create<numsim::materials::von_mises_state_function<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "yield");
    p.insert<std::string>("state_source", "state_func");
    p.insert<T>("kappa_critical", T{0.003});
    ctx.create<numsim::materials::strain_threshold_yield<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "damage_law");
    p.insert<std::string>("yield_source", "yield");
    p.insert<T>("kappa_0", T{0.003});
    p.insert<T>("beta", T{100});
    ctx.create<numsim::materials::exponential_damage_law<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "stepper");
    p.insert<T>("K", T{166.67});
    p.insert<T>("G", T{76.92});
    ctx.create<numsim::materials::linear_elasticity<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "damaged");
    p.insert<std::string>("elastic_source", "elastic");
    p.insert<std::string>("damage_source", "damage_law");
    p.insert<std::string>("state_source", "state_func");
    p.insert<std::string>("yield_source", "yield");
    ctx.create<numsim::materials::isotropic_damage<policy>>(p);

    ctx.finalize();
  }

  ctx_type ctx;
};

TEST_F(IsotropicDamageTest, ElasticBeforeYield) {
  // First steps should be elastic (D = 0)
  for (int i = 0; i < 3; ++i) {
    ctx.update();
    auto D = ctx.get<T>("damage_law", "damage");
    EXPECT_NEAR(D, 0.0, 1e-12) << "Step " << i << " should be elastic";
    ctx.commit();
  }
}

TEST_F(IsotropicDamageTest, DamageGrowsAfterYield) {
  T prev_D = 0;
  bool found_damage = false;
  for (int i = 0; i < 15; ++i) {
    ctx.update();
    auto D = ctx.get<T>("damage_law", "damage");
    if (D > 0) {
      found_damage = true;
      EXPECT_GE(D, prev_D) << "Damage must be monotonically increasing at step " << i;
      prev_D = D;
    }
    ctx.commit();
  }
  EXPECT_TRUE(found_damage) << "Damage should start growing within 15 steps";
  EXPECT_GT(prev_D, 0.3) << "Damage should reach significant level";
}

TEST_F(IsotropicDamageTest, StressSoftening) {
  T peak_stress = 0;
  T final_stress = 0;
  for (int i = 0; i < 20; ++i) {
    ctx.update();
    const auto& sig = ctx.get<tensor2>("damaged", "damaged_stress");
    auto s = std::abs(sig(0,0));
    if (s > peak_stress) peak_stress = s;
    final_stress = s;
    ctx.commit();
  }
  EXPECT_GT(peak_stress, final_stress)
      << "Stress should soften after peak (peak=" << peak_stress
      << ", final=" << final_stress << ")";
}

// the derivative accumulated from the partials of the chain is the hand-written damaged tangent
TEST_F(IsotropicDamageTest, AccumulatedTangentIsTheDamagedTangent) {
  auto const &derivative{ctx.derivative<tensor4>("damaged", "damaged_stress", "stepper", "strain")};
  int damaging{0};
  for (int i = 0; i < 20; ++i) {
    ctx.update();
    damaging += ctx.get<int>("yield", "is_yielding");
    EXPECT_LE(relative(derivative, ctx.get<tensor4>("damaged", "damaged_tangent")), 1e-12) << "step " << i;
    ctx.commit();
  }
  EXPECT_GT(damaging, 10);
}

/// The damage chain over each state function, loaded, unloaded and reloaded past its previous maximum.
class DamageChain : public ::testing::TestWithParam<const char *> {};

TEST_P(DamageChain, AccumulatedTangentIsTheDamagedTangentLoadingAndUnloading) {
  ctx_type ctx;
  numsim::materials::umat::make_json_builder<policy>(std::string{R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "linear_elasticity", "name": "elastic", "strain_producer_name": "strain", "K": 166.67, "G": 76.92},
    )"} + GetParam() + R"(,
    {"type": "strain_threshold_yield", "name": "yield", "state_source": "state", "kappa_critical": 0.002},
    {"type": "exponential_damage_law", "name": "law", "yield_source": "yield", "kappa_0": 0.002, "beta": 300.0},
    {"type": "isotropic_damage", "name": "damaged", "elastic_source": "elastic", "damage_source": "law", "state_source": "state", "yield_source": "yield"}]})")(ctx, {});
  auto const &derivative{ctx.derivative<tensor4>("damaged", "damaged_stress", "strain", "strain")};
  auto &source{*dynamic_cast<numsim::materials::external_strain_source<policy> *>(ctx.find("strain"))};
  tensor2 direction;
  direction(0, 0) = 1, direction(1, 1) = -0.3, direction(0, 1) = direction(1, 0) = 0.4;
  T previous{0};
  int damaging{0}, unloading{0}, differenced{0};
  for (T const load : {0.001, 0.003, 0.005, 0.007, 0.005, 0.002, -0.001, 0.004, 0.008, 0.01}) {
    // the damaged stress at the committed history, without committing
    auto const stress_at{[&](T value) {
      source.bind(tensor2{previous * direction}, tensor2{value * direction});
      ctx.update();
      return tensor2{ctx.get<tensor2>("damaged", "damaged_stress")};
    }};
    T constexpr h{1e-7};
    tensor2 const forward{(stress_at(load + h) - stress_at(load)) / h}, backward{(stress_at(load) - stress_at(load - h)) / h};
    tensor2 const fd{(stress_at(load + h) - stress_at(load - h)) / (2 * h)};
    stress_at(load);
    // at the yield threshold the one-sided slopes differ: no central difference there
    bool const smooth{tmech::norm(tensor2{forward - backward}) <= 1e-4 * tmech::norm(fd) + 1e-8};
    bool const yielding{ctx.get<int>("yield", "is_yielding") != 0};
    damaging += yielding;
    unloading += !yielding && ctx.get<T>("law", "damage") > 0;
    EXPECT_LE(relative(derivative, ctx.get<tensor4>("damaged", "damaged_tangent")), 1e-12) << "load " << load;
    // independent of the hand-written tangent: central differences of the stress along the load direction
    tensor2 const along{tmech::dcontract(derivative, direction)};
    if (smooth) {
      EXPECT_LE(tmech::norm(tensor2{along - fd}), 1e-5 * tmech::norm(fd) + 1e-8) << "load " << load;
      ++differenced;
    }
    ctx.commit();
    previous = load;
  }
  EXPECT_GT(damaging, 2);
  EXPECT_GT(unloading, 2);
  EXPECT_GT(differenced, 7);
}

INSTANTIATE_TEST_SUITE_P(
    StateFunctions, DamageChain,
    ::testing::Values(R"({"type": "von_mises_state_function", "name": "state", "strain_source": "strain"})",
                      R"({"type": "strain_energy_state_function", "name": "state", "strain_source": "strain", "tangent_source": "elastic", "youngs_modulus": 200.0})",
                      R"({"type": "vector_strain_state_function", "name": "state", "strain_source": "strain", "m_direction": [0], "n_direction": [1]})"));

// --- Tangent checker ---

class TangentCheckerTest : public ::testing::Test {
protected:
  void SetUp() override {
    param_type p;

    p.clear();
    p.insert<std::string>("name", "stepper");
    p.insert<T>("increment", T{0.001});
    p.insert<std::vector<std::size_t>>("indices", {0, 0});
    ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

    p.clear();
    p.insert<std::string>("name", "state_func");
    p.insert<std::string>("strain_source", "stepper");
    ctx.create<numsim::materials::von_mises_state_function<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "yield");
    p.insert<std::string>("state_source", "state_func");
    p.insert<T>("kappa_critical", T{0.003});
    ctx.create<numsim::materials::strain_threshold_yield<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "damage_law");
    p.insert<std::string>("yield_source", "yield");
    p.insert<T>("kappa_0", T{0.003});
    p.insert<T>("beta", T{100});
    ctx.create<numsim::materials::exponential_damage_law<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "stepper");
    p.insert<T>("K", T{166.67});
    p.insert<T>("G", T{76.92});
    ctx.create<numsim::materials::linear_elasticity<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "damaged");
    p.insert<std::string>("elastic_source", "elastic");
    p.insert<std::string>("damage_source", "damage_law");
    p.insert<std::string>("state_source", "state_func");
    p.insert<std::string>("yield_source", "yield");
    ctx.create<numsim::materials::isotropic_damage<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "checker");
    p.insert<ctx_type*>("context", &ctx);
    p.insert<std::string>("output_source", "damaged::damaged_stress");
    p.insert<std::string>("input_source", "stepper::strain");
    p.insert<std::string>("analytical_source", "damaged::damaged_tangent");
    p.insert<std::vector<std::string>>("history_sources", {"yield::kappa"});
    p.insert<T>("epsilon", T{1e-7});
    ctx.create<numsim::materials::tangent_checker<policy>>(p);

    ctx.finalize();
  }

  ctx_type ctx;
};

TEST_F(TangentCheckerTest, ExclusionWorks) {
  // Verify the exclude mechanism prevents the stepper from running
  ctx.update();
  auto eps_before = ctx.get<tensor2>("stepper", "strain")(0,0);

  auto* stepper_prop = ctx.find_property("stepper", "strain");
  ASSERT_NE(stepper_prop, nullptr);

  std::unordered_set<const numsim::materials::property_base*> exclude{stepper_prop};
  ctx.update_property("damaged", "damaged_stress", exclude);

  auto eps_after = ctx.get<tensor2>("stepper", "strain")(0,0);
  EXPECT_NEAR(eps_before, eps_after, 1e-15)
      << "Stepper should NOT have run when excluded";
}

// TODO: Passes in standalone test_material binary but fails in GTest.
// Elastic steps match (1e-12), damage steps show ~100% error.
// The get()/get_mutable() history fix didn't resolve it.
// TODO: tmech::num_diff_sym_central produces transposed results for the
// damage tangent in this GTest binary but works correctly in test_material.
// Same source, same compiler, same tmech headers. Needs tmech-level debugging.
// Non-symmetric num_diff_central produces correct results for all steps.
TEST_F(TangentCheckerTest, MachinePrecisionAllSteps) {
  T max_rel_error = 0;
  for (int i = 0; i < 15; ++i) {
    ctx.update();
    auto rel = ctx.get<T>("checker", "rel_error");
    if (rel > max_rel_error) max_rel_error = rel;
    ctx.commit();
  }
  EXPECT_LT(max_rel_error, 1e-8)
      << "Tangent should match numerical derivative to machine precision";
}

TEST_F(TangentCheckerTest, ElasticStepsMachinePrecision) {
  for (int i = 0; i < 3; ++i) {
    ctx.update();
    auto rel = ctx.get<T>("checker", "rel_error");
    EXPECT_LT(rel, 1e-8) << "Elastic step " << i << " should match";
    ctx.commit();
  }
}

} // namespace
