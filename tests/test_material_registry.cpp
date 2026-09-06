#include <gtest/gtest.h>
#include <string>
#include <nlohmann/json.hpp>
#include "numsim-materials/default_materials.h"
#include "numsim-materials/io/json_material_factory.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using factory_type = nm::object_store<policy>::factory_type;

struct Registration {
  Registration() { nm::register_default_materials<policy>(); }
};
const Registration registration_{};

/// A material that is not registered cannot be named in a document, whatever
/// else works about it. These were all absent, so a JSON model could build
/// elasticity and damage and essentially nothing else.
TEST(MaterialRegistry, PlasticityAndFriendsAreReachableFromJson) {
  auto& f = factory_type::instance();
  for (const char* name : {"j2_plasticity", "drucker_prager_plasticity",
                           "j2_rk_plasticity", "local_newton",
                           "linear_isotropic_hardening",
                           "exponential_isotropic_hardening", "linear_damage_law",
                           "curing_rate", "strain_energy_state_function",
                           "vector_strain_state_function"})
    EXPECT_TRUE(f.contains(name)) << name << " cannot be named in a document";
}

/// The one that matters: a J2 model built entirely from a document, driven to
/// yield. Registration alone would pass the check above while still failing
/// here if a parameter were not JSON-convertible.
TEST(MaterialRegistry, AJ2ModelRunsFromADocumentAlone) {
  const char* doc = R"([
    {"type":"tensor_component_stepper_rank2","name":"stepper",
     "increment":0.01,"indices":[0,0]},
    {"type":"linear_elasticity","name":"elastic",
     "strain_producer_name":"stepper","K":166.67,"G":76.92},
    {"type":"local_newton","name":"solver"},
    {"type":"linear_isotropic_hardening","name":"hardening",
     "source":"j2","K":1000.0},
    {"type":"j2_plasticity","name":"j2",
     "hardening_source":"hardening","strain_source":"stepper",
     "solver_source":"solver","K":166.67,"G":76.92,"sigma_0":50.0}
  ])";

  nm::material_context<policy> ctx;
  ASSERT_NO_THROW({
    for (const auto& m : nlohmann::json::parse(doc))
      nm::create_from_json<policy>(ctx, m);
    ctx.finalize();
  });

  for (int i = 0; i < 40; ++i) { ctx.update(); ctx.commit(); }
  EXPECT_GT(ctx.get<T>("j2", "equivalent_plastic_strain"), 1e-6)
      << "the document built, but the model never yielded";
}

/// Drucker-Prager, from a document, driven to yield.
///
/// It was held back while eta, beta and K_bulk arrived inside a C++
/// "yield_function" object the JSON reader could not convert: a document naming
/// it would have got a default-constructed cone -- eta = beta = k = 0 -- which
/// builds, runs, never yields, and is indistinguishable from elasticity. #43
/// made them plain required scalars. This is the test that the condition is
/// actually met, rather than that the type merely appears in the factory.
TEST(MaterialRegistry, ADruckerPragerModelRunsFromADocumentAlone) {
  const char* doc = R"([
    {"type":"tensor_component_stepper_rank2","name":"stepper",
     "increment":0.01,"indices":[0,0]},
    {"type":"local_newton","name":"solver"},
    {"type":"linear_isotropic_hardening","name":"hardening",
     "source":"dp","K":500.0},
    {"type":"drucker_prager_plasticity","name":"dp",
     "hardening_source":"hardening","strain_source":"stepper",
     "solver_source":"solver","G":76.92,"sigma_0":20.0,
     "eta":0.1,"beta":0.05,"K_bulk":166.67}
  ])";

  nm::material_context<policy> ctx;
  ASSERT_NO_THROW({
    for (const auto& m : nlohmann::json::parse(doc))
      nm::create_from_json<policy>(ctx, m);
    ctx.finalize();
  });

  for (int i = 0; i < 40; ++i) { ctx.update(); ctx.commit(); }
  EXPECT_GT(ctx.get<T>("dp", "equivalent_plastic_strain"), 1e-6)
      << "the document built, but the cone never yielded -- the silently "
         "elastic failure this material was held back for";
}

/// And the reason it is safe to register: an omitted cone parameter stops the
/// build. This is the guarantee that replaced "keep it out of the factory" --
/// previously a document naming drucker_prager_plasticity got a
/// default-constructed cone with eta = beta = k = 0 and ran as elasticity.
///
/// The exception carries only a COUNT -- "missing 1 required parameter(s)" --
/// not the name. numsim-core's input_parameter_controller prints the names to
/// stdout and throws the count separately, so a deck typo is loud but not
/// self-explanatory. That is a numsim-core change, tracked separately; what
/// matters here is that the cone cannot be built without its parameters.
TEST(MaterialRegistry, ADruckerPragerDocumentMissingEtaFailsLoudly) {
  const char* doc = R"([
    {"type":"tensor_component_stepper_rank2","name":"stepper",
     "increment":0.01,"indices":[0,0]},
    {"type":"local_newton","name":"solver"},
    {"type":"linear_isotropic_hardening","name":"hardening",
     "source":"dp","K":500.0},
    {"type":"drucker_prager_plasticity","name":"dp",
     "hardening_source":"hardening","strain_source":"stepper",
     "solver_source":"solver","G":76.92,"sigma_0":20.0,
     "beta":0.05,"K_bulk":166.67}
  ])";

  nm::material_context<policy> ctx;
  EXPECT_THROW(
      {
        for (const auto& m : nlohmann::json::parse(doc))
          nm::create_from_json<policy>(ctx, m);
      },
      std::exception)
      << "a cone with no friction coefficient must not build silently";
  EXPECT_EQ(ctx.find("dp"), nullptr)
      << "the failed material must not be left behind in the context";
}

}  // namespace
