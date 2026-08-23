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
  for (const char* name : {"j2_plasticity",
                           "j2_rk_plasticity", "linear_isotropic_hardening",
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
    {"type":"backward_euler","name":"solver"},
    {"type":"linear_isotropic_hardening","name":"hardening",
     "source":"j2","K":1000.0},
    {"type":"j2_plasticity","name":"j2","elastic_source":"elastic",
     "hardening_source":"hardening","strain_source":"stepper",
     "solver_source":"solver","G":76.92,"sigma_0":50.0}
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

/// drucker_prager_plasticity is deliberately NOT registered. Its yield function
/// carries eta, beta and K_bulk and arrives as a C++ object the JSON reader
/// cannot convert, so a document naming it would get a default-constructed one:
/// eta = beta = k = 0, which builds, runs, never yields, and is
/// indistinguishable from elasticity. An unknown type is a loud error; a
/// silently elastic Drucker-Prager is not.
TEST(MaterialRegistry, DruckerPragerStaysUnregisteredWhileUnconfigurable) {
  auto& f = factory_type::instance();
  EXPECT_FALSE(f.contains("drucker_prager_plasticity"))
      << "registered, a document could name it and silently get eta=beta=k=0; "
         "register it once yield_function is expressible in JSON";
}

}  // namespace
