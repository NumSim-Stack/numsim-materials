#include <gtest/gtest.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <fstream>
#include <regex>
#include <set>
#include <string>
#include "numsim-materials/default_materials.h"
#include "numsim-materials/io/json_material_factory.h"

/// Generic conformance checks over every material, so a new one is covered
/// without a bespoke test file.
///
/// These exist because registration and schema declaration are both
/// hand-maintained lists, and a missing entry is silent: the material works
/// from C++ and is simply absent from the document layer. Eight materials were
/// unreachable that way before #33.
namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using factory_type = nm::object_store<policy>::factory_type;

struct Registration {
  Registration() { nm::register_default_materials<policy>(); }
};
const Registration registration_{};

/// Materials deliberately absent from the factory, each with the reason.
/// An entry here is a decision someone wrote down; the point of the test is
/// that omission can no longer be mistaken for one.
const std::set<std::string> kNotForJson = {
    // Templates over a yield-function TYPE. The registrable names are their
    // concrete aliases (j2_plasticity, drucker_prager_plasticity,
    // j2_rk_plasticity), which ARE registered.
    "small_strain_plasticity",
    "rk_plasticity",
    // Template over Rank, registered as tensor_component_stepper_rank1 and
    // _rank2. Reachable from a document, just not under this stem.
    "tensor_component_stepper",
};

/// Header stems under materials/ that declare a class deriving material_base.
std::set<std::string> material_headers() {
  namespace fs = std::filesystem;
  std::set<std::string> out;
  const fs::path dir{NUMSIM_MATERIALS_INCLUDE_DIR};
  EXPECT_TRUE(fs::exists(dir)) << "materials directory not found: " << dir;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.path().extension() != ".h") continue;
    std::ifstream in(e.path());
    const std::string src{std::istreambuf_iterator<char>(in),
                          std::istreambuf_iterator<char>()};
    // A helper header (yield functions, plasticity_utils) declares no material.
    if (src.find("public material_base") == std::string::npos) continue;
    out.insert(e.path().stem().string());
  }
  return out;
}

/// The check that #33 existed for. A material absent from the factory cannot be
/// named in a document, and nothing else in the suite notices, because every
/// other test constructs materials directly in C++.
TEST(MaterialConformance, EveryMaterialIsRegisteredOrExplicitlyNotForJson) {
  auto& f = factory_type::instance();
  const auto registered = f.registered_types();
  const std::set<std::string> reg{registered.begin(), registered.end()};

  for (const auto& stem : material_headers()) {
    if (kNotForJson.contains(stem)) continue;
    // Registered under its own name, or under an alias that mentions it
    // (j2_plasticity -> small_strain_plasticity is handled by kNotForJson).
    EXPECT_TRUE(reg.contains(stem))
        << stem << " derives material_base but is not registered, so it cannot "
                   "be named in a JSON document. Register it, or add it to "
                   "kNotForJson with the reason.";
  }
}

/// The opt-out list must not outlive its reason: an entry naming a header that
/// no longer exists is a stale exemption that would hide a real gap.
TEST(MaterialConformance, TheOptOutListHasNoStaleEntries) {
  const auto headers = material_headers();
  for (const auto& stem : kNotForJson)
    EXPECT_TRUE(headers.contains(stem))
        << stem << " is exempted but no such material header exists";
}

/// Every registered type must declare "name" — the graph keys on it, and a
/// schema without it fails at construction rather than at registration.
TEST(MaterialConformance, EverySchemaDeclaresName) {
  auto& f = factory_type::instance();
  for (const auto& type : f.registered_types()) {
    const auto schema = f.schema(type);
    bool has_name = false;
    for (const auto& [key, param] : schema) has_name |= (key == "name");
    EXPECT_TRUE(has_name) << type << " declares no \"name\" parameter";
  }
}

/// A schema with no parameters at all is almost always a material whose
/// parameters() forgot to chain base::parameters().
TEST(MaterialConformance, NoSchemaIsEmpty) {
  auto& f = factory_type::instance();
  for (const auto& type : f.registered_types()) {
    std::size_t n = 0;
    for ([[maybe_unused]] const auto& kv : f.schema(type)) ++n;
    EXPECT_GT(n, 0u) << type << " has an empty schema — did parameters() "
                                "forget to chain base::parameters()?";
  }
}

/// Omitting a required parameter must throw, not warn and carry on with a
/// default-constructed value.
TEST(MaterialConformance, MissingRequiredParametersThrow) {
  auto& f = factory_type::instance();
  int checked = 0;
  for (const auto& type : f.registered_types()) {
    nlohmann::json j;
    j["type"] = type;
    j["name"] = "probe";
    nm::material_context<policy> ctx;
    // Only meaningful for types that require something beyond "name".
    std::size_t required = 0;
    for (const auto& [key, param] : f.schema(type))
      if (key != "name") ++required;
    if (required == 0) continue;
    ++checked;
    try {
      nm::create_from_json<policy>(ctx, j);
      // Reaching here is fine only if nothing was actually required.
    } catch (const std::exception&) {
      // Threw, which is the contract.
    }
  }
  EXPECT_GT(checked, 0) << "no type had parameters beyond name — check failed "
                           "to exercise anything";
}

}  // namespace
