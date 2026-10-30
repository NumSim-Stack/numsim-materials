#include <gtest/gtest.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <format>
#include <fstream>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <vector>
#include "numsim-materials/core/partial_check.h"
#include "numsim-materials/default_materials.h"
#include "numsim-materials/io/json_material_factory.h"
#include "numsim-materials/umat/external_state_source.h"

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
    // Template over Rank, registered as tensor_component_stepper_rank1 and
    // _rank2. Reachable from a document, just not under this stem.
    "tensor_component_stepper",
};

/// Header stems under one directory that declare a class deriving material_base.
std::set<std::string> material_headers_in(const char* directory) {
  namespace fs = std::filesystem;
  std::set<std::string> out;
  const fs::path dir{directory};
  EXPECT_TRUE(fs::exists(dir)) << "directory not found: " << dir;
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

/// Everything that derives material_base: materials/ and solvers/ (solvers are created and named the same way).
std::set<std::string> material_headers() {
  auto out{material_headers_in(NUMSIM_MATERIALS_INCLUDE_DIR)};
  out.merge(material_headers_in(NUMSIM_SOLVERS_INCLUDE_DIR));
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
    // Every plasticity class is now its own header and its own factory entry.
    EXPECT_TRUE(reg.contains(stem))
        << stem << " derives material_base but is not registered, so it cannot "
                   "be named in a JSON document. Register it, or add it to "
                   "kNotForJson with the reason.";
  }
}

// pins the coverage of the scan, not the current outcome: with every solver registered the test above passes
// either way, which is how rk_integrator shipped unregistered
TEST(MaterialConformance, TheScanCoversSolversAsWellAsMaterials) {
  const auto headers = material_headers();
  for (const char* solver : {"local_newton", "vector_newton", "rk_integrator",
                             "backward_euler"})
    EXPECT_TRUE(headers.contains(solver))
        << solver << " derives material_base but the conformance scan does "
                     "not see it -- is solvers/ still being scanned?";
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

/// Omitting a required parameter must throw, not carry on with a
/// default-constructed value.
///
/// NOT generic, deliberately. is_required is a check attached with
/// .add<is_required>() and the controller exposes no way to ask whether a
/// parameter is required, so a loop over every type cannot tell "threw because
/// something was required" from "succeeded because nothing was". The first
/// version of this test tried anyway and asserted nothing at all: it caught the
/// exception, ignored both outcomes, and checked only that the loop had run.
///
/// So the types are named. If the controller ever exposes required-ness, this
/// becomes a loop over registered_types() and the list goes.
TEST(MaterialConformance, MissingRequiredParametersThrow) {
  auto& f = factory_type::instance();
  for (const char* type : {"linear_elasticity",   // needs K, G
                           "isotropic_tangent",   // needs K_source, G_source
                           "linear_stress",       // needs tangent_source, strain_source
                           "props_scalar",        // needs index
                           "linear_damage_law"}) {  // needs yield_source, kappa_0, kappa_f
    ASSERT_TRUE(f.contains(type)) << type << " is not registered";
    nlohmann::json j;
    j["type"] = type;
    j["name"] = "probe";
    nm::material_context<policy> ctx;
    EXPECT_THROW(nm::create_from_json<policy>(ctx, j), std::exception)
        << type << " accepted a document with only \"name\": a required "
                   "parameter was silently defaulted";
  }
}

/// Materials whose callbacks declare partials or independence (.partials(), .independent_of()), checked by finite differences below.
const std::set<std::string> kPartialsChecked = {
    "exponential_damage_law", "incremental_linear_elasticity", "isotropic_damage", "isotropic_moduli",
    "isotropic_tangent", "linear_damage_law", "linear_elasticity", "linear_stress", "peerlings_damage_law",
    "power_damage_law", "saturation_damage_law", "strain_energy_state_function", "strain_threshold_yield",
    "thermal_eigenstrain", "vector_strain_state_function", "von_mises_state_function"};

/// Header stems under materials/ and solvers/ that declare partials or independent reads, themselves or through a
/// detail/ base they include.
std::set<std::string> headers_declaring_partials() {
  namespace fs = std::filesystem;
  auto const source{[](fs::path const& file) {
    std::ifstream in(file);
    return std::string{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  }};
  auto const declares{[](std::string const& src) {
    return src.find(".partials(") != std::string::npos || src.find(".independent_of(") != std::string::npos;
  }};
  std::vector<std::string> bases;
  for (const auto& e : fs::directory_iterator(fs::path{NUMSIM_MATERIALS_INCLUDE_DIR} / "detail"))
    if (e.path().extension() == ".h" && declares(source(e.path())))
      bases.push_back("detail/" + e.path().filename().string());
  std::set<std::string> out;
  for (const char* directory : {NUMSIM_MATERIALS_INCLUDE_DIR, NUMSIM_SOLVERS_INCLUDE_DIR})
    for (const auto& e : fs::directory_iterator(fs::path{directory})) {
      if (e.path().extension() != ".h") continue;
      const std::string src{source(e.path())};
      if (declares(src) || std::ranges::any_of(bases, [&](auto const& base) { return src.contains(base); }))
        out.insert(e.path().stem().string());
    }
  return out;
}

TEST(MaterialConformance, EveryMaterialDeclaringPartialsIsChecked) {
  EXPECT_EQ(headers_declaring_partials(), kPartialsChecked)
      << "a material declaring partials or independent reads joins kPartialsChecked and the model of DeclaredPartialsMatchFiniteDifferences";
}

/// The model of every material declaring partials; isotropic_damage runs over `damage_law`, which names one of the laws.
/// `kappa_f` is the end of the linear law.
std::unique_ptr<nm::material_context<policy>> partials_model(std::string const& damage_law, double kappa_f = 0.5) {
  auto owner{std::make_unique<nm::material_context<policy>>()};
  auto& ctx{*owner};
  for (const auto* source : {"strain", "temperature"}) {
    policy::ParameterHandler p;
    p.insert<std::string>("name", source);
    if (std::string{source} == "strain")
      ctx.create<nm::external_strain_source<policy>>(p);
    else
      ctx.create<nm::external_scalar_source<policy>>(p);
  }
  // strains of 1e-1 keep the differences of the nonlinear state functions accurate; the moduli skip the isotropy
  // check, which a perturbed tangent fails
  for (const std::string& document : std::vector<std::string>{
           R"({"type": "constant_scalar", "name": "alpha", "value": 1.2e-5})",
           R"({"type": "constant_scalar", "name": "K", "value": 150.0})",
           R"({"type": "constant_scalar", "name": "G", "value": 70.0})",
           R"({"type": "thermal_eigenstrain", "name": "thermal_eigenstrain", "alpha_source": "alpha::value", "reference_temperature": 20.0})",
           R"({"type": "linear_elasticity", "name": "linear_elasticity", "strain_producer_name": "strain", "eigenstrain_source": "thermal_eigenstrain", "K": 150.0, "G": 70.0})",
           R"({"type": "isotropic_tangent", "name": "isotropic_tangent", "K_source": "K", "G_source": "G"})",
           R"({"type": "isotropic_moduli", "name": "isotropic_moduli", "tangent_source": "isotropic_tangent", "tolerance": 0.0})",
           R"({"type": "linear_stress", "name": "linear_stress", "tangent_source": "isotropic_tangent", "strain_source": "strain"})",
           R"({"type": "incremental_linear_elasticity", "name": "incremental_linear_elasticity", "strain_source": "strain", "tangent_source": "isotropic_tangent"})",
           R"({"type": "von_mises_state_function", "name": "von_mises_state_function", "strain_source": "strain"})",
           R"({"type": "strain_energy_state_function", "name": "strain_energy_state_function", "strain_source": "strain", "tangent_source": "isotropic_tangent", "youngs_modulus": 180.0})",
           R"({"type": "vector_strain_state_function", "name": "vector_strain_state_function", "strain_source": "strain", "m_direction": [0], "n_direction": [1]})",
           R"({"type": "strain_threshold_yield", "name": "strain_threshold_yield", "state_source": "von_mises_state_function", "kappa_critical": 0.02})",
           R"({"type": "exponential_damage_law", "name": "exponential_damage_law", "yield_source": "strain_threshold_yield", "kappa_0": 0.02, "beta": 10.0})",
           std::format(R"({{"type": "linear_damage_law", "name": "linear_damage_law", "yield_source": "strain_threshold_yield", "kappa_0": 0.02, "kappa_f": {}}})", kappa_f),
           R"({"type": "power_damage_law", "name": "power_damage_law", "yield_source": "strain_threshold_yield", "K": 5.0, "m": 1.5, "kappa_0": 0.02})",
           R"({"type": "saturation_damage_law", "name": "saturation_damage_law", "yield_source": "strain_threshold_yield", "K": 0.9, "m": 20.0, "kappa_0": 0.02})",
           R"({"type": "peerlings_damage_law", "name": "peerlings_damage_law", "yield_source": "strain_threshold_yield", "kappa_0": 0.02, "alpha": 0.95, "beta": 10.0})"}) {
    const nlohmann::json j = nlohmann::json::parse(document);
    nm::create_from_json<policy>(ctx, j);
  }
  nm::create_from_json<policy>(ctx, nlohmann::json::parse(
      R"({"type": "isotropic_damage", "name": "isotropic_damage", "elastic_source": "linear_elasticity", "damage_source": ")" +
      damage_law + R"(", "state_source": "von_mises_state_function", "yield_source": "strain_threshold_yield"})"));
  ctx.finalize();
  return owner;
}

void bind_state(nm::material_context<policy>& ctx, tmech::tensor<double, 3, 2> const& previous,
                tmech::tensor<double, 3, 2> const& strain, double temperature) {
  dynamic_cast<nm::external_strain_source<policy>*>(ctx.find("strain"))->bind(previous, strain);
  dynamic_cast<nm::external_scalar_source<policy>*>(ctx.find("temperature"))->bind(20, temperature);
  ctx.update();
}

/// Every material with partials against central differences at the state the context holds.
void expect_partials_match(nm::material_context<policy>& ctx) {
  std::set<std::string> checked;
  for (const auto* material : ctx.materials())
    if (!material->partials().empty()) {
      checked.insert(material->name());
      for (const auto& mismatch : nm::check_partials(ctx, material->name()))
        ADD_FAILURE() << mismatch;
    }
  EXPECT_EQ(checked, kPartialsChecked);
}

tmech::tensor<double, 3, 2> const off_symmetry_strain{[] {
  tmech::tensor<double, 3, 2> strain;
  strain(0, 0) = 0.1, strain(1, 1) = -0.04, strain(2, 2) = 0.02;
  strain(0, 1) = strain(1, 0) = 0.03, strain(0, 2) = strain(2, 0) = -0.01, strain(1, 2) = strain(2, 1) = 0.05;
  return strain;
}()};

TEST(MaterialConformance, DeclaredPartialsMatchFiniteDifferences) {
  auto const owner{partials_model("exponential_damage_law")};
  bind_state(*owner, {}, off_symmetry_strain, 75);
  expect_partials_match(*owner);
}

/// Committed at the strain, then loaded back to 60 %: the equivalent strain is below kappa, the damage stays, the yield
/// node declares zero.
TEST(MaterialConformance, DeclaredPartialsMatchFiniteDifferencesWhileUnloading) {
  auto const owner{partials_model("exponential_damage_law")};
  bind_state(*owner, {}, off_symmetry_strain, 75);
  owner->commit();
  bind_state(*owner, off_symmetry_strain, tmech::tensor<double, 3, 2>{0.6 * off_symmetry_strain}, 75);
  ASSERT_EQ(owner->get<int>("strain_threshold_yield", "is_yielding"), 0);
  ASSERT_GT(owner->get<double>("exponential_damage_law", "damage"), 0.1);
  ASSERT_GT(owner->get<double>("strain_threshold_yield", "kappa"), owner->get<double>("von_mises_state_function", "equivalent_strain"));
  expect_partials_match(*owner);
}

/// The linear law reaches the cap d_max = 0.999 at kappa_0 + 0.999 (kappa_f - kappa_0); kappa_f is chosen so that the
/// equivalent strain of the state lands just below the cap (the slope is still 1 / (kappa_f - kappa_0)) and just above it (the
/// slope is zero).
TEST(MaterialConformance, DeclaredPartialsMatchFiniteDifferencesAtTheDamageCap) {
  auto const probe{partials_model("linear_damage_law")};
  bind_state(*probe, {}, off_symmetry_strain, 75);
  double const kappa{probe->get<double>("von_mises_state_function", "equivalent_strain")};
  for (double const margin : {1e-5, -1e-3}) {
    double const kappa_f{0.02 + (kappa + margin - 0.02) / 0.999};
    auto const owner{partials_model("linear_damage_law", kappa_f)};
    bind_state(*owner, {}, off_symmetry_strain, 75);
    bool const capped{margin < 0};
    ASSERT_EQ(owner->get<double>("linear_damage_law", "d_damage") == 0, capped) << margin;
    ASSERT_GT(owner->get<double>("linear_damage_law", "damage"), 0.9985) << margin;
    expect_partials_match(*owner);
  }
}

}  // namespace
