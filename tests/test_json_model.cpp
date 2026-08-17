#include <gtest/gtest.h>
#include <cstddef>
#include <string>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/umat/json_model.h"

// The Fortran-callable symbol, so the JSON path is exercised through the real
// ABI rather than only through the C++ evaluator.
NUMSIM_MATERIALS_DEFINE_UMAT(numsim::materials::material_policy_default)

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;

using policy = nm::material_policy_default;
using T = policy::value_type;
using registry = u::umat_registry<policy>;

/// Moduli bound to the deck's constants; nothing here is compiled.
const char* kElastic = R"({
  "materials": [
    {"type": "external_strain_source", "name": "strain_in"},
    {"type": "constant_scalar", "name": "K", "value": 0},
    {"type": "constant_scalar", "name": "G", "value": 0},
    {"type": "isotropic_tangent", "name": "stiffness",
     "K_source": "K", "G_source": "G"},
    {"type": "linear_stress", "name": "elastic",
     "tangent_source": "stiffness", "strain_source": "strain_in"}
  ],
  "constants": ["K::value", "G::value"]
})";

registry::config elastic_config() {
  registry::config cfg;
  cfg.strain_source = "strain_in";
  cfg.stress_source = "elastic";
  cfg.tangent_source = "stiffness";
  return cfg;
}

struct fortran_name {
  char buf[80];
  explicit fortran_name(const std::string& s) {
    for (auto& c : buf) c = ' ';
    for (std::size_t i = 0; i < s.size() && i < 80; ++i) buf[i] = s[i];
  }
};

/// DDSDDE(1,1) for a uniaxial increment, through the real umat_ entry point.
T uniaxial_tangent(const std::string& name, const T* props, int nprops) {
  const fortran_name cm(name);
  T statev[1] = {0};
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  T stress[6] = {0}, ddsdde[36] = {0}, pnewdt = 1.0;
  T sse = 0, spd = 0, scd = 0, rpl = 0, ddsddt[6] = {0}, drplde[6] = {0};
  T drpldt = 0;
  const T time[2] = {0, 0};
  T dtime = 0.1;
  const T temp = 0, dtemp = 0, predef = 0, dpred = 0, celent = 1;
  const T coords[3] = {0}, drot[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const T dfg[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  int noel = 1, npt = 1, layer = 1, kspt = 1, jstep = 1, kinc = 1;
  int ndi = 3, nshr = 3, ntens = 6, nstatv = 0;

  umat_(stress, statev, ddsdde, &sse, &spd, &scd, &rpl, ddsddt, drplde, &drpldt,
        stran, dstran, time, &dtime, &temp, &dtemp, &predef, &dpred, cm.buf,
        &ndi, &nshr, &ntens, &nstatv, props, &nprops, coords, drot, &pnewdt,
        &celent, dfg, dfg, &noel, &npt, &layer, &kspt, &jstep, &kinc, 80);

  EXPECT_DOUBLE_EQ(pnewdt, 1.0);
  return ddsdde[0];
}

struct Registration {
  Registration() {
    // One registered document, two deck materials — which is how a real deck
    // expresses two parameter sets: distinct *MATERIAL names.
    u::register_json_model<policy>("JSONSOFT", kElastic, elastic_config());
    u::register_json_model<policy>("JSONSTIFF", kElastic, elastic_config());
    u::register_json_model<policy>("JSONELASTIC", kElastic, elastic_config());
  }
};
const Registration registration_{};

// ---------------------------------------------------------------------------

/// The point: the model is a document and the constants come from the deck,
/// with no recompile either way.
TEST(JsonModel, DeckConstantsDriveAModelDefinedEntirelyInJson) {
  const T soft[2] = {100.0, 40.0};
  const T stiff[2] = {300.0, 140.0};

  EXPECT_NEAR(uniaxial_tangent("JSONSOFT", soft, 2),
              100.0 + 4.0 * 40.0 / 3.0, 1e-9);
  EXPECT_NEAR(uniaxial_tangent("JSONSTIFF", stiff, 2),
              300.0 + 4.0 * 140.0 / 3.0, 1e-9);
}

/// Values in the document are placeholders for anything listed in "constants"
/// — the deck wins.
TEST(JsonModel, DocumentValuesArePlaceholdersForBoundConstants) {
  // The document says 0 for both; if substitution failed the tangent would be
  // zero rather than wrong-but-plausible.
  const T props[2] = {250.0, 90.0};
  EXPECT_NEAR(uniaxial_tangent("JSONELASTIC", props, 2),
              250.0 + 4.0 * 90.0 / 3.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Validation, at registration rather than mid-analysis
// ---------------------------------------------------------------------------

TEST(JsonModel, RejectsAMalformedDocument) {
  EXPECT_THROW(u::make_json_builder<policy>("{not json"), u::fatal_error);
  EXPECT_THROW(u::make_json_builder<policy>(R"({"nope": 1})"), u::fatal_error);
}

/// The old "props" spelling would be accepted with every constant unbound,
/// leaving the placeholders as the moduli — wrong, plausible, silent.
TEST(JsonModel, RejectsAnUnrecognisedTopLevelKey) {
  const char* old_spelling = R"({
    "materials": [{"type": "constant_scalar", "name": "K", "value": 0}],
    "props": ["K::value"]
  })";
  EXPECT_THROW(u::make_json_builder<policy>(old_spelling), u::fatal_error);

  const char* typo = R"({
    "materials": [{"type": "constant_scalar", "name": "K", "value": 0}],
    "constant": ["K::value"]
  })";
  EXPECT_THROW(u::make_json_builder<policy>(typo), u::fatal_error);
}

TEST(JsonModel, RejectsAConstantsEntryThatIsNotQualified) {
  const char* doc = R"({
    "materials": [{"type": "constant_scalar", "name": "K", "value": 0}],
    "constants": ["Kvalue"]
  })";
  EXPECT_THROW(u::make_json_builder<policy>(doc), u::fatal_error);
}

/// The other half of the target. json CREATES a missing key rather than
/// failing, so a misspelled parameter went where nothing reads it while the
/// real one kept its placeholder: 7 + 4(90)/3 = 127 instead of 370, behind
/// nothing louder than a stderr warning.
TEST(JsonModel, RejectsAConstantsEntryNamingAnUndeclaredParameter) {
  const char* doc = R"({
    "materials": [
      {"type": "constant_scalar", "name": "K", "value": 7.0}
    ],
    "constants": ["K::vlaue"]
  })";
  EXPECT_THROW(u::make_json_builder<policy>(doc), u::fatal_error);

  // The correct spelling still registers.
  const char* good = R"({
    "materials": [
      {"type": "constant_scalar", "name": "K", "value": 7.0}
    ],
    "constants": ["K::value"]
  })";
  EXPECT_NO_THROW(u::make_json_builder<policy>(good));
}

/// Declared is not enough: every material declares "name", and most declare
/// *_source strings. Binding a host constant to one of those can only fail
/// later, with a JSON type error that says nothing about decks.
TEST(JsonModel, RejectsAConstantsEntryNamingANonNumericParameter) {
  const char* to_name = R"({
    "materials": [{"type": "constant_scalar", "name": "K", "value": 0}],
    "constants": ["K::name"]
  })";
  EXPECT_THROW(u::make_json_builder<policy>(to_name), u::fatal_error);

  const char* to_source = R"({
    "materials": [
      {"type": "constant_scalar", "name": "K", "value": 0},
      {"type": "constant_scalar", "name": "G", "value": 0},
      {"type": "isotropic_tangent", "name": "stiffness",
       "K_source": "K", "G_source": "G"}
    ],
    "constants": ["stiffness::K_source"]
  })";
  EXPECT_THROW(u::make_json_builder<policy>(to_source), u::fatal_error);
}

/// A repeat overwrites, dropping one deck value and leaving what it should
/// have bound at its placeholder — a zero modulus, in the case that prompted
/// this.
TEST(JsonModel, RejectsADuplicateConstantsTarget) {
  const char* doc = R"({
    "materials": [
      {"type": "constant_scalar", "name": "K", "value": 0},
      {"type": "constant_scalar", "name": "G", "value": 0}
    ],
    "constants": ["K::value", "K::value"]
  })";
  EXPECT_THROW(u::make_json_builder<policy>(doc), u::fatal_error);

  // Two entries against one material with different parameters is legitimate,
  // so the key is the whole target.
  const char* two_params = R"({
    "materials": [
      {"type": "external_strain_source", "name": "strain_in"},
      {"type": "linear_elasticity", "name": "el",
       "strain_producer_name": "strain_in", "K": 0, "G": 0}
    ],
    "constants": ["el::K", "el::G"]
  })";
  EXPECT_NO_THROW(u::make_json_builder<policy>(two_params));
}

/// A target naming an undefined material would substitute nothing and leave the
/// placeholder — a wrong-but-plausible modulus rather than an error.
TEST(JsonModel, RejectsAConstantsEntryTargetingAnUndefinedMaterial) {
  const char* doc = R"({
    "materials": [{"type": "constant_scalar", "name": "K", "value": 0}],
    "constants": ["Gee::value"]
  })";
  EXPECT_THROW(u::make_json_builder<policy>(doc), u::fatal_error);
}

// Changing the constants for one material name is fatal, but that belongs to
// the registry — see UmatInterface.ChangingPropsValuesForTheSameNameIsFatal.

/// Too few constants is a setup error: fatal, not a cutback.
TEST(JsonModel, TooFewDeckConstantsIsFatal) {
  const T only_one[1] = {100.0};
  const fortran_name cm("JSONELASTIC");
  T statev[1] = {0};
  const T stran[6] = {0}, dstran[6] = {0.001, 0, 0, 0, 0, 0};
  T stress[6] = {0}, ddsdde[36] = {0}, pnewdt = 1.0;
  T sse = 0, spd = 0, scd = 0, rpl = 0, ddsddt[6] = {0}, drplde[6] = {0};
  T drpldt = 0;
  const T time[2] = {0, 0};
  T dtime = 0.1;
  const T temp = 0, dtemp = 0, predef = 0, dpred = 0, celent = 1;
  const T coords[3] = {0}, drot[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const T dfg[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  int noel = 1, npt = 1, layer = 1, kspt = 1, jstep = 1, kinc = 1;
  int ndi = 3, nshr = 3, ntens = 6, nstatv = 0, nprops = 1;

  bool fatal = false;
  u::set_fatal_handler([](const char*) {});
  umat_(stress, statev, ddsdde, &sse, &spd, &scd, &rpl, ddsddt, drplde, &drpldt,
        stran, dstran, time, &dtime, &temp, &dtemp, &predef, &dpred, cm.buf,
        &ndi, &nshr, &ntens, &nstatv, only_one, &nprops, coords, drot, &pnewdt,
        &celent, dfg, dfg, &noel, &npt, &layer, &kspt, &jstep, &kinc, 80);
  fatal = (pnewdt == 1.0);   // fatal path leaves PNEWDT alone
  u::set_fatal_handler(nullptr);
  EXPECT_TRUE(fatal) << "a wrong CONSTANTS= count must not request a cutback";
}

}  // namespace
