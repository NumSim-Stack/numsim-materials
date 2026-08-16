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

/// Elastic model with the moduli bound to the deck's constants. Nothing here is
/// compiled: adding a material means editing this string.
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

/// The point of the whole exercise: the model is a document, the constants come
/// from the deck, and neither requires recompiling the UMAT.
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

/// A document using the old "props" spelling would otherwise be accepted with
/// every constant unbound, leaving the placeholders as the material's moduli —
/// wrong but plausible, and completely silent.
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

/// A constants entry naming a material the document does not define is a typo that
/// would otherwise substitute nothing and leave the placeholder in place — a
/// wrong-but-plausible modulus rather than an error.
TEST(JsonModel, RejectsAConstantsEntryTargetingAnUndefinedMaterial) {
  const char* doc = R"({
    "materials": [{"type": "constant_scalar", "name": "K", "value": 0}],
    "constants": ["Gee::value"]
  })";
  EXPECT_THROW(u::make_json_builder<policy>(doc), u::fatal_error);
}

/// One material NAME carries one PROPS array. Supplying different constants for
/// a name whose graph is already built would otherwise return the first call's
/// stiffness forever, silently — the cached context is keyed on the name.
TEST(JsonModel, ChangingConstantsForOneMaterialNameIsFatal) {
  const T first[2] = {100.0, 40.0};
  const T second[2] = {300.0, 140.0};

  EXPECT_NEAR(uniaxial_tangent("JSONELASTIC", first, 2),
              100.0 + 4.0 * 40.0 / 3.0, 1e-9);

  int fatal_count = 0;
  static int* counter = &fatal_count;
  u::set_fatal_handler([](const char*) { ++*counter; });
  uniaxial_tangent("JSONELASTIC", second, 2);
  u::set_fatal_handler(nullptr);

  EXPECT_EQ(fatal_count, 1)
      << "same name, different constants must be reported, not ignored";
}

/// Fewer constants than the document binds is a *DEPVAR-style setup error, and
/// must be fatal rather than a cutback.
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
