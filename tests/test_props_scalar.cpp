#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/isotropic_tangent.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/props_scalar.h"
#include "numsim-materials/umat/json_model.h"

// The Fortran-callable symbol, so live constants are exercised through the real
// ABI and not only through the C++ evaluator.
NUMSIM_MATERIALS_DEFINE_UMAT(numsim::materials::material_policy_default)

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using registry = u::umat_registry<policy>;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

/// C1111 for an isotropic tangent, the quantity every test here reads back.
constexpr T c1111(T K, T G) { return K + 4.0 * G / 3.0; }

// ---------------------------------------------------------------------------
// The material on its own
// ---------------------------------------------------------------------------

TEST(PropsScalar, PublishesTheConstantAtItsIndex) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "K");
  p.insert<std::size_t>("index", 1);
  auto& k = ctx.create<nm::props_scalar<policy>>(p);
  ctx.finalize();

  const T props[3] = {11.0, 22.0, 33.0};
  k.bind(std::span<const T>(props, 3));
  EXPECT_DOUBLE_EQ(ctx.get<T>("K", "value"), 22.0);
}

/// A definite value before the first bind, not whatever the storage held.
TEST(PropsScalar, IsZeroBeforeTheFirstBind) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "K");
  p.insert<std::size_t>("index", 0);
  ctx.create<nm::props_scalar<policy>>(p);
  ctx.finalize();
  EXPECT_DOUBLE_EQ(ctx.get<T>("K", "value"), 0.0);
}

TEST(PropsScalar, RejectsAnIndexPastTheSuppliedConstants) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "K");
  p.insert<std::size_t>("index", 5);
  auto& k = ctx.create<nm::props_scalar<policy>>(p);
  ctx.finalize();

  const T props[2] = {1.0, 2.0};
  EXPECT_THROW(k.bind(std::span<const T>(props, 2)), std::out_of_range);
}

/// Plain, not history: no STATEV slot. external_scalar_source would cost one
/// per constant.
TEST(PropsScalar, CostsNoStatevSlot) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "strain_in");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "K");
  p.insert<std::size_t>("index", 0);
  ctx.create<nm::props_scalar<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "G");
  p.insert<std::size_t>("index", 1);
  ctx.create<nm::props_scalar<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "stiffness");
  p.insert<std::string>("K_source", "K");
  p.insert<std::string>("G_source", "G");
  ctx.create<nm::isotropic_tangent<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("tangent_source", "stiffness");
  p.insert<std::string>("strain_source", "strain_in");
  ctx.create<nm::linear_stress<policy>>(p);
  ctx.finalize();

  u::material_point_evaluator<policy>::config cfg;
  cfg.strain_source = "strain_in";
  cfg.stress_source = "elastic";
  cfg.tangent_source = "stiffness";
  u::material_point_evaluator<policy> eval(ctx, cfg);

  EXPECT_EQ(eval.nstatv(), 0u);
  EXPECT_TRUE(eval.has_live_props());
}

// ---------------------------------------------------------------------------
// Through the evaluator
// ---------------------------------------------------------------------------

/// A model whose two moduli are live deck constants.
void build_live(ctx_type& ctx) {
  param_type p;
  p.insert<std::string>("name", "strain_in");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "K");
  p.insert<std::size_t>("index", 0);
  ctx.create<nm::props_scalar<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "G");
  p.insert<std::size_t>("index", 1);
  ctx.create<nm::props_scalar<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "stiffness");
  p.insert<std::string>("K_source", "K");
  p.insert<std::string>("G_source", "G");
  ctx.create<nm::isotropic_tangent<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("tangent_source", "stiffness");
  p.insert<std::string>("strain_source", "strain_in");
  ctx.create<nm::linear_stress<policy>>(p);
  ctx.finalize();
}

u::material_point_evaluator<policy>::config live_config() {
  u::material_point_evaluator<policy>::config cfg;
  cfg.strain_source = "strain_in";
  cfg.stress_source = "elastic";
  cfg.tangent_source = "stiffness";
  return cfg;
}

/// One uniaxial call, returning DDSDDE(1,1).
T call_once(u::material_point_evaluator<policy>& eval,
            std::span<const T> props) {
  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  std::vector<T> stress(6, 0.0), ddsdde(36, 0.0), statev;
  u::material_point_evaluator<policy>::call c;
  c.stran = stran;
  c.dstran = dstran;
  c.stress = stress;
  c.ddsdde = ddsdde;
  c.statev = statev;
  eval.bind_props(props);
  eval.evaluate(c);
  return ddsdde[0];
}

/// The point: one graph, two constant sets, two stiffnesses, no rebuild.
TEST(PropsScalar, ChangedConstantsChangeTheTangentWithoutARebuild) {
  ctx_type ctx;
  build_live(ctx);
  u::material_point_evaluator<policy> eval(ctx, live_config());

  const T soft[2] = {100.0, 40.0};
  const T stiff[2] = {300.0, 140.0};

  EXPECT_NEAR(call_once(eval, soft), c1111(100.0, 40.0), 1e-9);
  EXPECT_NEAR(call_once(eval, stiff), c1111(300.0, 140.0), 1e-9);
  // and back, so the second answer is not simply "the last one wins forever"
  EXPECT_NEAR(call_once(eval, soft), c1111(100.0, 40.0), 1e-9);
}

/// The stored-pointer trap. The buffer is overwritten AFTER bind_props and
/// before the graph runs: dereferencing at bind time makes that irrelevant,
/// keeping the pointer makes the tangent follow the clobbered values.
TEST(PropsScalar, ReadsTheConstantsAtBindTimeNotAtUpdateTime) {
  ctx_type ctx;
  build_live(ctx);
  u::material_point_evaluator<policy> eval(ctx, live_config());

  std::vector<T> buffer{100.0, 40.0};
  eval.bind_props(buffer);

  // Everything the host promised is gone by the time the graph runs.
  buffer[0] = -1.0e9;
  buffer[1] = -1.0e9;

  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  std::vector<T> stress(6, 0.0), ddsdde(36, 0.0), statev;
  u::material_point_evaluator<policy>::call c;
  c.stran = stran;
  c.dstran = dstran;
  c.stress = stress;
  c.ddsdde = ddsdde;
  c.statev = statev;
  eval.evaluate(c);

  EXPECT_NEAR(ddsdde[0], c1111(100.0, 40.0), 1e-9)
      << "the constants must be copied at bind(), not read through a kept "
         "pointer during update()";
}

/// Every reader publishes zero until bind_props runs, so an unbound model would
/// evaluate with moduli of zero and an all-zero DDSDDE.
TEST(PropsScalar, EvaluatingBeforeBindingIsFatal) {
  ctx_type ctx;
  build_live(ctx);
  u::material_point_evaluator<policy> eval(ctx, live_config());

  const T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.001, 0, 0, 0, 0, 0};
  std::vector<T> stress(6, 0.0), ddsdde(36, 0.0), statev;
  u::material_point_evaluator<policy>::call c;
  c.stran = stran;
  c.dstran = dstran;
  c.stress = stress;
  c.ddsdde = ddsdde;
  c.statev = statev;

  EXPECT_THROW(eval.evaluate(c), u::fatal_error);

  // Bound, it evaluates normally — the guard must not be a one-way latch.
  const T props[2] = {100.0, 40.0};
  eval.bind_props(props);
  EXPECT_NO_THROW(eval.evaluate(c));
  EXPECT_NEAR(ddsdde[0], c1111(100.0, 40.0), 1e-9);
}

/// The one place binding meets an ITERATIVE evaluator. If a change moved the
/// bind inside the loop, or dropped the forward, only this notices.
TEST(PropsScalar, PlaneStressUsesTheLiveConstants) {
  ctx_type ctx;
  build_live(ctx);
  u::plane_stress_evaluator<policy> ps(ctx, live_config(), {});

  constexpr T K = 100.0, G = 40.0;
  const T props[2] = {K, G};
  const T stran[3] = {0, 0, 0};
  const T dstran[3] = {0.001, 0, 0};
  std::vector<T> stress(3, 0.0), ddsdde(9, 0.0), statev(ps.nstatv(), 0.0);
  u::material_point_evaluator<policy>::call c;
  c.stran = stran;
  c.dstran = dstran;
  c.stress = stress;
  c.ddsdde = ddsdde;
  c.statev = statev;
  c.ec = u::element_case::plane_stress;

  ps.bind_props(props);
  ps.evaluate(c);

  // Condensed modulus, derived from the constants the host supplied.
  const T E = 9 * K * G / (3 * K + G);
  const T nu = (3 * K - 2 * G) / (2 * (3 * K + G));
  EXPECT_NEAR(ddsdde[0], E / (1 - nu * nu), 1e-9);
  EXPECT_NEAR(stress[2], 0.0, 1e-10) << "sigma_33 must be driven to zero";
}

/// Too few constants is a setup error: fatal, not a cutback.
TEST(PropsScalar, TooFewConstantsIsFatalAtTheEvaluator) {
  ctx_type ctx;
  build_live(ctx);
  u::material_point_evaluator<policy> eval(ctx, live_config());

  const T only_one[1] = {100.0};
  EXPECT_THROW(eval.bind_props(std::span<const T>(only_one, 1)), u::fatal_error);
}

/// A baked-constant model reports itself as such and pays for nothing.
TEST(PropsScalar, BakedConstantModelsHaveNoLiveProps) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "strain_in");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "K");
  p.insert<T>("value", 100.0);
  ctx.create<nm::constant_scalar<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "G");
  p.insert<T>("value", 40.0);
  ctx.create<nm::constant_scalar<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "stiffness");
  p.insert<std::string>("K_source", "K");
  p.insert<std::string>("G_source", "G");
  ctx.create<nm::isotropic_tangent<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("tangent_source", "stiffness");
  p.insert<std::string>("strain_source", "strain_in");
  ctx.create<nm::linear_stress<policy>>(p);
  ctx.finalize();

  u::material_point_evaluator<policy> eval(ctx, live_config());
  EXPECT_FALSE(eval.has_live_props());
  // bind_props is then a no-op, including for an empty array.
  EXPECT_NO_THROW(eval.bind_props({}));
}

// ---------------------------------------------------------------------------
// Through JSON and the real umat_ entry point
// ---------------------------------------------------------------------------

/// Identical to the baked document but for the material type — the claim the
/// JSON layer makes, so it is asserted rather than described.
const char* kLive = R"({
  "materials": [
    {"type": "external_strain_source", "name": "strain_in"},
    {"type": "props_scalar", "name": "K"},
    {"type": "props_scalar", "name": "G"},
    {"type": "isotropic_tangent", "name": "stiffness",
     "K_source": "K", "G_source": "G"},
    {"type": "linear_stress", "name": "elastic",
     "tangent_source": "stiffness", "strain_source": "strain_in"}
  ],
  "constants": ["K::value", "G::value"]
})";

registry::config json_config() {
  registry::config cfg;
  cfg.strain_source = "strain_in";
  cfg.stress_source = "elastic";
  cfg.tangent_source = "stiffness";
  return cfg;
}

/// K baked, G live. A document may mix the two binding times, so the
/// consistency check has to be answered per slot rather than per model.
const char* kMixed = R"({
  "materials": [
    {"type": "external_strain_source", "name": "strain_in"},
    {"type": "constant_scalar", "name": "K", "value": 0},
    {"type": "props_scalar", "name": "G"},
    {"type": "isotropic_tangent", "name": "stiffness",
     "K_source": "K", "G_source": "G"},
    {"type": "linear_stress", "name": "elastic",
     "tangent_source": "stiffness", "strain_source": "strain_in"}
  ],
  "constants": ["K::value", "G::value"]
})";

struct Registration {
  Registration() {
    u::register_json_model<policy>("LIVEELASTIC", kLive, json_config());
    // One name per test: each warms the cache itself, so sharing would let
    // test order decide the outcome.
    u::register_json_model<policy>("MIXEDBAKED", kMixed, json_config());
    u::register_json_model<policy>("MIXEDLIVE", kMixed, json_config());
  }
};
const Registration registration_{};

struct fortran_name {
  char buf[80];
  explicit fortran_name(const std::string& s) {
    for (auto& c : buf) c = ' ';
    for (std::size_t i = 0; i < s.size() && i < 80; ++i) buf[i] = s[i];
  }
};

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

/// Position is the slot: no "index" in the document, yet K takes PROPS[0].
TEST(PropsScalarJson, ConstantsPositionBindsTheSlot) {
  const T props[2] = {250.0, 90.0};
  EXPECT_NEAR(uniaxial_tangent("LIVEELASTIC", props, 2), c1111(250.0, 90.0),
              1e-9);
}

/// The target names the property the constant arrives on — "value" — not the
/// "index" the builder writes. Identical spelling to constant_scalar is what
/// lets the type be swapped, so the wrong one is rejected.
TEST(PropsScalarJson, RejectsATargetNamingIndexRatherThanTheProperty) {
  const char* names_index = R"({
    "materials": [{"type": "props_scalar", "name": "K"}],
    "constants": ["K::index"]
  })";
  EXPECT_THROW(u::make_json_builder<policy>(names_index), u::fatal_error);

  const char* names_property = R"({
    "materials": [{"type": "props_scalar", "name": "K"}],
    "constants": ["K::value"]
  })";
  EXPECT_NO_THROW(u::make_json_builder<policy>(names_property));
}

/// The registry's consistency check stands down for live constants: nothing
/// was baked in for a changed array to contradict.
TEST(PropsScalarJson, ChangedDeckConstantsAreHonouredNotRejected) {
  const T first[2] = {100.0, 40.0};
  const T second[2] = {300.0, 140.0};

  EXPECT_NEAR(uniaxial_tangent("LIVEELASTIC", first, 2), c1111(100.0, 40.0),
              1e-9);
  EXPECT_NEAR(uniaxial_tangent("LIVEELASTIC", second, 2), c1111(300.0, 140.0),
              1e-9);
}

// ---------------------------------------------------------------------------
// Mixed documents: baked and live constants side by side
// ---------------------------------------------------------------------------

/// A changed BAKED constant is still fatal beside a live one. Answering
/// has_live_props() per model let this through: G tracked the deck while K
/// kept its first value, with no diagnostic.
TEST(PropsScalarJson, ChangingABakedConstantIsFatalEvenBesideALiveOne) {
  const T first[2] = {100.0, 40.0};
  const T changed_both[2] = {300.0, 140.0};  // K baked, G live

  ASSERT_NEAR(uniaxial_tangent("MIXEDBAKED", first, 2), c1111(100.0, 40.0),
              1e-9);

  int fatal_count = 0;
  static int* counter = &fatal_count;
  u::set_fatal_handler([](const char*) { ++*counter; });
  const T got = uniaxial_tangent("MIXEDBAKED", changed_both, 2);
  u::set_fatal_handler(nullptr);

  EXPECT_EQ(fatal_count, 1)
      << "a changed baked constant must be reported, not served from the "
         "cached graph because some OTHER constant is live";
  // The wrong-but-plausible answer this used to return.
  EXPECT_FALSE(std::abs(got - c1111(100.0, 140.0)) < 1e-9)
      << "K kept its first value while G followed the deck";
}

/// The other half, so the fix cannot pass by rejecting every mixed document.
TEST(PropsScalarJson, ChangingOnlyTheLiveConstantIsHonouredInAMixedDocument) {
  const T first[2] = {100.0, 40.0};
  const T live_changed[2] = {100.0, 140.0};  // K unchanged, G changed

  EXPECT_NEAR(uniaxial_tangent("MIXEDLIVE", first, 2), c1111(100.0, 40.0),
              1e-9);
  EXPECT_NEAR(uniaxial_tangent("MIXEDLIVE", live_changed, 2),
              c1111(100.0, 140.0), 1e-9);
}

}  // namespace
