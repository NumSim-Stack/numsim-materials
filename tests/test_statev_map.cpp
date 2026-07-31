#include <gtest/gtest.h>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/small_strain_plasticity.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/statev_map.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using map_type = u::statev_map<policy>;

constexpr T K = 166.67;
constexpr T G = 76.92;

/// external_strain_source ("stepper") + the J2 chain.
///
/// History properties present:
///   stepper::strain                    tensor2, host-owned  -> excluded
///   j2::plastic_strain                 tensor2              -> 6 slots
///   j2::equivalent_plastic_strain      scalar               -> 1 slot
struct J2Fixture {
  ctx_type ctx;
  nm::external_strain_source<policy>* src{nullptr};

  J2Fixture() {
    param_type p;

    p.clear();
    p.insert<std::string>("name", "stepper");
    src = &ctx.create<nm::external_strain_source<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "stepper");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    ctx.create<nm::linear_elasticity<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "solver");
    ctx.create<nm::backward_euler<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "hardening");
    p.insert<std::string>("source", "j2");
    p.insert<T>("K", T{1000});
    ctx.create<nm::linear_isotropic_hardening<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "j2");
    p.insert<std::string>("elastic_source", "elastic");
    p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<std::string>("solver_source", "solver");
    p.insert<T>("G", G);
    p.insert<T>("sigma_0", T{50});
    ctx.create<nm::j2_plasticity<policy>>(p);

    ctx.finalize();
  }
};

const std::vector<u::statev_exclusion> host_owned = {{"stepper", "strain"}};

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

TEST(StatevMap, ExcludesHostOwnedStrain) {
  J2Fixture f;
  const map_type excluded(f.ctx, host_owned);
  const map_type included(f.ctx, {});

  // plastic_strain (6) + equivalent_plastic_strain (1)
  EXPECT_EQ(excluded.nstatv(), 7u);
  // ... plus the host-owned strain (6)
  EXPECT_EQ(included.nstatv(), 13u);
}

TEST(StatevMap, LayoutIsSortedByOwnerThenProperty) {
  J2Fixture f;
  const map_type map(f.ctx, host_owned);
  const auto desc = map.describe();

  ASSERT_EQ(desc.size(), 2u);
  EXPECT_EQ(desc[0], "0..0  j2::equivalent_plastic_strain");
  EXPECT_EQ(desc[1], "1..6  j2::plastic_strain");
}

/// The layout must not depend on graph execution order, so two maps built over
/// the same context always agree.
TEST(StatevMap, LayoutIsReproducible) {
  J2Fixture f;
  const map_type a(f.ctx, host_owned);
  const map_type b(f.ctx, host_owned);
  EXPECT_EQ(a.describe(), b.describe());
}

TEST(StatevMap, ThrowsOnUnmatchedExclusion) {
  J2Fixture f;
  EXPECT_THROW(map_type(f.ctx, {{"stepper", "strian"}}),  // typo
               std::runtime_error);
  EXPECT_THROW(map_type(f.ctx, {{"nosuchmaterial", "strain"}}),
               std::runtime_error);
  // A real property that is not history is also not a valid exclusion.
  EXPECT_THROW(map_type(f.ctx, {{"elastic", "stress"}}), std::runtime_error);
}

TEST(StatevMap, CheckNstatvRejectsUndersizedHostArray) {
  J2Fixture f;
  const map_type map(f.ctx, host_owned);
  EXPECT_NO_THROW(map.check_nstatv(7));
  EXPECT_NO_THROW(map.check_nstatv(20));  // host may over-allocate
  EXPECT_THROW(map.check_nstatv(6), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Round trip
// ---------------------------------------------------------------------------

TEST(StatevMap, RoundTripsHistoryThroughFlatArray) {
  J2Fixture f;
  const map_type map(f.ctx, host_owned);

  tensor2 eps_p;
  eps_p.fill(0.0);
  eps_p(0, 0) = 0.011;  eps_p(1, 1) = 0.022;  eps_p(2, 2) = 0.033;
  eps_p(0, 1) = eps_p(1, 0) = 0.044;
  eps_p(0, 2) = eps_p(2, 0) = 0.055;
  eps_p(1, 2) = eps_p(2, 1) = 0.066;

  f.ctx.get_mutable<tensor2>("j2", "plastic_strain") = eps_p;
  f.ctx.get_mutable<T>("j2", "equivalent_plastic_strain") = 0.123;

  std::vector<T> statev(map.nstatv(), 0.0);
  map.pack(statev.data());

  EXPECT_DOUBLE_EQ(statev[0], 0.123);

  // Clobber, then restore.
  f.ctx.get_mutable<tensor2>("j2", "plastic_strain").fill(0.0);
  f.ctx.get_mutable<T>("j2", "equivalent_plastic_strain") = -99.0;

  map.unpack(statev.data());

  const auto& back = f.ctx.get<tensor2>("j2", "plastic_strain");
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_DOUBLE_EQ(back(i, j), eps_p(i, j)) << i << "," << j;
  EXPECT_DOUBLE_EQ(f.ctx.get<T>("j2", "equivalent_plastic_strain"), 0.123);
}

/// unpack() must set BOTH sides: STATEV is the converged state at t_n, so a
/// step that never writes `new` has to report t_n, not a stale iterate.
TEST(StatevMap, UnpackSetsOldAndNew) {
  J2Fixture f;
  const map_type map(f.ctx, host_owned);

  std::vector<T> statev(map.nstatv(), 0.0);
  statev[0] = 0.42;  // equivalent_plastic_strain
  map.unpack(statev.data());

  auto* prop = f.ctx.find_property("j2", "equivalent_plastic_strain");
  ASSERT_NE(prop, nullptr);
  auto* hist =
      dynamic_cast<numsim_core::history_property<T, nm::property_traits>*>(prop);
  ASSERT_NE(hist, nullptr);

  EXPECT_DOUBLE_EQ(hist->old_value(), 0.42);
  EXPECT_DOUBLE_EQ(hist->new_value(), 0.42);
}

/// Storing a rank-2 history variable in 6 slots assumes symmetry. A tensor that
/// violates it must be reported, not silently stripped of its skew part.
TEST(StatevMap, PackRejectsNonSymmetricRankTwoHistory) {
  J2Fixture f;
  const map_type map(f.ctx, host_owned);

  auto& eps_p = f.ctx.get_mutable<tensor2>("j2", "plastic_strain");
  eps_p.fill(0.0);
  eps_p(0, 1) = 1.0;
  eps_p(1, 0) = -1.0;  // skew

  std::vector<T> statev(map.nstatv(), 0.0);
  EXPECT_THROW(map.pack(statev.data()), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Statelessness: the property that makes one context per thread sufficient
// ---------------------------------------------------------------------------

/// Driving a step, round-tripping STATEV, and driving the next step must give
/// the same answer as never touching STATEV at all — i.e. STATEV genuinely
/// carries everything the context needs.
TEST(StatevMap, CarriesEverythingNeededBetweenSteps) {
  auto uniaxial = [](T v) {
    tensor2 e;
    e.fill(0.0);
    e(0, 0) = v;
    return e;
  };

  // Reference: one context, never serialised.
  J2Fixture ref;
  std::vector<T> ref_stress;
  {
    tensor2 prev = uniaxial(0.0);
    for (int step = 0; step < 25; ++step) {
      const auto cur = uniaxial(0.004 * (step + 1));
      ref.src->bind(prev, cur);
      ref.ctx.update();
      ref_stress.push_back(ref.ctx.get<tensor2>("j2", "stress")(0, 0));
      ref.ctx.commit();
      prev = cur;
    }
  }

  // Through STATEV: a FRESH context every step, state carried only by the
  // flat array. If anything lived outside STATEV, these would diverge.
  std::vector<T> statev;
  {
    J2Fixture probe;
    statev.assign(map_type(probe.ctx, host_owned).nstatv(), 0.0);
  }

  tensor2 prev = uniaxial(0.0);
  for (int step = 0; step < 25; ++step) {
    J2Fixture f;
    const map_type map(f.ctx, host_owned);

    map.unpack(statev.data());
    const auto cur = uniaxial(0.004 * (step + 1));
    f.src->bind(prev, cur);
    f.ctx.update();
    map.pack(statev.data());

    EXPECT_DOUBLE_EQ(f.ctx.get<tensor2>("j2", "stress")(0, 0), ref_stress[step])
        << "step " << step;
    prev = cur;
  }
}

}  // namespace
