#include <gtest/gtest.h>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/statev_map.h"
#include "numsim-materials/solvers/local_newton.h"
#include "numsim-materials/materials/j2_plasticity.h"

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
    ctx.create<nm::local_newton<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "hardening");
    p.insert<std::string>("source", "j2");
    p.insert<T>("K", T{1000});
    ctx.create<nm::linear_isotropic_hardening<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "j2");
    p.insert<T>("K", K);
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

/// A material carrying rank-4 history. No production material has one, so
/// without this fixture the entire tensor4 branch of statev_map — its 36-slot
/// pack/unpack, its minor-symmetry check and its rotation — is unreachable from
/// the tests and could be deleted without a single failure.
template <typename Traits>
class tensor4_history_material final
    : public nm::material_base<tensor4_history_material<Traits>, Traits> {
public:
  using base = nm::material_base<tensor4_history_material<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using tensor4 = tmech::tensor<value_type, 3, 4>;

  template <typename... Args>
  explicit tensor4_history_material(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_c(base::template add_history_output<tensor4>("stored_tangent")) {}

  static input_parameter_controller parameters() { return base::parameters(); }

  nm::history_property<tensor4>& stored() { return m_c; }

private:
  nm::history_property<tensor4>& m_c;
};

/// Scalar history under a property name that sorts BEFORE the J2 fixture's
/// property names. Needed to tell (owner, name) apart from (name, owner):
/// with every property called "state" the two keys agree, which is why the
/// first attempt at this test could not detect a swapped comparator.
template <typename Traits>
class early_name_history_material final
    : public nm::material_base<early_name_history_material<Traits>, Traits> {
public:
  using base = nm::material_base<early_name_history_material<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit early_name_history_material(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_v(base::template add_history_output<value_type>("a_state")) {}

  static input_parameter_controller parameters() { return base::parameters(); }

private:
  nm::history_property<value_type>& m_v;
};

/// History of a type STATEV cannot represent, to exercise the rejection branch.
template <typename Traits>
class int_history_material final
    : public nm::material_base<int_history_material<Traits>, Traits> {
public:
  using base = nm::material_base<int_history_material<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit int_history_material(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_flag(base::template add_history_output<int>("flag")) {}

  static input_parameter_controller parameters() { return base::parameters(); }

private:
  nm::history_property<int>& m_flag;
};

/// Isotropic (hence minor-symmetric) rank-4 tensor.
tmech::tensor<T, 3, 4> isotropic4(T K, T G) {
  const auto I = tmech::eye<T, 3, 2>();
  const auto IIsym = (tmech::otimesu(I, I) + tmech::otimesl(I, I)) * 0.5;
  const auto IIvol = tmech::otimes(I, I) / 3.0;
  tmech::tensor<T, 3, 4> C;
  C = 3.0 * K * IIvol + 2.0 * G * (IIsym - IIvol);
  return C;
}

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
  EXPECT_EQ(desc[1], "1..6  j2::plastic_strain  [engineering shear]");
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


// ---------------------------------------------------------------------------
// The STATEV encoding itself
// ---------------------------------------------------------------------------

/// Every other test round-trips through the same pack/unpack pair, so they only
/// establish self-consistency — swapping the shear convention, or permuting two
/// slots, passes them all. This asserts what the raw slots MEAN, which is the
/// only thing a restart file or an SDV field output depends on.
TEST(StatevMap, RawSlotsUseAbaqusOrderAndEngineeringShear) {
  J2Fixture f;
  const map_type map(f.ctx, host_owned);

  tensor2 eps_p;
  eps_p.fill(0.0);
  eps_p(0, 0) = 0.011;  eps_p(1, 1) = 0.022;  eps_p(2, 2) = 0.033;
  eps_p(0, 1) = eps_p(1, 0) = 0.044;   // -> gamma_12 = 0.088
  eps_p(0, 2) = eps_p(2, 0) = 0.055;   // -> gamma_13 = 0.110
  eps_p(1, 2) = eps_p(2, 1) = 0.066;   // -> gamma_23 = 0.132

  f.ctx.get_mutable<tensor2>("j2", "plastic_strain") = eps_p;
  f.ctx.get_mutable<T>("j2", "equivalent_plastic_strain") = 0.123;

  std::vector<T> statev(map.nstatv(), 0.0);
  map.pack(statev.data());

  // Layout is sorted by (owner, name): the scalar first, then the tensor in
  // Abaqus order {11,22,33,12,13,23} with ENGINEERING shear.
  EXPECT_DOUBLE_EQ(statev[0], 0.123);
  EXPECT_DOUBLE_EQ(statev[1], 0.011);
  EXPECT_DOUBLE_EQ(statev[2], 0.022);
  EXPECT_DOUBLE_EQ(statev[3], 0.033);
  EXPECT_DOUBLE_EQ(statev[4], 0.088);
  EXPECT_DOUBLE_EQ(statev[5], 0.110);
  EXPECT_DOUBLE_EQ(statev[6], 0.132);
}

/// The sort key is (owner, then name). With a single-owner fixture that is
/// indistinguishable from (name, then owner), so this uses two owners whose
/// orderings disagree between the two keys.
TEST(StatevMap, SortsByOwnerBeforeProperty) {
  // Owner sorts LAST, property name sorts FIRST, so the two candidate keys
  // disagree: by (owner, name) this entry comes last; by (name, owner) it would
  // come first. A fixture whose properties share a name cannot tell them apart.
  ctx_type ctx;
  param_type q;
  q.insert<std::string>("name", "zzz");
  ctx.create<early_name_history_material<policy>>(q);
  q.clear();
  q.insert<std::string>("name", "aaa");
  ctx.create<tensor4_history_material<policy>>(q);  // property "stored_tangent"
  ctx.finalize();

  const map_type map(ctx, {});
  const auto desc = map.describe();
  ASSERT_EQ(desc.size(), 2u);
  // (owner, name): aaa::stored_tangent then zzz::a_state.
  // (name, owner) would invert this, since "a_state" < "stored_tangent".
  EXPECT_EQ(desc[0], "0..35  aaa::stored_tangent");
  EXPECT_EQ(desc[1], "36..36  zzz::a_state");
}

// ---------------------------------------------------------------------------
// Rank-4 history
// ---------------------------------------------------------------------------

TEST(StatevMap, RoundTripsRankFourHistory) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "store");
  auto& mat = ctx.create<tensor4_history_material<policy>>(p);
  ctx.finalize();

  const map_type map(ctx, {});
  EXPECT_EQ(map.nstatv(), 36u);

  const auto C = isotropic4(166.67, 76.92);
  mat.stored().new_value() = C;

  std::vector<T> statev(map.nstatv(), 0.0);
  map.pack(statev.data());

  mat.stored().new_value() = tmech::tensor<T, 3, 4>();
  mat.stored().old_value() = tmech::tensor<T, 3, 4>();
  map.unpack(statev.data());

  const auto& back = mat.stored().new_value();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          EXPECT_NEAR(back(i, j, k, l), C(i, j, k, l), 1e-10);
  // unpack sets both sides, as for every other type.
  EXPECT_NEAR(mat.stored().old_value()(0, 0, 0, 0), C(0, 0, 0, 0), 1e-10);
}

/// 36 slots can only hold a minor-symmetric rank-4 tensor; anything else must
/// be reported rather than silently truncated.
TEST(StatevMap, PackRejectsRankFourWithoutMinorSymmetry) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "store");
  auto& mat = ctx.create<tensor4_history_material<policy>>(p);
  ctx.finalize();
  const map_type map(ctx, {});

  const auto I = tmech::eye<T, 3, 2>();
  tmech::tensor<T, 3, 4> bad;
  bad = tmech::otimesu(I, I);  // not minor-symmetric
  mat.stored().new_value() = bad;

  std::vector<T> statev(map.nstatv(), 0.0);
  EXPECT_THROW(map.pack(statev.data()), u::fatal_error);
}

TEST(StatevMap, RotatesRankFourHistory) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "store");
  auto& mat = ctx.create<tensor4_history_material<policy>>(p);
  ctx.finalize();
  const map_type map(ctx, {});

  EXPECT_TRUE(map.has_rotatable_history());

  // A non-isotropic but minor-symmetric tensor, so the rotation is observable.
  tensor2 A, B;
  A.fill(0.0);
  B.fill(0.0);
  A(0, 0) = 1.0;  A(1, 1) = 2.0;  A(2, 2) = 3.0;
  B(0, 0) = 5.0;  B(1, 1) = 6.0;  B(2, 2) = 7.0;
  tmech::tensor<T, 3, 4> C;
  C = tmech::otimes(A, B);
  mat.stored().new_value() = C;
  mat.stored().old_value() = C;

  T drot[9];
  for (auto& v : drot) v = 0.0;
  drot[0 + 3 * 1] = -1.0;
  drot[1 + 3 * 0] = 1.0;
  drot[2 + 3 * 2] = 1.0;
  map.rotate_history(u::rotation_from_buffer<T>(drot));

  // R (A (x) B) R^T = (R A R^T) (x) (R B R^T); a 90-degree z-rotation swaps the
  // 11 and 22 entries of both factors, so C_1111 becomes A_22 * B_22.
  EXPECT_NEAR(mat.stored().new_value()(0, 0, 0, 0), 2.0 * 6.0, 1e-10);
  EXPECT_NEAR(mat.stored().old_value()(0, 0, 0, 0), 2.0 * 6.0, 1e-10);
}

/// A history type outside the supported set must be refused, not ignored.
/// Silently skipping it would leave that state out of STATEV entirely, so it
/// would reset to its default at the start of every increment.
TEST(StatevMap, RejectsUnsupportedHistoryType) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "weird");
  ctx.create<int_history_material<policy>>(p);
  ctx.finalize();
  EXPECT_THROW(map_type(ctx, {}), u::fatal_error);
}

}  // namespace
