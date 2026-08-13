#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/isotropic_tangent.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/small_strain_plasticity.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/umat/external_state_source.h"

namespace {

namespace nm = numsim::materials;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

constexpr T K = 166.67;
constexpr T G = 76.92;

/// strain source -> isotropic_tangent -> linear_stress
nm::external_strain_source<policy>& build_decomposed(ctx_type& ctx, T k, T g,
                                                     bool recompute) {
  param_type p;
  p.insert<std::string>("name", "strain_in");
  auto& src = ctx.create<nm::external_strain_source<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "stiffness");
  p.insert<T>("K", k);
  p.insert<T>("G", g);
  p.insert<bool>("recompute", recompute);
  ctx.create<nm::isotropic_tangent<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("tangent_source", "stiffness");
  p.insert<std::string>("strain_source", "strain_in");
  ctx.create<nm::linear_stress<policy>>(p);

  ctx.finalize();
  return src;
}

tensor2 uniaxial(T v) {
  tensor2 e;
  e.fill(0.0);
  e(0, 0) = v;
  return e;
}

// ---------------------------------------------------------------------------
// The ordering this decomposition exists to fix
// ---------------------------------------------------------------------------

/// The whole point. Within one material, `stress` sorts BEFORE `tangent`
/// because a material reading its own property creates no graph edge — so
/// linear_elasticity could never recompute its tangent safely. Consuming
/// another material's property creates a real Global edge, and the topological
/// sort then puts the producer first.
TEST(TangentGenerator, StiffnessIsOrderedBeforeTheStressThatConsumesIt) {
  ctx_type ctx;
  build_decomposed(ctx, K, G, /*recompute=*/true);

  // Not a sentinel of 0: the tangent has no inputs, so index 0 is exactly where
  // it legitimately lands, and a not-found sentinel of 0 would be
  // indistinguishable from the correct answer. The test would then pass on a
  // graph that no longer contains the property being ordered.
  std::optional<std::size_t> i_tangent, i_stress;
  std::size_t n = 0;
  for (const auto* prop : ctx.property_execution_order()) {
    const auto& id = prop->traits().id;
    if (id.owner == "stiffness" && id.name == "tangent") i_tangent = n;
    if (id.owner == "elastic" && id.name == "stress") i_stress = n;
    ++n;
  }
  ASSERT_TRUE(i_tangent.has_value()) << "stiffness::tangent is not in the graph";
  ASSERT_TRUE(i_stress.has_value()) << "elastic::stress is not in the graph";
  EXPECT_LT(*i_tangent, *i_stress)
      << "the stiffness must be produced before the stress reads it";
}

// ---------------------------------------------------------------------------
// Equivalence with the monolithic material
// ---------------------------------------------------------------------------

/// Decomposed and monolithic must agree exactly — same physics, different
/// graph shape.
TEST(TangentGenerator, MatchesLinearElasticityExactly) {
  ctx_type dec;
  auto& dec_src = build_decomposed(dec, K, G, /*recompute=*/false);

  ctx_type mono;
  nm::external_strain_source<policy>* mono_src = nullptr;
  {
    param_type p;
    p.insert<std::string>("name", "strain_in");
    mono_src = &mono.create<nm::external_strain_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "strain_in");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    mono.create<nm::linear_elasticity<policy>>(p);
    mono.finalize();
  }

  for (int step = 1; step <= 5; ++step) {
    const auto eps = uniaxial(0.001 * step);
    dec_src.bind(eps, eps);
    mono_src->bind(eps, eps);
    dec.update();
    mono.update();

    const auto& a = dec.get<tensor2>("elastic", "stress");
    const auto& b = mono.get<tensor2>("elastic", "stress");
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        EXPECT_DOUBLE_EQ(a(i, j), b(i, j)) << "step " << step;
  }

  // ... and the stiffness itself.
  const auto& Cd = dec.get<tensor4>("stiffness", "tangent");
  const auto& Cm = mono.get<tensor4>("elastic", "tangent");
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          EXPECT_DOUBLE_EQ(Cd(i, j, k, l), Cm(i, j, k, l));
}

// ---------------------------------------------------------------------------
// recompute: bound vs unbound callback
// ---------------------------------------------------------------------------

/// With recompute=false the property carries NO callback, so the engine skips
/// it entirely — the per-call cost is zero rather than merely small.
TEST(TangentGenerator, RecomputeFalseLeavesThePropertyWithoutACallback) {
  ctx_type ctx;
  build_decomposed(ctx, K, G, /*recompute=*/false);

  const auto* prop = ctx.find_property("stiffness", "tangent");
  ASSERT_NE(prop, nullptr);
  EXPECT_FALSE(static_cast<bool>(prop->traits().update))
      << "recompute=false must not bind an update callback";
}

TEST(TangentGenerator, RecomputeTrueBindsTheCallback) {
  ctx_type ctx;
  build_decomposed(ctx, K, G, /*recompute=*/true);

  const auto* prop = ctx.find_property("stiffness", "tangent");
  ASSERT_NE(prop, nullptr);
  EXPECT_TRUE(static_cast<bool>(prop->traits().update));
}

/// Even with no callback the tangent must be valid: it is built once in the
/// constructor, before any update() runs.
TEST(TangentGenerator, TangentIsValidBeforeTheFirstUpdate) {
  ctx_type ctx;
  build_decomposed(ctx, K, G, /*recompute=*/false);

  const auto& C = ctx.get<tensor4>("stiffness", "tangent");
  // C_1111 = K + 4G/3
  EXPECT_NEAR(C(0, 0, 0, 0), K + 4.0 * G / 3.0, 1e-9);
}

/// The behaviour the whole design is for: writing new constants in place and
/// having the graph pick them up, with correct ordering, on the next update.
///
/// Writing goes through the non-const get<T>(), which mutates the value inside
/// the std::any in place. insert() would replace the whole any and, for a type
/// past its small-buffer, relocate the object — invalidating the reference the
/// material bound at construction.
TEST(TangentGenerator, RecomputeTrueFollowsAParameterWrittenInPlace) {
  ctx_type ctx;
  auto& src = build_decomposed(ctx, K, G, /*recompute=*/true);

  const auto eps = uniaxial(0.001);
  src.bind(eps, eps);
  ctx.update();
  const T before = ctx.get<tensor2>("elastic", "stress")(0, 0);
  EXPECT_NEAR(before, (K + 4.0 * G / 3.0) * 0.001, 1e-12);

  // Double the moduli in place, as a props writer would.
  auto* stiffness = ctx.find("stiffness");
  ASSERT_NE(stiffness, nullptr);
  auto* typed = dynamic_cast<nm::isotropic_tangent<policy>*>(stiffness);
  ASSERT_NE(typed, nullptr);
  ASSERT_TRUE(typed->recomputes());
  typed->template set_parameter<T>("K", 2 * K);
  typed->template set_parameter<T>("G", 2 * G);

  ctx.update();
  const T after = ctx.get<tensor2>("elastic", "stress")(0, 0);
  EXPECT_NEAR(after, 2 * (K + 4.0 * G / 3.0) * 0.001, 1e-12);
  EXPECT_NEAR(after, 2 * before, 1e-12);
}

/// The counterpart: with recompute=false the write is visible in the parameter
/// but the tangent does NOT follow it. This is the documented trade, and the
/// test exists so the behaviour is pinned rather than discovered.
TEST(TangentGenerator, RecomputeFalseIgnoresALaterParameterWrite) {
  ctx_type ctx;
  auto& src = build_decomposed(ctx, K, G, /*recompute=*/false);

  const auto eps = uniaxial(0.001);
  src.bind(eps, eps);
  ctx.update();
  const T before = ctx.get<tensor2>("elastic", "stress")(0, 0);

  auto* typed =
      dynamic_cast<nm::isotropic_tangent<policy>*>(ctx.find("stiffness"));
  ASSERT_NE(typed, nullptr);
  EXPECT_FALSE(typed->recomputes());
  typed->template set_parameter<T>("K", 2 * K);

  ctx.update();
  EXPECT_DOUBLE_EQ(ctx.get<tensor2>("elastic", "stress")(0, 0), before)
      << "recompute=false must leave the tangent as built";
}


// ---------------------------------------------------------------------------
// The flag is construction-time only
// ---------------------------------------------------------------------------

/// The callback can only be bound at construction, so "recompute" is read once
/// and stored BY VALUE. Reading the live parameter instead would let
/// recomputes() claim the tangent tracks K and G after someone flipped the flag
/// with set_parameter — while no callback exists and the tangent is permanently
/// stale. The accessor must describe what was actually bound.
TEST(TangentGenerator, RecomputeIsReadOnceAndTheAccessorCannotLie) {
  ctx_type ctx;
  build_decomposed(ctx, K, G, /*recompute=*/false);

  auto* typed =
      dynamic_cast<nm::isotropic_tangent<policy>*>(ctx.find("stiffness"));
  ASSERT_NE(typed, nullptr);
  ASSERT_FALSE(typed->recomputes());

  // Flipping the parameter afterwards cannot bind a callback...
  typed->template set_parameter<bool>("recompute", true);

  // ... so the accessor must still report false,
  EXPECT_FALSE(typed->recomputes())
      << "recomputes() must report what was bound, not the live parameter";
  // ... and the property must still carry no callback.
  const auto* prop = ctx.find_property("stiffness", "tangent");
  ASSERT_NE(prop, nullptr);
  EXPECT_FALSE(static_cast<bool>(prop->traits().update));
}

// ---------------------------------------------------------------------------
// Composition with a pre-existing consumer
// ---------------------------------------------------------------------------

/// The header claims a tangent producer is a drop-in for any consumer. That has
/// to be tested against a consumer that already existed, not only against two
/// contexts built here. small_strain_plasticity takes its tangent from a NAMED
/// source, so it needs no change: point elastic_source at the generator.
TEST(TangentGenerator, DrivesJ2PlasticityIdenticallyToLinearElasticity) {
  auto drive = [](bool decomposed) {
    ctx_type ctx;
    param_type p;
    p.insert<std::string>("name", "strain_in");
    auto& src = ctx.create<nm::external_strain_source<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "stiffness");
    if (decomposed) {
      p.insert<T>("K", K);
      p.insert<T>("G", G);
      p.insert<bool>("recompute", false);
      ctx.create<nm::isotropic_tangent<policy>>(p);
    } else {
      p.insert<std::string>("strain_producer_name", "strain_in");
      p.insert<T>("K", K);
      p.insert<T>("G", G);
      ctx.create<nm::linear_elasticity<policy>>(p);
    }

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
    p.insert<std::string>("elastic_source", "stiffness");
    p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "strain_in");
    p.insert<std::string>("solver_source", "solver");
    p.insert<T>("G", G);
    p.insert<T>("sigma_0", T{50});
    ctx.create<nm::j2_plasticity<policy>>(p);
    ctx.finalize();

    std::vector<T> out;
    for (int step = 1; step <= 30; ++step) {
      const auto eps = uniaxial(0.02 * step);   // enough to pass sigma_0 = 50
      src.bind(eps, eps);
      ctx.update();
      out.push_back(ctx.get<tensor2>("j2", "stress")(0, 0));
      out.push_back(ctx.get<T>("j2", "equivalent_plastic_strain"));
      ctx.commit();
    }
    return out;
  };

  const auto with_generator = drive(true);
  const auto with_monolith = drive(false);
  ASSERT_EQ(with_generator.size(), with_monolith.size());
  bool went_plastic = false;
  for (std::size_t i = 0; i < with_generator.size(); ++i) {
    EXPECT_DOUBLE_EQ(with_generator[i], with_monolith[i]) << "sample " << i;
    if (i % 2 == 1 && with_generator[i] > 1e-8) went_plastic = true;
  }
  EXPECT_TRUE(went_plastic) << "the path must yield for this to mean anything";
}

}  // namespace
