#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/isotropic_tangent.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/small_strain_plasticity.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/material_point_evaluator.h"
#include "numsim-materials/umat/statev_map.h"
#include "numsim-materials/umat/tensor_conversion.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

constexpr T K = 166.67;
constexpr T G = 76.92;

/// strain source + constant moduli -> isotropic_tangent -> linear_stress
nm::external_strain_source<policy>& build_decomposed(ctx_type& ctx, T k, T g) {
  param_type p;
  p.insert<std::string>("name", "strain_in");
  auto& src = ctx.create<nm::external_strain_source<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "K");
  p.insert<T>("value", k);
  ctx.create<nm::constant_scalar<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "G");
  p.insert<T>("value", g);
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
  return src;
}

/// v * (e1 (x) e1) — a uniaxial strain, built as a tensor expression.
tensor2 uniaxial(T v) {
  tmech::tensor<T, 3, 1> e1;
  e1.fill(0.0);
  e1(0) = 1.0;
  tensor2 out;
  out = v * tmech::otimes(e1, e1);
  return out;
}

/// The isotropic stiffness the generator should produce, built independently.
tensor4 isotropic(T k, T g) {
  const auto I = tmech::eye<T, 3, 2>();
  const auto IIsym = (tmech::otimesu(I, I) + tmech::otimesl(I, I)) * 0.5;
  const auto IIvol = tmech::otimes(I, I) / 3.0;
  tensor4 C;
  C = 3.0 * k * IIvol + 2.0 * g * (IIsym - IIvol);
  return C;
}

/// Exact tensor equality, expressed through tmech rather than a component loop.
/// norm(a - b) is identically zero only when every component matches.
template <typename A, typename B>
::testing::AssertionResult TensorsIdentical(const A& a, const B& b) {
  const auto d = tmech::norm(a - b);
  if (d == T{0}) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << "norm(a - b) = " << d;
}

// ---------------------------------------------------------------------------
// Ordering — what the decomposition exists for
// ---------------------------------------------------------------------------

/// A cross-material property is ordered before its consumer; an intra-material
/// one is not. Both the moduli and the stiffness are edges here.
TEST(TangentGenerator, EveryProducerIsOrderedBeforeItsConsumer) {
  ctx_type ctx;
  build_decomposed(ctx, K, G);

  // optional, not 0: a producer legitimately lands at index 0, so a 0 sentinel
  // would pass even with the property missing from the graph.
  std::optional<std::size_t> i_k, i_g, i_tangent, i_stress;
  std::size_t n = 0;
  for (const auto* prop : ctx.property_execution_order()) {
    const auto& id = prop->traits().id;
    if (id.owner == "K" && id.name == "value") i_k = n;
    if (id.owner == "G" && id.name == "value") i_g = n;
    if (id.owner == "stiffness" && id.name == "tangent") i_tangent = n;
    if (id.owner == "elastic" && id.name == "stress") i_stress = n;
    ++n;
  }
  ASSERT_TRUE(i_k.has_value() && i_g.has_value());
  ASSERT_TRUE(i_tangent.has_value()) << "stiffness::tangent is not in the graph";
  ASSERT_TRUE(i_stress.has_value()) << "elastic::stress is not in the graph";

  EXPECT_LT(*i_k, *i_tangent);
  EXPECT_LT(*i_g, *i_tangent);
  EXPECT_LT(*i_tangent, *i_stress);
}

// ---------------------------------------------------------------------------
// Equivalence with the monolithic material
// ---------------------------------------------------------------------------

/// Same physics, different graph shape: must agree exactly.
TEST(TangentGenerator, MatchesLinearElasticityExactly) {
  ctx_type dec;
  auto& dec_src = build_decomposed(dec, K, G);

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

    EXPECT_TRUE(TensorsIdentical(dec.get<tensor2>("elastic", "stress"),
                                 mono.get<tensor2>("elastic", "stress")))
        << "step " << step;
  }

  EXPECT_TRUE(TensorsIdentical(dec.get<tensor4>("stiffness", "tangent"),
                               mono.get<tensor4>("elastic", "tangent")));
}

// ---------------------------------------------------------------------------
// Constants as materials
// ---------------------------------------------------------------------------

/// constant_scalar publishes a PLAIN property, so it costs no STATEV slot and
/// needs no exclusion. A history property here would be one wasted slot per
/// integration point, per constant.
TEST(TangentGenerator, ConstantsCostNoStatevSlot) {
  ctx_type ctx;
  build_decomposed(ctx, K, G);
  // Only the host-driven strain is excluded; the moduli are not mentioned.
  const u::statev_map<policy> map(ctx, {{"strain_in", "strain"}});
  EXPECT_EQ(map.nstatv(), 0u);
}

/// Inputs are not wired until finalize(), so nothing can be computed in the
/// constructor: the stiffness is built on the first update, not before it.
/// Everything that reads it goes through ctx.update() first, but the ordering
/// is worth pinning because it differs from the parameter-based version.
TEST(TangentGenerator, TangentIsBuiltOnTheFirstUpdateNotAtConstruction) {
  ctx_type ctx;
  auto& src = build_decomposed(ctx, K, G);

  // Default-constructed until something runs the callback.
  tensor4 zero;
  zero.fill(0.0);
  EXPECT_TRUE(TensorsIdentical(ctx.get<tensor4>("stiffness", "tangent"), zero));

  const auto eps = uniaxial(0.001);
  src.bind(eps, eps);
  ctx.update();

  EXPECT_TRUE(tmech::almost_equal(ctx.get<tensor4>("stiffness", "tangent"),
                                  isotropic(K, G), 1e-12));
}

/// Repeated updates with fixed moduli must keep giving the same answer. The
/// self-guard makes that cheap, but it is an optimisation with no observable
/// behaviour — this checks the observable part.
TEST(TangentGenerator, RepeatedUpdatesWithFixedModuliAreStable) {
  ctx_type ctx;
  auto& src = build_decomposed(ctx, K, G);
  const auto eps = uniaxial(0.001);
  src.bind(eps, eps);
  ctx.update();
  const tensor2 first = ctx.get<tensor2>("elastic", "stress");
  for (int i = 0; i < 20; ++i) ctx.update();
  EXPECT_TRUE(TensorsIdentical(ctx.get<tensor2>("elastic", "stress"), first));
}

/// And when a modulus does move, the stiffness follows on the next update —
/// with the ordering guaranteed by the edge, not by registration order.
TEST(TangentGenerator, StiffnessFollowsAChangedModulus) {
  ctx_type ctx;
  auto& src = build_decomposed(ctx, K, G);

  const auto eps = uniaxial(0.001);
  src.bind(eps, eps);
  ctx.update();
  tensor2 expected;
  expected = tmech::dcontract(isotropic(K, G), eps);
  EXPECT_TRUE(tmech::almost_equal(ctx.get<tensor2>("elastic", "stress"),
                                  expected, 1e-12));

  // Write the constant material's published value directly.
  ctx.get_mutable<T>("K", "value") = 2 * K;
  ctx.update();

  expected = tmech::dcontract(isotropic(2 * K, G), eps);
  EXPECT_TRUE(tmech::almost_equal(ctx.get<tensor2>("elastic", "stress"),
                                  expected, 1e-12));
}

// ---------------------------------------------------------------------------
// Composition with a pre-existing consumer
// ---------------------------------------------------------------------------

/// The drop-in claim, tested against a PRE-EXISTING consumer.
/// small_strain_plasticity already names its tangent source, so it needs no
/// change: point elastic_source at the generator.
TEST(TangentGenerator, DrivesJ2PlasticityIdenticallyToLinearElasticity) {
  auto drive = [](bool decomposed) {
    ctx_type ctx;
    param_type p;
    p.insert<std::string>("name", "strain_in");
    auto& src = ctx.create<nm::external_strain_source<policy>>(p);

    if (decomposed) {
      p.clear();
      p.insert<std::string>("name", "K");
      p.insert<T>("value", K);
      ctx.create<nm::constant_scalar<policy>>(p);
      p.clear();
      p.insert<std::string>("name", "G");
      p.insert<T>("value", G);
      ctx.create<nm::constant_scalar<policy>>(p);
      p.clear();
      p.insert<std::string>("name", "stiffness");
      p.insert<std::string>("K_source", "K");
      p.insert<std::string>("G_source", "G");
      ctx.create<nm::isotropic_tangent<policy>>(p);
    } else {
      p.clear();
      p.insert<std::string>("name", "stiffness");
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

    std::vector<std::pair<tensor2, T>> out;
    for (int step = 1; step <= 30; ++step) {
      const auto eps = uniaxial(0.02 * step);
      src.bind(eps, eps);
      ctx.update();
      out.emplace_back(ctx.get<tensor2>("j2", "stress"),
                       ctx.get<T>("j2", "equivalent_plastic_strain"));
      ctx.commit();
    }
    return out;
  };

  const auto with_generator = drive(true);
  const auto with_monolith = drive(false);
  ASSERT_EQ(with_generator.size(), with_monolith.size());
  bool went_plastic = false;
  for (std::size_t i = 0; i < with_generator.size(); ++i) {
    EXPECT_TRUE(TensorsIdentical(with_generator[i].first,
                                 with_monolith[i].first))
        << "stress at step " << i;
    EXPECT_DOUBLE_EQ(with_generator[i].second, with_monolith[i].second)
        << "equivalent plastic strain at step " << i;
    if (with_generator[i].second > 1e-8) went_plastic = true;
  }
  EXPECT_TRUE(went_plastic) << "the path must yield for this to mean anything";
}

// ---------------------------------------------------------------------------
// The optional tangent_source
// ---------------------------------------------------------------------------

/// Absent tangent_source falls back to the stress source; supplied is honoured.
TEST(TangentSource, AbsentFallsBackToTheStressSource) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "strain_in");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "strain_in");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  ctx.create<nm::linear_elasticity<policy>>(p);
  ctx.finalize();

  u::material_point_evaluator<policy>::config cfg;
  cfg.strain_source = "strain_in";
  cfg.stress_source = "elastic";
  ASSERT_FALSE(cfg.tangent_source.has_value());
  EXPECT_NO_THROW(u::material_point_evaluator<policy>(ctx, cfg));
}

TEST(TangentSource, SuppliedResolvesTheTangentElsewhere) {
  ctx_type ctx;
  build_decomposed(ctx, K, G);

  u::material_point_evaluator<policy>::config cfg;
  cfg.strain_source = "strain_in";
  cfg.stress_source = "elastic";  // linear_stress publishes only "stress"
  EXPECT_THROW(u::material_point_evaluator<policy>(ctx, cfg), u::fatal_error);

  cfg.tangent_source = "stiffness";
  EXPECT_NO_THROW(u::material_point_evaluator<policy>(ctx, cfg));
}

/// The decomposed pair driving a UMAT end to end.
TEST(TangentSource, DecomposedPairDrivesTheEvaluatorLikeLinearElasticity) {
  ctx_type dec;
  build_decomposed(dec, K, G);
  u::material_point_evaluator<policy>::config dcfg;
  dcfg.strain_source = "strain_in";
  dcfg.stress_source = "elastic";
  dcfg.tangent_source = "stiffness";
  u::material_point_evaluator<policy> deval(dec, dcfg);

  ctx_type mono;
  {
    param_type p;
    p.insert<std::string>("name", "strain_in");
    mono.create<nm::external_strain_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "strain_in");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    mono.create<nm::linear_elasticity<policy>>(p);
    mono.finalize();
  }
  u::material_point_evaluator<policy>::config mcfg;
  mcfg.strain_source = "strain_in";
  mcfg.stress_source = "elastic";
  u::material_point_evaluator<policy> meval(mono, mcfg);

  std::vector<T> dsv(deval.nstatv(), 0.0), msv(meval.nstatv(), 0.0);
  T stran[6] = {0, 0, 0, 0, 0, 0};
  const T dstran[6] = {0.002, -0.0005, 0.0, 0.001, 0.0, 0.0};
  T ds[6], dd[36], ms[6], md[36];

  for (int step = 0; step < 10; ++step) {
    deval.evaluate({.stran = stran, .dstran = dstran, .stress = ds,
                    .ddsdde = dd, .statev = dsv});
    meval.evaluate({.stran = stran, .dstran = dstran, .stress = ms,
                    .ddsdde = md, .statev = msv});

    // Compare in tensor space rather than slot by slot: a slot permutation on
    // both sides would cancel in a componentwise check, and the tensors are
    // what the host actually consumes.
    EXPECT_TRUE(TensorsIdentical(u::stress_from_buffer<T>(ds),
                                 u::stress_from_buffer<T>(ms)))
        << "stress at step " << step;
    EXPECT_TRUE(TensorsIdentical(u::tangent_from_buffer<T>(dd),
                                 u::tangent_from_buffer<T>(md)))
        << "tangent at step " << step;

    for (std::size_t i = 0; i < 6; ++i) stran[i] += dstran[i];
  }
}

}  // namespace
