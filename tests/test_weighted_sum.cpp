#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/isotropic_tangent.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/scalar_identity_weight.h"
#include "numsim-materials/materials/scalar_stepper.h"
#include "numsim-materials/materials/weighted_sum.h"
#include "numsim-materials/driver/external_state_source.h"
#include "test_support.h"

namespace {

namespace nm = numsim::materials;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using terms_type = std::vector<std::pair<std::string, std::string>>;
using numsim::materials::test::add;
using numsim::materials::test::uniaxial;

constexpr T KA = 100.0, GA = 40.0;
constexpr T KB = 300.0, GB = 140.0;

/// A weight in [0,1] published as "value", via scalar_identity_weight over a
/// scalar_stepper's history.
void add_weight(ctx_type& ctx, const std::string& name, T increment) {
  add<nm::scalar_stepper<policy>>(ctx, name + "_drv", {{"increment", increment}});
  add<nm::scalar_identity_weight<policy>>(ctx, name, {{"source", name + "_drv::state"}});
}

void add_monolithic(ctx_type& ctx, const std::string& name, T k, T g) {
  add<nm::linear_elasticity<policy>>(ctx, name, {{"strain_producer_name", "strain_in"}, {"K", k}, {"G", g}});
}

/// A constituent whose stiffness lives in its OWN material, so its stress and
/// tangent come from two different names.
void add_decomposed(ctx_type& ctx, const std::string& name, T k, T g) {
  add<nm::constant_scalar<policy>>(ctx, name + "_K", {{"value", k}});
  add<nm::constant_scalar<policy>>(ctx, name + "_G", {{"value", g}});
  add<nm::isotropic_tangent<policy>>(ctx, name + "_stiff", {{"K_source", name + "_K"}, {"G_source", name + "_G"}});
  add<nm::linear_stress<policy>>(ctx, name, {{"tangent_source", name + "_stiff"}, {"strain_source", "strain_in"}});
}

nm::external_strain_source<policy>& add_strain(ctx_type& ctx) {
  return add<nm::external_strain_source<policy>>(ctx, "strain_in");
}

// ---------------------------------------------------------------------------
// Baseline: the mixture rule itself
// ---------------------------------------------------------------------------

/// Two monolithic constituents at weights 0.25 and 0.5. Both stress and tangent
/// must be the weighted sum.
TEST(WeightedSum, SumsStressAndTangentByWeight) {
  ctx_type ctx;
  auto& src = add_strain(ctx);
  add_weight(ctx, "wA", 0.25);
  add_weight(ctx, "wB", 0.5);
  add_monolithic(ctx, "matA", KA, GA);
  add_monolithic(ctx, "matB", KB, GB);

  add<nm::weighted_sum<policy>>(ctx, "mix", {{"terms", terms_type{{"wA", "matA"}, {"wB", "matB"}}}});
  ctx.finalize();

  src.bind(uniaxial(0.001), uniaxial(0.001));
  ctx.update();

  const T cA = KA + 4.0 * GA / 3.0;
  const T cB = KB + 4.0 * GB / 3.0;
  EXPECT_NEAR(ctx.get<tensor2>("mix", "stress")(0, 0),
              (0.25 * cA + 0.5 * cB) * 0.001, 1e-10);
  EXPECT_NEAR(ctx.get<tensor4>("mix", "tangent")(0, 0, 0, 0),
              0.25 * cA + 0.5 * cB, 1e-9);
}

// ---------------------------------------------------------------------------
// tangent_sources
// ---------------------------------------------------------------------------

/// Absent: every term takes its tangent from the material producing its stress.
TEST(WeightedSum, AbsentTangentSourcesUsesEachTermsOwnMaterial) {
  ctx_type ctx;
  auto& src = add_strain(ctx);
  add_weight(ctx, "wA", 1.0);
  add_monolithic(ctx, "matA", KA, GA);

  add<nm::weighted_sum<policy>>(ctx, "mix", {{"terms", terms_type{{"wA", "matA"}}}});
  ctx.finalize();

  src.bind(uniaxial(0.001), uniaxial(0.001));
  ctx.update();
  EXPECT_NEAR(ctx.get<tensor4>("mix", "tangent")(0, 0, 0, 0),
              KA + 4.0 * GA / 3.0, 1e-9);
}

/// What tangent_sources exists for: a constituent whose stress and tangent have
/// different owners.
TEST(WeightedSum, OverridesOneTermsTangentOwner) {
  ctx_type ctx;
  auto& src = add_strain(ctx);
  add_weight(ctx, "wA", 1.0);
  add_decomposed(ctx, "matA", KA, GA);

  add<nm::weighted_sum<policy>>(ctx, "mix",
      {{"terms", terms_type{{"wA", "matA"}}}, {"tangent_sources", std::vector<std::string>{"matA_stiff"}}});
  ctx.finalize();

  src.bind(uniaxial(0.001), uniaxial(0.001));
  ctx.update();
  EXPECT_NEAR(ctx.get<tensor4>("mix", "tangent")(0, 0, 0, 0),
              KA + 4.0 * GA / 3.0, 1e-9);
}

/// An empty entry keeps that term's own tangent, so a NON-LEADING term can be
/// overridden alone; without it a positional list could only override a prefix.
TEST(WeightedSum, EmptyEntryKeepsATermsOwnTangent) {
  ctx_type ctx;
  auto& src = add_strain(ctx);
  add_weight(ctx, "wA", 0.5);
  add_weight(ctx, "wB", 0.5);
  add_monolithic(ctx, "matA", KA, GA);      // keeps its own tangent
  add_decomposed(ctx, "matB", KB, GB);      // tangent lives elsewhere

  param_type p;
  p.insert<std::string>("name", "mix");
  p.insert<terms_type>("terms", {{"wA", "matA"}, {"wB", "matB"}});
  p.insert<std::vector<std::string>>("tangent_sources", {"", "matB_stiff"});
  ctx.create<nm::weighted_sum<policy>>(p);
  ctx.finalize();

  src.bind(uniaxial(0.001), uniaxial(0.001));
  ctx.update();

  const T cA = KA + 4.0 * GA / 3.0;
  const T cB = KB + 4.0 * GB / 3.0;
  EXPECT_NEAR(ctx.get<tensor4>("mix", "tangent")(0, 0, 0, 0),
              0.5 * cA + 0.5 * cB, 1e-9);
}

/// A shorter list is rejected: positional matching would apply the override to
/// the WRONG constituent, and both names still resolve. The only symptom would
/// be a wrong summed tangent — slow Newton, correct stresses.
TEST(WeightedSum, RejectsATangentSourcesListThatDoesNotMatchTheTermCount) {
  auto build = [](std::vector<std::string> sources) {
    ctx_type ctx;
    add_strain(ctx);
    add_weight(ctx, "wA", 0.5);
    add_weight(ctx, "wB", 0.5);
    add_monolithic(ctx, "matA", KA, GA);
    add_decomposed(ctx, "matB", KB, GB);
    add<nm::weighted_sum<policy>>(ctx, "mix",
        {{"terms", terms_type{{"wA", "matA"}, {"wB", "matB"}}},
         {"tangent_sources", std::vector<std::string>(std::move(sources))}});
  };

  EXPECT_THROW(build({"matB_stiff"}), std::runtime_error);          // too short
  EXPECT_THROW(build({"", "matB_stiff", "extra"}), std::runtime_error);  // long
  EXPECT_NO_THROW(build({"", "matB_stiff"}));                       // exact
}

}  // namespace
