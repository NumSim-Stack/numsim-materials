// Finite-strain interface: external_deformation_gradient_source driving the
// hyperelastic models, their tangents against finite differences, the
// small-strain limit, and the evaluator's deformation-gradient path (from C++
// and from a JSON document).
#include <gtest/gtest.h>
#include <tmech/tmech.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <vector>

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/neo_hooke.h"
#include "numsim-materials/materials/saint_venant_kirchhoff.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/io/json_material_factory.h"
#include "numsim-materials/umat/json_model.h"
#include "numsim-materials/umat/material_point_evaluator.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

constexpr T K = 0.833;
constexpr T G = 0.386;

template <typename Model>
nm::external_deformation_gradient_source<policy>& add_model(ctx_type& ctx) {
  param_type p;
  p.insert<std::string>("name", "F");
  auto& src = ctx.create<nm::external_deformation_gradient_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "model");
  p.insert<std::string>("deformation_gradient_source", "F");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  ctx.create<Model>(p);
  ctx.finalize();
  return src;
}

tensor2 test_deformation() {
  tensor2 F{tmech::eye<T, 3, 2>()};
  F(0, 1) = 0.3;
  F(1, 1) = 1.1;
  F(2, 0) = -0.2;
  F(2, 2) = 0.95;
  return F;
}

template <typename Model>
void expect_tangent_matches_finite_differences(ctx_type& ctx,
                                               nm::external_deformation_gradient_source<policy>& src) {
  const auto& P = ctx.template get<tensor2>("model", "stress");
  const auto& A = ctx.template get<tensor4>("model", "tangent");
  const tensor2 F{test_deformation()};
  src.bind(F, F);
  ctx.update();
  const tensor4 A0{A};
  const T h{1e-6};
  for (std::size_t k = 0; k < 3; ++k)
    for (std::size_t l = 0; l < 3; ++l) {
      tensor2 Fp{F}, Fm{F};
      Fp(k, l) += h;
      Fm(k, l) -= h;
      src.bind(Fp, Fp);
      ctx.update();
      const tensor2 Pp{P};
      src.bind(Fm, Fm);
      ctx.update();
      const tensor2 Pm{P};
      for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
          EXPECT_NEAR(A0(i, j, k, l), (Pp(i, j) - Pm(i, j)) / (2 * h), 1e-7)
              << "dP_" << i << j << "/dF_" << k << l;
    }
}

tensor4 isotropic_stiffness() {
  const auto I{tmech::eye<T, 3, 2>()};
  const auto IIsym{(tmech::otimesu(I, I) + tmech::otimesl(I, I)) * 0.5};
  const auto IIvol{tmech::otimes(I, I) / 3.0};
  return tensor4{3 * K * IIvol + 2 * G * (IIsym - IIvol)};
}

}  // namespace

TEST(SaintVenantKirchhoff, TangentMatchesFiniteDifferences) {
  ctx_type ctx;
  auto& src = add_model<nm::saint_venant_kirchhoff<policy>>(ctx);
  expect_tangent_matches_finite_differences<nm::saint_venant_kirchhoff<policy>>(ctx, src);
}

TEST(NeoHooke, TangentMatchesFiniteDifferences) {
  ctx_type ctx;
  auto& src = add_model<nm::neo_hooke<policy>>(ctx);
  expect_tangent_matches_finite_differences<nm::neo_hooke<policy>>(ctx, src);
}

TEST(NeoHooke, IsStressFreeAtIdentityAndLinearElasticAtSmallStrain) {
  ctx_type ctx;
  auto& src = add_model<nm::neo_hooke<policy>>(ctx);
  const auto& P = ctx.get<tensor2>("model", "stress");
  const auto& A = ctx.get<tensor4>("model", "tangent");
  const tensor2 I{tmech::eye<T, 3, 2>()};
  src.bind(I, I);
  ctx.update();
  EXPECT_LT(tmech::norm(tensor2{P}), 1e-14);
  // at F = I the tangent is the small-strain stiffness acting on symmetric
  // increments: A : sym(dF) = C : sym(dF) for every dF
  const tensor4 C{isotropic_stiffness()};
  tensor2 dF;
  dF(0, 1) = 0.7;
  dF(1, 0) = -0.2;
  dF(2, 2) = 0.4;
  const tensor2 sym_dF{0.5 * (dF + tmech::trans(dF))};
  const tensor2 lhs{tmech::dcontract(tensor4{A}, sym_dF)};
  const tensor2 rhs{tmech::dcontract(C, sym_dF)};
  EXPECT_LT(tmech::norm(tensor2{lhs - rhs}), 1e-12);
  // a small deformation of size e: P agrees with the linear law up to O(e^2)
  tensor2 F{I};
  F(0, 1) = 1e-4;
  src.bind(F, F);
  ctx.update();
  const tensor2 eps{0.5 * (F + tmech::trans(F)) - I};
  const tensor2 sigma{tmech::dcontract(C, eps)};
  EXPECT_LT(tmech::norm(tensor2{P - sigma}), 1e-3 * tmech::norm(sigma));
}

TEST(SaintVenantKirchhoff, ReducesToLinearElasticityAtSmallStrain) {
  ctx_type ctx;
  auto& src = add_model<nm::saint_venant_kirchhoff<policy>>(ctx);
  const auto& P = ctx.get<tensor2>("model", "stress");
  const tensor2 I{tmech::eye<T, 3, 2>()};
  tensor2 F{I};
  F(0, 1) = 1e-4;
  F(2, 2) = 1 + 1e-4;
  src.bind(F, F);
  ctx.update();
  const tensor2 eps{0.5 * (F + tmech::trans(F)) - I};
  const tensor2 sigma{tmech::dcontract(isotropic_stiffness(), eps)};
  EXPECT_LT(tmech::norm(tensor2{P - sigma}), 1e-3 * tmech::norm(sigma));  // O(e^2) terms
}

TEST(MaterialPointEvaluator, DeformationGradientPathReturnsPAndItsTangent) {
  ctx_type ctx;
  add_model<nm::saint_venant_kirchhoff<policy>>(ctx);
  u::material_point_evaluator<policy> ev{
      ctx, {.strain_source = {}, .stress_source = "model", .deformation_gradient_source = "F"}};
  EXPECT_TRUE(ev.finite_strain());
  EXPECT_EQ(ev.nstatv(), 0u);  // the source's history is host-owned
  const tensor2 F{test_deformation()};
  tensor2 P;
  tensor4 A;
  ev.evaluate_deformation_gradient(nullptr, tmech::eye<T, 3, 2>(), F, 0.0, 1.0, P, A);
  // the same from the graph directly
  ctx_type direct;
  auto& src = add_model<nm::saint_venant_kirchhoff<policy>>(direct);
  src.bind(F, F);
  direct.update();
  EXPECT_LT(tmech::norm(tensor2{P - direct.get<tensor2>("model", "stress")}), 1e-14);
  // and no small-strain call for a finite-strain model
  T stran[6]{}, dstran[6]{}, stress[6]{}, ddsdde[36]{};
  std::vector<T> statev;
  EXPECT_THROW(ev.evaluate({.stran = stran, .dstran = dstran, .stress = stress, .ddsdde = ddsdde,
                            .statev = statev}),
               u::fatal_error);
}

TEST(MaterialPointEvaluator, NeedsExactlyOneOfStrainAndDeformationGradientSource) {
  ctx_type ctx;
  add_model<nm::neo_hooke<policy>>(ctx);
  EXPECT_THROW((u::material_point_evaluator<policy>{ctx, {.strain_source = {}, .stress_source = "model"}}),
               u::fatal_error);
}

TEST(JsonModel, BuildsAFiniteStrainModel) {
  const nlohmann::json doc = nlohmann::json::parse(R"({
    "materials": [
      {"type": "external_deformation_gradient_source", "name": "F"},
      {"type": "neo_hooke", "name": "model", "deformation_gradient_source": "F",
       "K": 0.833, "G": 0.386}
    ]
  })");
  u::ensure_materials_registered<policy>();
  ctx_type ctx;
  for (const auto& entry : doc["materials"]) nm::create_from_json<policy>(ctx, entry);
  ctx.finalize();
  u::material_point_evaluator<policy> ev{
      ctx, {.strain_source = {}, .stress_source = "model", .deformation_gradient_source = "F"}};
  tensor2 F{tmech::eye<T, 3, 2>()};
  F(0, 1) = 0.5;
  F(1, 1) = 1.2;
  tensor2 P;
  tensor4 A;
  ev.evaluate_deformation_gradient(nullptr, tmech::eye<T, 3, 2>(), F, 0.0, 1.0, P, A);
  EXPECT_GT(std::abs(P(0, 1)), 0.1);
  EXPECT_NE(P(0, 1), P(1, 0));  // P is not symmetric ...
  const tensor2 PFt{P * tmech::trans(F)};  // ... but P F^T (J sigma) is
  EXPECT_LT(tmech::norm(tensor2{PFt - tmech::trans(PFt)}), 1e-14);
}
