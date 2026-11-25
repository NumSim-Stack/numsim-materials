#include <gtest/gtest.h>
#include <memory>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/driver/material_point_driver.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/j2_plasticity.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/phase.h"
#include "numsim-materials/materials/mean_field_composite_global.h"
#include "numsim-materials/materials/weighted_sum.h"
#include "numsim-materials/solvers/local_newton.h"
#include "numsim-materials/driver/external_state_source.h"
#include "test_support.h"

/// The nonlinear Voigt rule of mixtures (equal strain in every phase) is a wiring, not a material:
/// every phase reads the composite strain and weighted_sum adds c_i sigma_i and c_i C_i.
namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using terms_type = nm::weighted_sum<policy>::terms_type;
using numsim::materials::test::add;

constexpr T K{166.67}, G{76.92}, sigma_0{0.5}, H{1000.0}, c_plastic{0.4};

std::unique_ptr<ctx_type> build() {
  auto owner{std::make_unique<ctx_type>()};
  auto &ctx{*owner};
  add<nm::external_strain_source<policy>>(ctx, "strain");
  add<nm::linear_elasticity<policy>>(ctx, "elastic", {{"strain_producer_name", "strain"}, {"K", K}, {"G", G}});
  add<nm::local_newton<policy>>(ctx, "solver");
  add<nm::linear_isotropic_hardening<policy>>(ctx, "hardening", {{"source", "j2"}, {"K", H}});
  add<nm::j2_plasticity<policy>>(ctx, "j2",
      {{"K", K}, {"G", G}, {"sigma_0", sigma_0}, {"hardening_source", "hardening"}, {"strain_source", "strain"},
       {"solver_source", "solver"}});
  for (auto const &[name, value] : {std::pair{"c_plastic", c_plastic}, std::pair{"c_elastic", 1 - c_plastic}}) {
    add<nm::constant_scalar<policy>>(ctx, name, {{"value", value}});
  }
  add<nm::weighted_sum<policy>>(ctx, "voigt", {{"terms", terms_type{{"c_plastic", "j2"}, {"c_elastic", "elastic"}}}});
  // the linear rule of mixtures over the same tangents: phases of produced properties, a tangent, a linear_stress
  for (auto const &[phase, material, fraction] : {std::tuple{"plastic_phase", "j2", "c_plastic"}, std::tuple{"elastic_phase", "elastic", "c_elastic"}}) {
    add<nm::phase<policy>>(ctx, phase, {{"tangent_source", material}, {"fraction_source", fraction}});
  }
  add<nm::mean_field_composite_global<policy>>(ctx, "linear_voigt",
      {{"phases", std::vector<std::string>{"plastic_phase", "elastic_phase"}},
       {"kernel", nm::object_spec<param_type>{"voigt", {}}}});
  add<nm::linear_stress<policy>>(ctx, "linear_voigt_stress",
      {{"tangent_source", "linear_voigt"}, {"strain_source", "strain"}});
  ctx.finalize();
  return owner;
}

TEST(VoigtComposite, StressIsTheFractionWeightedSumOfThePhaseStresses) {
  auto ctx{build()};
  param_type point;
  point.insert<std::string>("strain_source", "strain");
  point.insert<std::string>("stress_source", "voigt");
  nm::material_point_driver<policy> d(*ctx, point);
  tensor2 eps;
  for (int n{1}; n <= 20; ++n) {
    eps(0, 0) = n * 5e-4;
    auto const res{d.step(eps, tensor2{}, nm::load_control<T>::uniaxial_stress(), n - 1, 1)};
    ASSERT_TRUE(res.converged) << n;
    tensor2 const expected{c_plastic * ctx->get<tensor2>("j2", "stress") + (1 - c_plastic) * ctx->get<tensor2>("elastic", "stress")};
    EXPECT_NEAR(tmech::norm(tensor2{res.stress - expected}), 0, 1e-12) << n;
    EXPECT_LT(d.tangent_error(res), 1e-6) << n; // the Voigt tangent is consistent: sum c_i C_i
    d.accept();
  }
  EXPECT_GT(ctx->get<T>("j2", "equivalent_plastic_strain"), 0); // the plastic phase did yield
}

TEST(VoigtComposite, EqualsTheLinearRuleOfMixturesWhileElastic) {
  auto ctx{build()};
  tensor2 eps;
  eps(0, 0) = 1e-3;
  eps(1, 2) = eps(2, 1) = 5e-4;
  dynamic_cast<nm::external_strain_source<policy> *>(ctx->find("strain"))->bind(tensor2{}, eps);
  ctx->update();
  EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("voigt", "stress") - ctx->get<tensor2>("linear_voigt_stress", "stress")}), 0, 1e-12);
  EXPECT_NEAR(tmech::norm(tensor4{ctx->get<tensor4>("voigt", "tangent") - ctx->get<tensor4>("linear_voigt", "tangent")}), 0, 1e-12);
}

} // namespace
