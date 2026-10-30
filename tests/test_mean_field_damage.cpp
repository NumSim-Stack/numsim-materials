#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/errors.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/eshelby_sphere.h"
#include "numsim-materials/materials/exponential_damage_law.h"
#include "numsim-materials/materials/isotropic_damage.h"
#include "numsim-materials/materials/isotropic_moduli.h"
#include "numsim-materials/materials/isotropic_tangent.h"
#include "numsim-materials/materials/j2_plasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/mean_field_strain.h"
#include "numsim-materials/materials/mean_field_composite_local_linear.h"
#include "numsim-materials/materials/mean_field_composite_local_nonlinear.h"
#include "numsim-materials/materials/phase.h"
#include "numsim-materials/materials/strain_threshold_yield.h"
#include "numsim-materials/materials/von_mises_state_function.h"
#include "numsim-materials/solvers/local_newton.h"
#include "numsim-materials/driver/material_point_driver.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/umat/json_model.h"
#include "numsim-materials/umat/material_point_evaluator.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

constexpr T Km{3.0}, Gm{1.0}, Ki{60.0}, Gi{30.0}, c{0.3}, kappa_0{0.002}, beta{300.0};

void constant(ctx_type &ctx, std::string const &name, T value) {
  param_type p;
  p.insert<std::string>("name", name);
  p.insert<T>("value", value);
  ctx.create<nm::constant_scalar<policy>>(p);
}

/// Stiff elastic spheres in a matrix evaluated at its local strain: linear (B1, every edge global) or a
/// damage chain (B2, the matrix phase reads the algorithmic tangent over a local edge). Isotropic damage
/// keeps Poisson's ratio, so the Eshelby node reads nu of the undamaged matrix.
std::unique_ptr<ctx_type> build(bool nonlinear, std::string const &kernel = "mori_tanaka") {
  auto owner{std::make_unique<ctx_type>()};
  auto &ctx{*owner};
  param_type p;
  p.insert<std::string>("name", "strain");
  ctx.create<nm::external_strain_source<policy>>(p);
  for (auto const &[name, K, G] : {std::tuple{"m", Km, Gm}, std::tuple{"i", Ki, Gi}}) {
    constant(ctx, std::string{"K_"} + name, K);
    constant(ctx, std::string{"G_"} + name, G);
    p.clear();
    p.insert<std::string>("name", std::string{"C_"} + name);
    p.insert<std::string>("K_source", std::string{"K_"} + name);
    p.insert<std::string>("G_source", std::string{"G_"} + name);
    ctx.create<nm::isotropic_tangent<policy>>(p);
  }
  p.clear();
  p.insert<std::string>("name", "moduli");
  p.insert<std::string>("tangent_source", "C_m");
  ctx.create<nm::isotropic_moduli<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "sphere");
  p.insert<std::string>("nu_source", "moduli");
  p.insert<std::string>("nu_property", "poisson_ratio");
  ctx.create<nm::eshelby_sphere<policy>>(p);
  constant(ctx, "c", c);
  constant(ctx, "cm", 1 - c);
  p.clear(); // the matrix phase: the algorithmic tangent of the previous pass closes the cycle
  p.insert<std::string>("name", "matrix");
  p.insert<std::string>("tangent_source", nonlinear ? "damage" : "C_m");
  p.insert<std::string>("fraction_source", "cm");
  if (nonlinear) {
    p.insert<std::string>("tangent_property", "damaged_tangent");
    p.insert<std::string>("tangent_edge", "local");
    p.insert<std::string>("initial_tangent_source", "C_m");
  }
  ctx.create<nm::phase<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "spheres");
  p.insert<std::string>("tangent_source", "C_i");
  p.insert<std::string>("fraction_source", "c");
  p.insert<std::string>("eshelby_source", "sphere");
  ctx.create<nm::phase<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "mt_strain");
  p.insert<std::string>("strain_source", "strain");
  p.insert<std::vector<std::string>>("phases", {"matrix", "spheres"});
  p.insert<nm::object_spec<param_type>>("kernel", {kernel, {}});
  ctx.create<nm::mean_field_strain<policy>>(p);
  // the phase materials at their local strains
  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "mt_strain");
  p.insert<std::string>("strain_property", "strain_matrix");
  p.insert<T>("K", Km);
  p.insert<T>("G", Gm);
  ctx.create<nm::linear_elasticity<policy>>(p);
  if (nonlinear) {
    p.clear();
    p.insert<std::string>("name", "state");
    p.insert<std::string>("strain_source", "mt_strain");
    p.insert<std::string>("strain_property", "strain_matrix");
    ctx.create<nm::von_mises_state_function<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "yield");
    p.insert<std::string>("state_source", "state");
    p.insert<T>("kappa_critical", kappa_0);
    ctx.create<nm::strain_threshold_yield<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "law");
    p.insert<std::string>("yield_source", "yield");
    p.insert<T>("kappa_0", kappa_0);
    p.insert<T>("beta", beta);
    ctx.create<nm::exponential_damage_law<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "damage");
    p.insert<std::string>("elastic_source", "elastic");
    p.insert<std::string>("damage_source", "law");
    p.insert<std::string>("state_source", "state");
    p.insert<std::string>("yield_source", "yield");
    ctx.create<nm::isotropic_damage<policy>>(p);
  }
  p.clear();
  p.insert<std::string>("name", "stiff");
  p.insert<std::string>("tangent_source", "C_i");
  p.insert<std::string>("strain_source", "mt_strain");
  p.insert<std::string>("strain_property", "strain_spheres");
  ctx.create<nm::linear_stress<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "mt");
  nm::phase_sources phases{{{"concentration", "mt_strain::concentration_matrix"},
                            {"stress", nonlinear ? "damage::damaged_stress" : "elastic::stress"},
                            {"tangent", nonlinear ? "damage::damaged_tangent" : "elastic::tangent"},
                            {"fraction", "matrix::volume_fraction"}},
                           {{"concentration", "mt_strain::concentration_spheres"}, {"stress", "stiff::stress"},
                            {"tangent", "C_i::tangent"}, {"fraction", "spheres::volume_fraction"}}};
  if (nonlinear)
    phases[0]["used_tangent"] = "matrix::tangent";
  p.insert<nm::phase_sources>("phases", phases);
  if (nonlinear)
    ctx.create<nm::mean_field_composite_local_nonlinear<policy>>(p);
  else // a converging composite needs the local edge
    ctx.create<nm::mean_field_composite_local_linear<policy>>(p);
  ctx.finalize();
  return owner;
}

tensor2 uniaxial(T e) {
  tensor2 eps;
  eps(0, 0) = e;
  return eps;
}

/// Loads the composite in equal steps to the strain `eps`, committing every step: each update() repeats the
/// passes until the phase tangents the strains were formed with agree with the materials' (converges()).
void load(ctx_type &ctx, tensor2 const &eps, int steps) {
  auto &strain{*dynamic_cast<nm::external_strain_source<policy> *>(ctx.find("strain"))};
  for (int n{1}; n <= steps; ++n) {
    strain.bind(tensor2{eps * T(n - 1) / T(steps)}, tensor2{eps * T(n) / T(steps)});
    ctx.update();
    ctx.commit();
  }
}


TEST(MeanFieldDamage, BelowTheThresholdTheDamageChainReproducesTheLinearComposite) {
  auto nonlinear{build(true)}, linear{build(false)};
  tensor2 const eps{uniaxial(1e-4)}; // the matrix strain stays below kappa_0
  load(*nonlinear, eps, 4);
  load(*linear, eps, 1);
  EXPECT_NEAR(tmech::norm(tensor2{nonlinear->get<tensor2>("mt", "stress") - linear->get<tensor2>("mt", "stress")}) / tmech::norm(linear->get<tensor2>("mt", "stress")), 0, 1e-10);
  EXPECT_NEAR(tmech::norm(tensor4{nonlinear->get<tensor4>("mt", "tangent") - linear->get<tensor4>("mt", "tangent")}) / tmech::norm(linear->get<tensor4>("mt", "tangent")), 0, 1e-10);
  EXPECT_NEAR(tmech::norm(tensor2{nonlinear->get<tensor2>("mt_strain", "strain_matrix") - linear->get<tensor2>("mt_strain", "strain_matrix")}), 0, 1e-14);
  EXPECT_DOUBLE_EQ(nonlinear->get<T>("law", "damage"), 0);
}

TEST(MeanFieldDamage, TheLocalMatrixStrainDrivesTheDamage) {
  auto ctx{build(true)};
  tensor2 const eps{uniaxial(0.01)};
  load(*ctx, eps, 20);
  auto const &eps_m{ctx->get<tensor2>("mt_strain", "strain_matrix")};
  auto const &eps_i{ctx->get<tensor2>("mt_strain", "strain_spheres")};
  // the local strains partition the composite strain and the phases were evaluated at them
  EXPECT_NEAR(tmech::norm(tensor2{(1 - c) * eps_m + c * eps_i - eps}), 0, 1e-12);
  tensor2 const dev_m{tmech::dev(eps_m)};
  EXPECT_NEAR(ctx->get<T>("state", "equivalent_strain"), std::sqrt(2.0 / 3 * tmech::dcontract(dev_m, dev_m)), 1e-12); // the state function saw the matrix strain
  EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("stiff", "stress") - tmech::dcontract(ctx->get<tensor4>("C_i", "tangent"), eps_i)}), 0, 1e-12);
  EXPECT_LT(ctx->get<T>("mt", "residual"), 1e-10); // the pass is converged: the strains were formed with the final tangents
  // the matrix softened, the stiff spheres carry more than their share, the stress is the phase average
  T const D{ctx->get<T>("law", "damage")};
  EXPECT_GT(D, 0.3);
  EXPECT_LT(D, 1);
  EXPECT_GT(eps_m(0, 0), eps(0, 0));
  EXPECT_LT(eps_i(0, 0), eps(0, 0));
  tensor2 const average{(1 - c) * ctx->get<tensor2>("damage", "damaged_stress") + c * ctx->get<tensor2>("stiff", "stress")};
  EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("mt", "stress") - average}) / tmech::norm(average), 0, 1e-12);
  auto undamaged{build(false)};
  load(*undamaged, uniaxial(1e-4), 1);
  EXPECT_LT(ctx->get<tensor2>("mt", "stress")(0, 0), tmech::dcontract(undamaged->get<tensor4>("mt", "tangent"), eps)(0, 0));
}

TEST(MeanFieldDamage, TangentIsTheConcentrationWeightedPhaseTangent) {
  auto ctx{build(true)};
  load(*ctx, uniaxial(0.006), 6);
  tensor4 const expected{(1 - c) * tmech::dcontract(ctx->get<tensor4>("damage", "damaged_tangent"), ctx->get<tensor4>("mt_strain", "concentration_matrix")) +
                         c * tmech::dcontract(ctx->get<tensor4>("C_i", "tangent"), ctx->get<tensor4>("mt_strain", "concentration_spheres"))};
  EXPECT_NEAR(tmech::norm(tensor4{ctx->get<tensor4>("mt", "tangent") - expected}) / tmech::norm(expected), 0, 1e-12);
}

TEST(MeanFieldDamage, ResidualDecreasesOverThePassesOfAStep) {
  auto ctx{build(true)};
  load(*ctx, uniaxial(0.004), 4);
  ctx->find("mt")->set_parameter<T>("tolerance", std::numeric_limits<T>::infinity()); // one pass per update
  auto &strain{*dynamic_cast<nm::external_strain_source<policy> *>(ctx->find("strain"))};
  strain.bind(uniaxial(0.004), uniaxial(0.006)); // a large damaging step: several passes
  ctx->update();
  T const first{ctx->get<T>("mt", "residual")};
  EXPECT_GT(first, 1e-6);
  ctx->update();
  T const second{ctx->get<T>("mt", "residual")};
  EXPECT_LT(second, first);
  int passes{2};
  while (ctx->get<T>("mt", "residual") > 1e-10 && passes < 200) {
    ctx->update();
    ++passes;
  }
  EXPECT_LT(passes, 200);
  EXPECT_LT(ctx->get<T>("mt", "residual"), 1e-10);
}

/// The driver on "strain" -> "mt", closing the fixed point on "mt::residual".
policy::ParameterHandler mt_point() {
  policy::ParameterHandler p;
  p.insert<std::string>("strain_source", "strain");
  p.insert<std::string>("stress_source", "mt");
  p.insert<std::string>("residual_source", "mt::residual");
  return p;
}

// the tangent sum c_r C_r : A_r is exact as the step vanishes: central differences on a small step after
// loading through the driver, whose material point closes the fixed point per call
TEST(MeanFieldDamage, TangentMatchesFiniteDifferencesOnASmallStep) {
  auto ctx{build(true)};
  nm::material_point_driver<policy> d(*ctx, mt_point());
  for (int n{1}; n <= 6; ++n) {
    ASSERT_TRUE(d.step(uniaxial(n * 1e-3), tensor2{}, nm::load_control<T>::strain(), n - 1, 1).converged) << n;
    d.accept();
  }
  auto const res{d.step(uniaxial(0.006 + 1e-6), tensor2{}, nm::load_control<T>::strain(), 6, 1)};
  ASSERT_TRUE(res.converged);
  EXPECT_LT(d.tangent_error(res), 1e-2);
}

// the fixed point inside update() takes bitwise the passes of the material point's loop it replaces: there, the
// composite converges after every pass and residual_source repeats the whole update
TEST(MeanFieldDamage, TheEngineIteratesBitwiseAsTheMaterialPointDid) {
  for (auto const &control : {nm::load_control<T>::strain(), nm::load_control<T>::uniaxial_stress()}) {
    auto engine{build(true)}, host{build(true)};
    host->find("mt")->set_parameter<T>("tolerance", std::numeric_limits<T>::infinity());
    auto in_engine{mt_point()};
    in_engine.insert<std::string>("residual_source", std::string{});
    nm::material_point_driver<policy> e(*engine, in_engine), h(*host, mt_point());
    for (int n{1}; n <= 24; ++n) {
      T const eps{(n <= 16 ? n : 32 - n) * 5e-4}; // loading, then unloading
      auto const a{e.step(uniaxial(eps), tensor2{}, control, n - 1, 1)}, b{h.step(uniaxial(eps), tensor2{}, control, n - 1, 1)};
      ASSERT_TRUE(a.converged) << n;
      ASSERT_TRUE(b.converged) << n;
      EXPECT_EQ(a.residuals, b.residuals) << n;
      EXPECT_EQ(a.stress, b.stress) << n;
      EXPECT_EQ(engine->get<tensor4>("mt", "tangent"), host->get<tensor4>("mt", "tangent")) << n;
      EXPECT_EQ(engine->get<T>("law", "damage"), host->get<T>("law", "damage")) << n;
      EXPECT_EQ(engine->get<T>("mt", "residual"), host->get<T>("mt", "residual")) << n;
      e.accept();
      h.accept();
    }
    EXPECT_GT(engine->get<T>("law", "damage"), 0);
  }
}

// the UMAT closes the fixed point without residual_property: the composite converges in update()
TEST(MeanFieldDamage, TheUmatEvaluatorConvergesWithoutAResidualProperty) {
  auto ctx{build(true)}, reference{build(true)};
  nm::umat::material_point_evaluator<policy>::config cfg;
  cfg.strain_source = "strain";
  cfg.stress_source = "mt";
  nm::umat::material_point_evaluator<policy> eval(*ctx, cfg);
  std::vector<T> statev(eval.nstatv(), 0);
  T stran[6]{}, stress[6]{}, ddsdde[36]{};
  T const dstran[6]{1e-3, 0, 0, 0, 0, 0};
  for (int n{1}; n <= 6; ++n) {
    eval.evaluate({.stran = stran, .dstran = dstran, .stress = stress, .ddsdde = ddsdde, .statev = statev});
    EXPECT_LE(ctx->get<T>("mt", "residual"), 1e-10) << n;
    stran[0] += dstran[0];
  }
  load(*reference, uniaxial(0.006), 6);
  EXPECT_GT(reference->get<T>("law", "damage"), 0);
  T const expected{reference->get<tensor2>("mt", "stress")(0, 0)};
  EXPECT_NEAR(stress[0], expected, 1e-10 * std::abs(expected));
}

TEST(MeanFieldDamage, StressControlReEvaluatesFromTheCommittedLocalStrains) {
  auto ctx{build(true)};
  nm::material_point_driver<policy> d(*ctx, mt_point());
  for (int n{1}; n <= 16; ++n) {
    auto const res{d.step(uniaxial(n * 5e-4), tensor2{}, nm::load_control<T>::uniaxial_stress(), n - 1, 1)};
    ASSERT_TRUE(res.converged) << n;
    EXPECT_NEAR(res.stress(1, 1), 0, 1e-8 * std::abs(res.stress(0, 0))) << n;
    d.accept();
  }
  EXPECT_GT(ctx->get<T>("law", "damage"), 0);
}

/// A hardening nonlinear matrix (J2 with linear hardening) and stiff elastic spheres through the Reuss kernel.
std::unique_ptr<ctx_type> build_plastic() {
  constexpr T sigma_0{0.002 * 3 * Gm}, H{Gm};
  auto owner{std::make_unique<ctx_type>()};
  auto &ctx{*owner};
  param_type p;
  p.insert<std::string>("name", "strain");
  ctx.create<nm::external_strain_source<policy>>(p);
  for (auto const &[name, K, G] : {std::tuple{"m", Km, Gm}, std::tuple{"i", Ki, Gi}}) {
    constant(ctx, std::string{"K_"} + name, K);
    constant(ctx, std::string{"G_"} + name, G);
    p.clear();
    p.insert<std::string>("name", std::string{"C_"} + name);
    p.insert<std::string>("K_source", std::string{"K_"} + name);
    p.insert<std::string>("G_source", std::string{"G_"} + name);
    ctx.create<nm::isotropic_tangent<policy>>(p);
  }
  constant(ctx, "c", c);
  constant(ctx, "cm", 1 - c);
  p.clear();
  p.insert<std::string>("name", "matrix");
  p.insert<std::string>("tangent_source", "plastic");
  p.insert<std::string>("tangent_edge", "local");
  p.insert<std::string>("initial_tangent_source", "C_m");
  p.insert<std::string>("fraction_source", "cm");
  ctx.create<nm::phase<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "spheres");
  p.insert<std::string>("tangent_source", "C_i");
  p.insert<std::string>("fraction_source", "c");
  ctx.create<nm::phase<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "mt_strain");
  p.insert<std::string>("strain_source", "strain");
  p.insert<std::vector<std::string>>("phases", {"matrix", "spheres"});
  p.insert<nm::object_spec<param_type>>("kernel", {"reuss", {}});
  ctx.create<nm::mean_field_strain<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "solver");
  ctx.create<nm::local_newton<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "plastic");
  p.insert<T>("K", H);
  ctx.create<nm::linear_isotropic_hardening<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "plastic");
  p.insert<T>("K", Km);
  p.insert<T>("G", Gm);
  p.insert<T>("sigma_0", sigma_0);
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "mt_strain");
  p.insert<std::string>("strain_property", "strain_matrix");
  p.insert<std::string>("solver_source", "solver");
  ctx.create<nm::j2_plasticity<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "stiff");
  p.insert<std::string>("tangent_source", "C_i");
  p.insert<std::string>("strain_source", "mt_strain");
  p.insert<std::string>("strain_property", "strain_spheres");
  ctx.create<nm::linear_stress<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "mt");
  p.insert<nm::phase_sources>("phases", {{{"concentration", "mt_strain::concentration_matrix"}, {"stress", "plastic::stress"},
                                          {"tangent", "plastic::tangent"}, {"fraction", "matrix::volume_fraction"},
                                          {"used_tangent", "matrix::tangent"}},
                                         {{"concentration", "mt_strain::concentration_spheres"}, {"stress", "stiff::stress"},
                                          {"tangent", "C_i::tangent"}, {"fraction", "spheres::volume_fraction"}}});
  ctx.create<nm::mean_field_composite_local_nonlinear<policy>>(p);
  ctx.finalize();
  return owner;
}

// the Reuss kernel through the same passes: the concentrations equalise the stress increments of the
// algorithmic tangents, so with a hardening phase the stresses agree to first order in the step (a softening
// phase in series is a snap-back problem with no stable equal-stress path, see the docs)
TEST(MeanFieldDamage, ReussEqualisesThePhaseStressesAsTheStepVanishes) {
  auto mismatch = [](int steps) {
    auto ctx{build_plastic()};
    load(*ctx, uniaxial(0.006), steps);
    EXPECT_LT(ctx->get<T>("mt", "residual"), 1e-10) << steps;
    EXPECT_GT(ctx->get<T>("plastic", "equivalent_plastic_strain"), 0) << steps;
    EXPECT_NEAR(tmech::norm(tensor2{(1 - c) * ctx->get<tensor2>("mt_strain", "strain_matrix") + c * ctx->get<tensor2>("mt_strain", "strain_spheres") - uniaxial(0.006)}), 0, 1e-12);
    auto const &sig_i{ctx->get<tensor2>("stiff", "stress")};
    auto const &sig_m{ctx->get<tensor2>("plastic", "stress")};
    return tmech::norm(tensor2{sig_m - sig_i}) / tmech::norm(sig_i);
  };
  T const coarse{mismatch(6)}, fine{mismatch(600)}; // the error sits in the step that crosses the yield surface
  EXPECT_LT(fine, coarse / 20);
  EXPECT_LT(fine, 1e-5);
}

TEST(MeanFieldDamage, ANaNTangentIsNotConverged) {
  auto ctx{build(true)};
  dynamic_cast<nm::linear_elasticity<policy> &>(*ctx->find("elastic")).set_parameter<T>("K", std::numeric_limits<T>::quiet_NaN());
  auto &strain{*dynamic_cast<nm::external_strain_source<policy> *>(ctx->find("strain"))};
  strain.bind(tensor2{}, uniaxial(1e-3));
  EXPECT_THROW(ctx->update(), nm::cutback_error) << "the residual of a NaN tangent is NaN, not 0";
}

TEST(MeanFieldDamage, AZeroTangentIsNotConverged) {
  auto ctx{build(true)};
  auto &elastic{dynamic_cast<nm::linear_elasticity<policy> &>(*ctx->find("elastic"))};
  elastic.set_parameter<T>("K", 0);
  elastic.set_parameter<T>("G", 0);
  auto &strain{*dynamic_cast<nm::external_strain_source<policy> *>(ctx->find("strain"))};
  strain.bind(tensor2{}, uniaxial(1e-3));
  EXPECT_THROW(ctx->update(), nm::cutback_error) << "norm(C) = 0 makes the residual 0 / 0";
}

TEST(MeanFieldDamage, RejectsAnInconsistentWiring) {
  {
    ctx_type ctx; // a local tangent edge without an initial tangent
    constant(ctx, "c", 0.3);
    param_type p;
    p.insert<std::string>("name", "p");
    p.insert<std::string>("tangent_source", "x");
    p.insert<std::string>("fraction_source", "c");
    p.insert<std::string>("tangent_edge", "local");
    EXPECT_THROW(ctx.create<nm::phase<policy>>(p), std::invalid_argument);
  }
  {
    ctx_type ctx; // a nonlinear composite with nothing to compare: no phase names the tangent it was localised with
    param_type p;
    p.insert<std::string>("name", "mt");
    p.insert<nm::phase_sources>("phases", {{{"concentration", "s::concentration_m"}, {"stress", "m::stress"}, {"tangent", "m::tangent"},
                                            {"fraction", "m::volume_fraction"}}});
    EXPECT_THROW(ctx.create<nm::mean_field_composite_local_nonlinear<policy>>(p), std::invalid_argument);
  }
}

TEST(MeanFieldDamage, FromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "constant_scalar", "name": "Km", "value": 3.0}, {"type": "constant_scalar", "name": "Gm", "value": 1.0},
    {"type": "constant_scalar", "name": "Ki", "value": 60.0}, {"type": "constant_scalar", "name": "Gi", "value": 30.0},
    {"type": "isotropic_tangent", "name": "C_m", "K_source": "Km", "G_source": "Gm"},
    {"type": "isotropic_tangent", "name": "C_i", "K_source": "Ki", "G_source": "Gi"},
    {"type": "isotropic_moduli", "name": "moduli", "tangent_source": "C_m"},
    {"type": "eshelby_sphere", "name": "sphere", "nu_source": "moduli", "nu_property": "poisson_ratio"},
    {"type": "constant_scalar", "name": "c", "value": 0.3},
    {"type": "constant_scalar", "name": "cm", "value": 0.7},
    {"type": "phase", "name": "matrix", "tangent_source": "damage", "tangent_property": "damaged_tangent", "tangent_edge": "local",
     "initial_tangent_source": "C_m", "fraction_source": "cm"},
    {"type": "phase", "name": "spheres", "tangent_source": "C_i", "fraction_source": "c", "eshelby_source": "sphere"},
    {"type": "mean_field_strain", "name": "mt_strain", "strain_source": "strain", "phases": ["matrix", "spheres"], "kernel": {"type": "mori_tanaka"}},
    {"type": "linear_elasticity", "name": "elastic", "strain_producer_name": "mt_strain", "strain_property": "strain_matrix", "K": 3.0, "G": 1.0},
    {"type": "von_mises_state_function", "name": "state", "strain_source": "mt_strain", "strain_property": "strain_matrix"},
    {"type": "strain_threshold_yield", "name": "yield", "state_source": "state", "kappa_critical": 0.002},
    {"type": "exponential_damage_law", "name": "law", "yield_source": "yield", "kappa_0": 0.002, "beta": 300.0},
    {"type": "isotropic_damage", "name": "damage", "elastic_source": "elastic", "damage_source": "law", "state_source": "state", "yield_source": "yield"},
    {"type": "linear_stress", "name": "stiff", "tangent_source": "C_i", "strain_source": "mt_strain", "strain_property": "strain_spheres"},
    {"type": "mean_field_composite_local_nonlinear", "name": "mt", "phases": [
      {"concentration": "mt_strain::concentration_matrix", "stress": "damage::damaged_stress", "tangent": "damage::damaged_tangent",
       "fraction": "matrix::volume_fraction", "used_tangent": "matrix::tangent"},
      {"concentration": "mt_strain::concentration_spheres", "stress": "stiff::stress", "tangent": "C_i::tangent", "fraction": "spheres::volume_fraction"}]}]})")(ctx, {});
  load(ctx, uniaxial(0.01), 10);
  EXPECT_GT(ctx.get<T>("law", "damage"), 0);
  EXPECT_NEAR(tmech::norm(tensor2{0.7 * ctx.get<tensor2>("mt_strain", "strain_matrix") + 0.3 * ctx.get<tensor2>("mt_strain", "strain_spheres") - uniaxial(0.01)}), 0, 1e-12);
}

} // namespace
