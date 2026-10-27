#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/core/isotropize.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/depolarisation_sphere.h"
#include "numsim-materials/materials/eshelby_sphere.h"
#include "numsim-materials/materials/isotropic_tangent.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/phase.h"
#include "numsim-materials/materials/poisson_ratio_from_moduli.h"
#include "numsim-materials/materials/rule_of_mixture.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

constexpr T K1{100}, G1{40}, K2{10}, G2{3}, c1{0.3}, c2{0.7};

void constant(ctx_type &ctx, std::string const &name, T value) {
  param_type p;
  p.insert<std::string>("name", name);
  p.insert<T>("value", value);
  ctx.create<nm::constant_scalar<policy>>(p);
}

/// K, G producers -> isotropic tangents -> phases with produced fractions -> rule of mixture `kernel` -> linear_stress.
std::unique_ptr<ctx_type> build(std::string const &kernel, std::vector<T> fractions = {c1, c2},
                                std::vector<std::string> phases = {"stiff", "soft"}) {
  auto owner{std::make_unique<ctx_type>()};
  auto &ctx{*owner};
  param_type p;
  p.insert<std::string>("name", "strain");
  ctx.create<nm::external_strain_source<policy>>(p);
  for (auto const &[name, K, G] : {std::tuple{"stiff", K1, G1}, std::tuple{"soft", K2, G2}}) {
    constant(ctx, std::string{name} + "_K", K);
    constant(ctx, std::string{name} + "_G", G);
    p.clear();
    p.insert<std::string>("name", std::string{name} + "_C");
    p.insert<std::string>("K_source", std::string{name} + "_K");
    p.insert<std::string>("G_source", std::string{name} + "_G");
    ctx.create<nm::isotropic_tangent<policy>>(p);
  }
  for (std::size_t i{0}; i < phases.size(); ++i) {
    constant(ctx, "c_" + phases[i], fractions[i]);
    p.clear();
    p.insert<std::string>("name", phases[i]);
    p.insert<std::string>("tangent_source", phases[i] + "_C");
    p.insert<std::string>("fraction_source", "c_" + phases[i]);
    ctx.create<nm::phase<policy>>(p);
  }
  p.clear();
  p.insert<std::string>("name", "rom");
  p.insert<std::vector<std::string>>("phases", phases);
  p.insert<std::string>("kernel", kernel);
  ctx.create<nm::rule_of_mixture<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "composite");
  p.insert<std::string>("tangent_source", "rom");
  p.insert<std::string>("strain_source", "strain");
  ctx.create<nm::linear_stress<policy>>(p);
  ctx.finalize();
  ctx.update(); // the composite tangent is computed per update, not at construction
  return owner;
}

TEST(RuleOfMixture, VoigtIsTheArithmeticMean) {
  auto ctx{build("voigt")};
  auto const [K, G]{nm::isotropize<T, 3>(ctx->get<tensor4>("rom", "tangent"))};
  EXPECT_NEAR(K, c1 * K1 + c2 * K2, 1e-12);
  EXPECT_NEAR(G, c1 * G1 + c2 * G2, 1e-12);
}

TEST(RuleOfMixture, ReussIsTheHarmonicMean) {
  auto ctx{build("reuss")};
  auto const [K, G]{nm::isotropize<T, 3>(ctx->get<tensor4>("rom", "tangent"))};
  EXPECT_NEAR(K, 1 / (c1 / K1 + c2 / K2), 1e-12);
  EXPECT_NEAR(G, 1 / (c1 / G1 + c2 / G2), 1e-12);
}

TEST(RuleOfMixture, HillIsBetween) {
  auto v{build("voigt")}, r{build("reuss")}, h{build("hill")};
  tensor4 const expected{(v->get<tensor4>("rom", "tangent") + r->get<tensor4>("rom", "tangent")) / 2};
  EXPECT_NEAR(tmech::norm(tensor4{h->get<tensor4>("rom", "tangent") - expected}), 0, 1e-12);
  auto const [Kv, Gv]{nm::isotropize<T, 3>(v->get<tensor4>("rom", "tangent"))};
  auto const [Kh, Gh]{nm::isotropize<T, 3>(h->get<tensor4>("rom", "tangent"))};
  auto const [Kr, Gr]{nm::isotropize<T, 3>(r->get<tensor4>("rom", "tangent"))};
  EXPECT_GT(Kv, Kh);
  EXPECT_GT(Kh, Kr);
  EXPECT_GT(Gv, Gh);
  EXPECT_GT(Gh, Gr);
}

TEST(RuleOfMixture, SinglePhaseReturnsItsTangent) {
  auto ctx{build("reuss", {1.0}, {"soft"})};
  EXPECT_NEAR(tmech::norm(tensor4{ctx->get<tensor4>("rom", "tangent") - ctx->get<tensor4>("soft", "tangent")}), 0, 1e-12);
}

TEST(RuleOfMixture, StressOnTheTangentAndConcentrationsAverageToIdentity) {
  for (auto const *kernel : {"voigt", "reuss", "hill"}) {
    auto ctx{build(kernel)};
    tensor2 eps;
    eps(0, 0) = 1e-3;
    eps(0, 1) = eps(1, 0) = 2e-3;
    dynamic_cast<nm::external_strain_source<policy> *>(ctx->find("strain"))->bind(tensor2{}, eps);
    ctx->update();
    tensor2 const expected{tmech::dcontract(ctx->get<tensor4>("rom", "tangent"), eps)};
    EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("composite", "stress") - expected}), 0, 1e-12) << kernel;
    tensor4 const A_sum{c1 * ctx->get<tensor4>("rom", "concentration_stiff") + c2 * ctx->get<tensor4>("rom", "concentration_soft")};
    EXPECT_NEAR(tmech::norm(tensor4{A_sum - nm::sym_identity<T, 3>()}), 0, 1e-12) << kernel;
    // phase strains eps_i = A_i : eps average to eps, phase stresses to the composite stress
    tensor2 const sig_avg{c1 * tmech::dcontract(ctx->get<tensor4>("stiff", "tangent"), tensor2{tmech::dcontract(ctx->get<tensor4>("rom", "concentration_stiff"), eps)}) +
                          c2 * tmech::dcontract(ctx->get<tensor4>("soft", "tangent"), tensor2{tmech::dcontract(ctx->get<tensor4>("rom", "concentration_soft"), eps)})};
    EXPECT_NEAR(tmech::norm(tensor2{sig_avg - ctx->get<tensor2>("composite", "stress")}), 0, 1e-12) << kernel;
  }
}

TEST(RuleOfMixture, RejectsBadFractionsAndKernels) {
  EXPECT_THROW(build("voigt", {0.5, 0.6}), std::invalid_argument);  // the produced fractions do not sum to one
  EXPECT_THROW(build("voigt", {-0.1, 1.1}), std::invalid_argument); // a negative fraction, rejected by the phase
  EXPECT_THROW(build("taylor"), std::invalid_argument);
}

TEST(RuleOfMixture, FromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "constant_scalar", "name": "Ka", "value": 100.0}, {"type": "constant_scalar", "name": "Ga", "value": 40.0},
    {"type": "constant_scalar", "name": "Kb", "value": 10.0}, {"type": "constant_scalar", "name": "Gb", "value": 3.0},
    {"type": "isotropic_tangent", "name": "Ca", "K_source": "Ka", "G_source": "Ga"},
    {"type": "isotropic_tangent", "name": "Cb", "K_source": "Kb", "G_source": "Gb"},
    {"type": "constant_scalar", "name": "ca", "value": 0.3},
    {"type": "constant_scalar", "name": "cb", "value": 0.7},
    {"type": "phase", "name": "pa", "tangent_source": "Ca", "fraction_source": "ca"},
    {"type": "phase", "name": "pb", "tangent_source": "Cb", "fraction_source": "cb"},
    {"type": "rule_of_mixture", "name": "rom", "phases": ["pa", "pb"], "kernel": "voigt"},
    {"type": "linear_stress", "name": "composite", "tangent_source": "rom", "strain_source": "strain"}]})")(ctx, {});
  ctx.update();
  auto const [K, G]{nm::isotropize<T, 3>(ctx.get<tensor4>("rom", "tangent"))};
  EXPECT_NEAR(K, 0.3 * 100 + 0.7 * 10, 1e-12);
  EXPECT_NEAR(tmech::norm(ctx.get<tensor2>("composite", "stress")), 0, 1e-15); // no strain bound yet
}

// a phase republishes the Eshelby and depolarisation tensors of its shape as properties
TEST(Phase, CarriesTheShapeTensorsAsProperties) {
  ctx_type ctx;
  constant(ctx, "K", K1);
  constant(ctx, "G", G1);
  constant(ctx, "c", 0.2);
  param_type p;
  p.insert<std::string>("name", "C");
  p.insert<std::string>("K_source", "K");
  p.insert<std::string>("G_source", "G");
  ctx.create<nm::isotropic_tangent<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "nu");
  p.insert<T>("K", K1);
  p.insert<T>("G", G1);
  ctx.create<nm::poisson_ratio_from_moduli<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "S");
  p.insert<std::string>("nu_source", "nu");
  ctx.create<nm::eshelby_sphere<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "N");
  ctx.create<nm::depolarisation_sphere<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "spheres");
  p.insert<std::string>("tangent_source", "C");
  p.insert<std::string>("fraction_source", "c");
  p.insert<std::string>("eshelby_source", "S");
  p.insert<std::string>("depolarisation_source", "N");
  ctx.create<nm::phase<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "bare");
  p.insert<std::string>("tangent_source", "C");
  p.insert<std::string>("fraction_source", "c");
  ctx.create<nm::phase<policy>>(p);
  ctx.finalize();
  ctx.update();
  EXPECT_NEAR(ctx.get<T>("spheres", "volume_fraction"), 0.2, 1e-15);
  EXPECT_NEAR(tmech::norm(tensor4{ctx.get<tensor4>("spheres", "tangent") - ctx.get<tensor4>("C", "tangent")}), 0, 1e-15);
  EXPECT_NEAR(tmech::norm(tensor4{ctx.get<tensor4>("spheres", "eshelby") - nm::eshelby::sphere(nm::poisson_ratio(K1, G1))}), 0, 1e-14);
  EXPECT_NEAR(tmech::norm(tensor2{ctx.get<tensor2>("spheres", "depolarisation") - tmech::eye<T, 3, 2>() / 3}), 0, 1e-15);
  EXPECT_ANY_THROW((void)ctx.get<tensor4>("bare", "eshelby")); // a bare phase publishes no shape tensor
}

} // namespace
