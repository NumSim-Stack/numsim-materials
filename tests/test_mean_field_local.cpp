#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/eshelby_sphere.h"
#include "numsim-materials/materials/isotropic_moduli.h"
#include "numsim-materials/materials/isotropic_tangent.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/mean_field_strain.h"
#include "numsim-materials/materials/mean_field_stress.h"
#include "numsim-materials/materials/mean_field_composite_global.h"
#include "numsim-materials/materials/phase.h"
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

constexpr T Km{3.0}, Gm{1.0}, Ki{60.0}, Gi{30.0}, c{0.3};

void constant(ctx_type &ctx, std::string const &name, T value) {
  param_type p;
  p.insert<std::string>("name", name);
  p.insert<T>("value", value);
  ctx.create<nm::constant_scalar<policy>>(p);
}

/// Stiff elastic spheres in an elastic matrix, both evaluated at their local strains: every edge is global.
std::unique_ptr<ctx_type> build(std::string const &kernel = "mori_tanaka") {
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
  p.clear();
  p.insert<std::string>("name", "matrix");
  p.insert<std::string>("tangent_source", "C_m");
  p.insert<std::string>("fraction_source", "cm");
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
  p.insert<std::string>("matrix", "matrix");
  p.insert<std::vector<std::string>>("phases", {"spheres"});
  p.insert<std::string>("kernel", kernel);
  ctx.create<nm::mean_field_strain<policy>>(p);
  // the phase materials at their local strains
  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "mt_strain");
  p.insert<std::string>("strain_property", "strain_matrix");
  p.insert<T>("K", Km);
  p.insert<T>("G", Gm);
  ctx.create<nm::linear_elasticity<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "stiff");
  p.insert<std::string>("tangent_source", "C_i");
  p.insert<std::string>("strain_source", "mt_strain");
  p.insert<std::string>("strain_property", "strain_spheres");
  ctx.create<nm::linear_stress<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "mt");
  p.insert<std::string>("concentration_source", "mt_strain");
  p.insert<std::string>("matrix", "matrix");
  p.insert<std::vector<std::string>>("phases", {"spheres"});
  p.insert<std::string>("matrix_stress_source", "elastic");
  p.insert<std::string>("matrix_tangent_source", "C_m");
  p.insert<std::vector<std::string>>("stress_sources", {"stiff"});
  p.insert<std::vector<std::string>>("tangent_sources", {"C_i"});
  ctx.create<nm::mean_field_stress<policy>>(p);
  ctx.finalize();
  return owner;
}

tensor2 uniaxial(T e) {
  tensor2 eps;
  eps(0, 0) = e;
  return eps;
}

/// One step: passes until the phase tangents the strains were formed with agree with the materials'.
int converge(ctx_type &ctx, T tol = 1e-10) {
  int passes{1};
  ctx.update();
  while (ctx.get<T>("mt", "residual") > tol && passes < 200) {
    ctx.update();
    ++passes;
  }
  return passes;
}

/// Loads the composite in equal steps to the strain `eps`, committing every step.
void load(ctx_type &ctx, tensor2 const &eps, int steps) {
  auto &strain{*dynamic_cast<nm::external_strain_source<policy> *>(ctx.find("strain"))};
  for (int n{1}; n <= steps; ++n) {
    strain.bind(tensor2{eps * T(n - 1) / T(steps)}, tensor2{eps * T(n) / T(steps)});
    converge(ctx);
    ctx.commit();
  }
}

TEST(MeanFieldLocal, LinearPhasesReproduceTheTangentHomogenisationInOnePass) {
  auto ctx{build()};
  param_type p; // the same phases through the global composite (approach A)
  p.insert<std::string>("name", "mf");
  p.insert<std::vector<std::string>>("phases", {"matrix", "spheres"});
  p.insert<nm::object_spec<param_type>>("kernel", {"mori_tanaka", {}});
  ctx_type ref;
  // approach A on the same producers: built inline to keep one graph per context
  for (auto const &[name, K, G] : {std::tuple{"m", Km, Gm}, std::tuple{"i", Ki, Gi}}) {
    constant(ref, std::string{"K_"} + name, K);
    constant(ref, std::string{"G_"} + name, G);
    param_type q;
    q.insert<std::string>("name", std::string{"C_"} + name);
    q.insert<std::string>("K_source", std::string{"K_"} + name);
    q.insert<std::string>("G_source", std::string{"G_"} + name);
    ref.create<nm::isotropic_tangent<policy>>(q);
  }
  param_type q;
  q.insert<std::string>("name", "moduli");
  q.insert<std::string>("tangent_source", "C_m");
  ref.create<nm::isotropic_moduli<policy>>(q);
  q.clear();
  q.insert<std::string>("name", "sphere");
  q.insert<std::string>("nu_source", "moduli");
  q.insert<std::string>("nu_property", "poisson_ratio");
  ref.create<nm::eshelby_sphere<policy>>(q);
  constant(ref, "c", c);
  constant(ref, "cm", 1 - c);
  for (auto const &[name, tangent, fraction] : {std::tuple{"matrix", "C_m", "cm"}, std::tuple{"spheres", "C_i", "c"}}) {
    q.clear();
    q.insert<std::string>("name", name);
    q.insert<std::string>("tangent_source", tangent);
    q.insert<std::string>("fraction_source", fraction);
    q.insert<std::string>("eshelby_source", "sphere");
    ref.create<nm::phase<policy>>(q);
  }
  ref.create<nm::mean_field_composite_global<policy>>(p);
  ref.finalize();
  ref.update();
  tensor2 const eps{uniaxial(1e-3)};
  EXPECT_EQ(converge(*ctx), 1); // linear: the first pass is the fixed point
  load(*ctx, eps, 3);
  EXPECT_EQ(ctx->get<T>("mt", "residual"), 0);
  auto const &C_A{ref.get<tensor4>("mf", "tangent")};
  EXPECT_NEAR(tmech::norm(tensor4{ctx->get<tensor4>("mt", "tangent") - C_A}) / tmech::norm(C_A), 0, 1e-12);
  EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("mt", "stress") - tmech::dcontract(C_A, eps)}) / tmech::norm(tmech::dcontract(C_A, eps)), 0, 1e-12);
  EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("mt_strain", "strain_matrix") - tmech::dcontract(ref.get<tensor4>("mf", "concentration_matrix"), eps)}), 0, 1e-15);
}

// the rule-of-mixture kernels localise too: the same concentrations as the tangent producer, the same tangent
TEST(MeanFieldLocal, RuleOfMixtureKernelsEqualTheirTangentProducer) {
  for (auto const *kernel : {"voigt", "reuss", "hill"}) {
    auto ctx{build(kernel)};
    tensor2 const eps{uniaxial(1e-3)};
    EXPECT_EQ(converge(*ctx), 1) << kernel;
    load(*ctx, eps, 2);
    ctx_type ref;
    for (auto const &[name, K, G] : {std::tuple{"m", Km, Gm}, std::tuple{"i", Ki, Gi}}) {
      constant(ref, std::string{"K_"} + name, K);
      constant(ref, std::string{"G_"} + name, G);
      param_type q;
      q.insert<std::string>("name", std::string{"C_"} + name);
      q.insert<std::string>("K_source", std::string{"K_"} + name);
      q.insert<std::string>("G_source", std::string{"G_"} + name);
      ref.create<nm::isotropic_tangent<policy>>(q);
    }
    constant(ref, "c", c);
    constant(ref, "cm", 1 - c);
    for (auto const &[name, tangent, fraction] : {std::tuple{"matrix", "C_m", "cm"}, std::tuple{"spheres", "C_i", "c"}}) {
      param_type q;
      q.insert<std::string>("name", name);
      q.insert<std::string>("tangent_source", tangent);
      q.insert<std::string>("fraction_source", fraction);
      ref.create<nm::phase<policy>>(q);
    }
    param_type q;
    q.insert<std::string>("name", "rom");
    q.insert<std::vector<std::string>>("phases", {"matrix", "spheres"});
    q.insert<nm::object_spec<param_type>>("kernel", {kernel, {}});
    ref.create<nm::mean_field_composite_global<policy>>(q);
    ref.finalize();
    ref.update();
    auto const &C{ref.get<tensor4>("rom", "tangent")};
    EXPECT_NEAR(tmech::norm(tensor4{ctx->get<tensor4>("mt", "tangent") - C}) / tmech::norm(C), 0, 1e-12) << kernel;
    EXPECT_NEAR(tmech::norm(tensor4{ctx->get<tensor4>("mt_strain", "concentration_spheres") - ref.get<tensor4>("rom", "concentration_spheres")}), 0, 1e-12) << kernel;
    EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("mt", "stress") - tmech::dcontract(C, eps)}) / tmech::norm(tmech::dcontract(C, eps)), 0, 1e-12) << kernel;
    if (std::string{kernel} == "voigt") {
      EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("mt_strain", "strain_spheres") - eps}), 0, 1e-15); // equal strain
    }
    if (std::string{kernel} == "reuss") {
      EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("stiff", "stress") - ctx->get<tensor2>("elastic", "stress")}), 0, 1e-12); // equal stress
    }
  }
}

TEST(MeanFieldLocal, RejectsNonParallelSourceLists) {
  {
    ctx_type ctx; // stress sources not parallel to the phases
    param_type p;
    p.insert<std::string>("name", "mt");
    p.insert<std::string>("concentration_source", "s");
    p.insert<std::string>("matrix", "m");
    p.insert<std::vector<std::string>>("phases", {"a", "b"});
    p.insert<std::string>("matrix_stress_source", "m");
    p.insert<std::vector<std::string>>("stress_sources", {"a"});
    EXPECT_THROW(ctx.create<nm::mean_field_stress<policy>>(p), std::invalid_argument);
  }
}

TEST(MeanFieldLocal, FromJson) {
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
    {"type": "phase", "name": "matrix", "tangent_source": "C_m", "fraction_source": "cm"},
    {"type": "phase", "name": "spheres", "tangent_source": "C_i", "fraction_source": "c", "eshelby_source": "sphere"},
    {"type": "mean_field_strain", "name": "mt_strain", "strain_source": "strain", "matrix": "matrix", "phases": ["spheres"]},
    {"type": "linear_stress", "name": "m", "tangent_source": "C_m", "strain_source": "mt_strain", "strain_property": "strain_matrix"},
    {"type": "linear_stress", "name": "s", "tangent_source": "C_i", "strain_source": "mt_strain", "strain_property": "strain_spheres"},
    {"type": "mean_field_stress", "name": "mt", "concentration_source": "mt_strain", "matrix": "matrix", "phases": ["spheres"],
     "matrix_stress_source": "m", "matrix_tangent_source": "C_m", "stress_sources": ["s"], "tangent_sources": ["C_i"]}]})")(ctx, {});
  load(ctx, uniaxial(0.01), 2);
  EXPECT_EQ(ctx.get<T>("mt", "residual"), 0);
  EXPECT_NEAR(tmech::norm(tensor2{0.7 * ctx.get<tensor2>("mt_strain", "strain_matrix") + 0.3 * ctx.get<tensor2>("mt_strain", "strain_spheres") - uniaxial(0.01)}), 0, 1e-12);
}

} // namespace
