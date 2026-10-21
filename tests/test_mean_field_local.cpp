#include <gtest/gtest.h>
#include <map>
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
#include "numsim-materials/materials/mean_field_composite_local_linear.h"
#include "numsim-materials/materials/mean_field_composite_local_nonlinear.h"
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

/// The kernel object of `type`; self-consistent carries the shapes of both phases.
nm::object_spec<param_type> kernel_spec(std::string const &type) {
  param_type k;
  if (type == "self_consistent") {
    k.insert<std::map<std::string, nm::object_spec<param_type>>>("shapes", {{"matrix", {"sphere", {}}}, {"spheres", {"sphere", {}}}});
    k.insert<T>("tolerance", 1e-13);
  }
  return {type, k};
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
  p.insert<std::vector<std::string>>("phases", {"matrix", "spheres"});
  p.insert<nm::object_spec<param_type>>("kernel", kernel_spec(kernel));
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
  p.insert<nm::phase_sources>("phases", {{{"concentration", "mt_strain::concentration_matrix"}, {"stress", "elastic::stress"},
                                          {"tangent", "C_m::tangent"}, {"fraction", "matrix::volume_fraction"}},
                                         {{"concentration", "mt_strain::concentration_spheres"}, {"stress", "stiff::stress"},
                                          {"tangent", "C_i::tangent"}, {"fraction", "spheres::volume_fraction"}}});
  ctx.create<nm::mean_field_composite_local_linear<policy>>(p);
  ctx.finalize();
  return owner;
}

struct policy_2d : policy {
  static constexpr std::size_t Dim = 2;
};

tensor2 uniaxial(T e) {
  tensor2 eps;
  eps(0, 0) = e;
  return eps;
}

/// Loads the composite in equal steps to the strain `eps`, committing every step.
void load(ctx_type &ctx, tensor2 const &eps, int steps) {
  auto &strain{*dynamic_cast<nm::external_strain_source<policy> *>(ctx.find("strain"))};
  for (int n{1}; n <= steps; ++n) {
    strain.bind(tensor2{eps * T(n - 1) / T(steps)}, tensor2{eps * T(n) / T(steps)});
    ctx.update(); // linear localisation: one pass
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
  ctx->update();
  load(*ctx, eps, 3);
  auto const &C_A{ref.get<tensor4>("mf", "tangent")};
  EXPECT_NEAR(tmech::norm(tensor4{ctx->get<tensor4>("mt", "tangent") - C_A}) / tmech::norm(C_A), 0, 1e-12);
  EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("mt", "stress") - tmech::dcontract(C_A, eps)}) / tmech::norm(tmech::dcontract(C_A, eps)), 0, 1e-12);
  EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("mt_strain", "strain_matrix") - tmech::dcontract(ref.get<tensor4>("mf", "concentration_matrix"), eps)}), 0, 1e-15);
}

// every kernel localises like the global composite holding the same kernel object: one pass, the same
// concentrations, the same tangent
TEST(MeanFieldLocal, EveryKernelLocalisesLikeTheGlobalComposite) {
  for (auto const *kernel : {"voigt", "reuss", "hill", "dilute", "mori_tanaka", "self_consistent"}) {
    auto ctx{build(kernel)};
    tensor2 const eps{uniaxial(1e-3)};
    ctx->update();
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
    param_type q;
    q.insert<std::string>("name", "moduli");
    q.insert<std::string>("tangent_source", "C_m");
    ref.create<nm::isotropic_moduli<policy>>(q);
    q.clear();
    q.insert<std::string>("name", "sphere");
    q.insert<std::string>("nu_source", "moduli");
    q.insert<std::string>("nu_property", "poisson_ratio");
    ref.create<nm::eshelby_sphere<policy>>(q);
    for (auto const &[name, tangent, fraction] : {std::tuple{"matrix", "C_m", "cm"}, std::tuple{"spheres", "C_i", "c"}}) {
      q.clear();
      q.insert<std::string>("name", name);
      q.insert<std::string>("tangent_source", tangent);
      q.insert<std::string>("fraction_source", fraction);
      q.insert<std::string>("eshelby_source", "sphere");
      ref.create<nm::phase<policy>>(q);
    }
    q.clear();
    q.insert<std::string>("name", "rom");
    q.insert<std::vector<std::string>>("phases", {"matrix", "spheres"});
    q.insert<nm::object_spec<param_type>>("kernel", kernel_spec(kernel));
    ref.create<nm::mean_field_composite_global<policy>>(q);
    ref.finalize();
    ref.update();
    auto const &C{ref.get<tensor4>("rom", "tangent")};
    EXPECT_NEAR(tmech::norm(tensor4{ctx->get<tensor4>("mt", "tangent") - C}) / tmech::norm(C), 0, 1e-11) << kernel;
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

TEST(MeanFieldLocal, RejectsAnIncompletePhase) {
  auto const rejects = [](nm::phase_sources phases, std::string const &what) {
    ctx_type ctx;
    param_type p;
    p.insert<std::string>("name", "mt");
    p.insert<nm::phase_sources>("phases", std::move(phases));
    try {
      ctx.create<nm::mean_field_composite_local_linear<policy>>(p);
      ADD_FAILURE() << "accepted: " << what;
    } catch (std::invalid_argument const &e) {
      EXPECT_NE(std::string{e.what()}.find(what), std::string::npos) << e.what();
    }
  };
  rejects({}, "phase");
  rejects({{{"concentration", "s::concentration_m"}, {"tangent", "m::tangent"}, {"fraction", "m::volume_fraction"}}}, "stress");
  rejects({{{"concentration", "s::concentration_m"}, {"stress", "m"}, {"tangent", "m::tangent"}, {"fraction", "m::volume_fraction"}}},
          "material::property");
  rejects({{{"concentration", "s::concentration_m"}, {"stress", "m::stress"}, {"tangent", "m::tangent"}, {"fraction", "m::volume_fraction"},
            {"tangnet", "m::tangent"}}},
          "tangnet");
}

// the phase strains average to the composite strain: a fraction wired to the wrong phase shows at the first update
TEST(MeanFieldLocal, AWrongFractionBreaksThePartition) {
  ctx_type wrong;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "constant_scalar", "name": "Km", "value": 3.0}, {"type": "constant_scalar", "name": "Gm", "value": 1.0},
    {"type": "constant_scalar", "name": "Ki", "value": 60.0}, {"type": "constant_scalar", "name": "Gi", "value": 30.0},
    {"type": "isotropic_tangent", "name": "C_m", "K_source": "Km", "G_source": "Gm"},
    {"type": "isotropic_tangent", "name": "C_i", "K_source": "Ki", "G_source": "Gi"},
    {"type": "constant_scalar", "name": "c", "value": 0.3},
    {"type": "constant_scalar", "name": "cm", "value": 0.7},
    {"type": "phase", "name": "matrix", "tangent_source": "C_m", "fraction_source": "cm"},
    {"type": "phase", "name": "spheres", "tangent_source": "C_i", "fraction_source": "c"},
    {"type": "mean_field_strain", "name": "mt_strain", "strain_source": "strain", "phases": ["matrix", "spheres"], "kernel": {"type": "reuss"}},
    {"type": "linear_stress", "name": "m", "tangent_source": "C_m", "strain_source": "mt_strain", "strain_property": "strain_matrix"},
    {"type": "linear_stress", "name": "s", "tangent_source": "C_i", "strain_source": "mt_strain", "strain_property": "strain_spheres"},
    {"type": "mean_field_composite_local_linear", "name": "mt", "phases": [
      {"concentration": "mt_strain::concentration_matrix", "stress": "m::stress", "tangent": "C_m::tangent", "fraction": "spheres::volume_fraction"},
      {"concentration": "mt_strain::concentration_spheres", "stress": "s::stress", "tangent": "C_i::tangent", "fraction": "matrix::volume_fraction"}]}]})")(wrong, {});
  EXPECT_THROW(wrong.update(), std::invalid_argument);
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
    {"type": "mean_field_strain", "name": "mt_strain", "strain_source": "strain", "phases": ["matrix", "spheres"], "kernel": {"type": "mori_tanaka"}},
    {"type": "linear_stress", "name": "m", "tangent_source": "C_m", "strain_source": "mt_strain", "strain_property": "strain_matrix"},
    {"type": "linear_stress", "name": "s", "tangent_source": "C_i", "strain_source": "mt_strain", "strain_property": "strain_spheres"},
    {"type": "mean_field_composite_local_linear", "name": "mt", "phases": [
      {"concentration": "mt_strain::concentration_matrix", "stress": "m::stress", "tangent": "C_m::tangent", "fraction": "matrix::volume_fraction"},
      {"concentration": "mt_strain::concentration_spheres", "stress": "s::stress", "tangent": "C_i::tangent", "fraction": "spheres::volume_fraction"}]}]})")(ctx, {});
  load(ctx, uniaxial(0.01), 2);
  EXPECT_NEAR(tmech::norm(tensor2{0.7 * ctx.get<tensor2>("mt_strain", "strain_matrix") + 0.3 * ctx.get<tensor2>("mt_strain", "strain_spheres") - uniaxial(0.01)}), 0, 1e-12);
}

} // namespace

TEST(Phase, RejectsInconsistentParameters) {
  auto const make{[](std::initializer_list<std::pair<char const *, char const *>> extra) {
    ctx_type ctx;
    constant(ctx, "c", 0.3);
    param_type p;
    p.insert<std::string>("name", "p");
    p.insert<std::string>("tangent_source", "x");
    p.insert<std::string>("fraction_source", "c");
    for (auto const &[key, value] : extra)
      p.insert<std::string>(key, value);
    ctx.create<nm::phase<policy>>(p);
  }};
  EXPECT_NO_THROW(make({}));
  EXPECT_THROW(make({{"initial_tangent_source", "x"}}), std::invalid_argument); // only with a local edge
  EXPECT_THROW(make({{"tangent_edge", "sideways"}}), std::invalid_argument);
  EXPECT_THROW(make({{"tangent_source", ""}}), std::invalid_argument);
  EXPECT_THROW(make({{"fraction_source", ""}}), std::invalid_argument);
  {
    nm::material_context<policy_2d> ctx2; // the Eshelby tensor of a 2D phase
    param_type p;
    p.insert<std::string>("name", "p");
    p.insert<std::string>("tangent_source", "x");
    p.insert<std::string>("fraction_source", "c");
    p.insert<std::string>("eshelby_source", "s");
    EXPECT_THROW(ctx2.create<nm::phase<policy_2d>>(p), std::invalid_argument);
  }
}

TEST(Phase, RejectsANegativeFraction) {
  ctx_type ctx;
  constant(ctx, "c", -0.1);
  constant(ctx, "x", 1.0);
  param_type p;
  p.insert<std::string>("name", "p");
  p.insert<std::string>("tangent_source", "C_x");
  p.insert<std::string>("fraction_source", "c");
  ctx.create<nm::phase<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "C_x");
  p.insert<std::string>("K_source", "x");
  p.insert<std::string>("G_source", "x");
  ctx.create<nm::isotropic_tangent<policy>>(p);
  ctx.finalize();
  EXPECT_THROW(ctx.update(), std::invalid_argument);
}

TEST(Phase, RelaysItsSourcesAndOnlyTheWiredTensors) {
  auto ctx{build()};
  ctx->update();
  EXPECT_EQ(tmech::norm(tensor4{ctx->get<tensor4>("spheres", "tangent") - ctx->get<tensor4>("C_i", "tangent")}), 0);
  EXPECT_EQ(ctx->get<T>("spheres", "volume_fraction"), c);
  EXPECT_EQ(tmech::norm(tensor4{ctx->get<tensor4>("spheres", "eshelby") - ctx->get<tensor4>("sphere", "tensor")}), 0);
  EXPECT_EQ(ctx->find_property("matrix", "eshelby"), nullptr);
  EXPECT_EQ(ctx->find_property("spheres", "depolarisation"), nullptr);
}
