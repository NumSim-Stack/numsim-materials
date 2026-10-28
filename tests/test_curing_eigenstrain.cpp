#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <tmech/tmech.h>
#include "numsim-materials/core/isotropize.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/eshelby_sphere.h"
#include "numsim-materials/materials/isotropic_moduli.h"
#include "numsim-materials/materials/isotropic_tangent.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/mean_field_composite_global.h"
#include "numsim-materials/materials/phase.h"
#include "numsim-materials/materials/scalar_complement_weight.h"
#include "numsim-materials/materials/scalar_stepper.h"
#include "numsim-materials/materials/thermal_eigenstrain.h"
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

constexpr T Km{3.0}, Gm{1.0}, Ki{60.0}, Gi{30.0}, beta{-0.02};
T mori_tanaka_K(T c) { return Km + c * (Ki - Km) / (1 + (1 - c) * (Ki - Km) / (Km + 4 * Gm / 3)); }

/// degree of cure xi (scalar_stepper) drives the filler fraction c = 0.4 xi and the shrinkage -0.02 xi I.
std::unique_ptr<ctx_type> build() {
  auto owner{std::make_unique<ctx_type>()};
  auto &ctx{*owner};
  param_type p;
  p.insert<std::string>("name", "strain");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "cure");
  p.insert<T>("increment", T{0});
  ctx.create<nm::scalar_stepper<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "fraction");
  p.insert<T>("increment", T{0});
  ctx.create<nm::scalar_stepper<policy>>(p);
  for (auto const &[name, K, G] : {std::tuple{"matrix", Km, Gm}, std::tuple{"filler", Ki, Gi}}) {
    for (auto const &[modulus, value] : {std::pair{"K", K}, std::pair{"G", G}}) {
      p.clear();
      p.insert<std::string>("name", std::string{name} + "_" + modulus);
      p.insert<T>("value", value);
      ctx.create<nm::constant_scalar<policy>>(p);
    }
    p.clear();
    p.insert<std::string>("name", name);
    p.insert<std::string>("K_source", std::string{name} + "_K");
    p.insert<std::string>("G_source", std::string{name} + "_G");
    ctx.create<nm::isotropic_tangent<policy>>(p);
  }
  p.clear();
  p.insert<std::string>("name", "moduli");
  p.insert<std::string>("tangent_source", "matrix");
  ctx.create<nm::isotropic_moduli<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "sphere");
  p.insert<std::string>("nu_source", "moduli");
  p.insert<std::string>("nu_property", "poisson_ratio");
  ctx.create<nm::eshelby_sphere<policy>>(p);
  p.clear(); // the produced filler fraction and its complement for the matrix phase
  p.insert<std::string>("name", "spheres");
  p.insert<std::string>("tangent_source", "filler");
  p.insert<std::string>("eshelby_source", "sphere");
  p.insert<std::string>("fraction_source", "fraction");
  p.insert<std::string>("fraction_property", "state");
  ctx.create<nm::phase<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "cm");
  p.insert<std::string>("source", "fraction::state");
  ctx.create<nm::scalar_complement_weight<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "matrix_phase");
  p.insert<std::string>("tangent_source", "matrix");
  p.insert<std::string>("fraction_source", "cm");
  ctx.create<nm::phase<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "mf");
  p.insert<std::vector<std::string>>("phases", {"matrix_phase", "spheres"});
  p.insert<nm::object_spec<param_type>>("kernel", {"mori_tanaka", {}});
  ctx.create<nm::mean_field_composite_global<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "beta");
  p.insert<T>("value", beta);
  ctx.create<nm::constant_scalar<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "shrinkage");
  p.insert<std::string>("temperature_source", "cure");
  p.insert<std::string>("alpha_source", "beta::value");
  ctx.create<nm::thermal_eigenstrain<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "strain");
  p.insert<std::string>("eigenstrain_source", "shrinkage");
  p.insert<T>("K", Km);
  p.insert<T>("G", Gm);
  ctx.create<nm::linear_elasticity<policy>>(p);
  ctx.finalize();
  return owner;
}

TEST(CuringEigenstrain, PhaseFractionFollowsItsSource) {
  auto ctx{build()};
  for (T const c : {0.05, 0.3}) {
    ctx->get_mutable<T>("fraction", "state") = c;
    ctx->update();
    EXPECT_DOUBLE_EQ(ctx->get<T>("spheres", "volume_fraction"), c);
    auto const [K, G]{nm::isotropize<T, 3>(ctx->get<tensor4>("mf", "tangent"))};
    EXPECT_NEAR(K, mori_tanaka_K(c), 1e-10 * mori_tanaka_K(c)) << c;
  }
}

TEST(CuringEigenstrain, ChemicalShrinkageIsAnEigenstrainOfTheDegreeOfCure) {
  auto ctx{build()};
  ctx->get_mutable<T>("cure", "state") = 0.5;
  ctx->update(); // constrained (zero strain): sigma = -3 K beta xi I
  auto const &sig{ctx->get<tensor2>("elastic", "stress")};
  EXPECT_NEAR(sig(0, 0), -3 * Km * beta * 0.5, 1e-12);
  EXPECT_NEAR(sig(1, 1), sig(0, 0), 1e-12);
  EXPECT_NEAR(ctx->get<tensor2>("shrinkage", "eigenstrain")(2, 2), beta * 0.5, 1e-15);
}

// the stress accumulated from the partials: d sigma / d eps is the tangent, d sigma / d xi = -C : beta I
TEST(CuringEigenstrain, AccumulatedDerivativesOfTheStress) {
  auto ctx{build()};
  auto const &by_strain{ctx->derivative<tensor4>("elastic", "stress", "strain", "strain")};
  auto const &by_cure{ctx->derivative<tensor2>("elastic", "stress", "cure", "state")};
  auto &strain{*dynamic_cast<nm::external_strain_source<policy> *>(ctx->find("strain"))};
  tensor2 eps;
  eps(0, 0) = 1e-3, eps(1, 2) = eps(2, 1) = -4e-4;
  T previous{0};
  for (auto const &[load, cure] : {std::pair{1.0, 0.2}, std::pair{2.0, 0.5}, std::pair{0.5, 0.7}, std::pair{-1.0, 0.9}}) {
    strain.bind(tensor2{previous * eps}, tensor2{load * eps});
    ctx->get_mutable<T>("cure", "state") = cure;
    ctx->update();
    auto const &C{ctx->get<tensor4>("elastic", "tangent")};
    EXPECT_LE(tmech::norm(tensor4{by_strain - C}), 1e-12 * tmech::norm(C)) << load;
    tensor2 const expected{-beta * tmech::dcontract(C, tmech::eye<T, 3, 2>())};
    EXPECT_LE(tmech::norm(tensor2{by_cure - expected}), 1e-12 * tmech::norm(expected)) << cure;
    EXPECT_NEAR(by_cure(0, 0), -3 * Km * beta, 1e-12);
    ctx->commit();
    previous = load;
  }
}

TEST(CuringEigenstrain, FromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "scalar_stepper", "name": "cure", "increment": 0.0},
    {"type": "constant_scalar", "name": "Km", "value": 3.0}, {"type": "constant_scalar", "name": "Gm", "value": 1.0},
    {"type": "constant_scalar", "name": "Ki", "value": 60.0}, {"type": "constant_scalar", "name": "Gi", "value": 30.0},
    {"type": "isotropic_tangent", "name": "m", "K_source": "Km", "G_source": "Gm"},
    {"type": "isotropic_tangent", "name": "i", "K_source": "Ki", "G_source": "Gi"},
    {"type": "isotropic_moduli", "name": "moduli", "tangent_source": "m"},
    {"type": "eshelby_sphere", "name": "sphere", "nu_source": "moduli", "nu_property": "poisson_ratio"},
    {"type": "phase", "name": "spheres", "tangent_source": "i", "eshelby_source": "sphere", "fraction_source": "cure", "fraction_property": "state"},
    {"type": "scalar_complement_weight", "name": "cm", "source": "cure::state"},
    {"type": "phase", "name": "matrix", "tangent_source": "m", "fraction_source": "cm"},
    {"type": "mean_field_composite_global", "name": "mf", "phases": ["matrix", "spheres"], "kernel": {"type": "mori_tanaka"}},
    {"type": "linear_stress", "name": "composite", "tangent_source": "mf", "strain_source": "strain"}]})")(ctx, {});
  ctx.get_mutable<T>("cure", "state") = 0.2;
  ctx.update();
  EXPECT_DOUBLE_EQ(ctx.get<T>("spheres", "volume_fraction"), 0.2);
}

} // namespace
