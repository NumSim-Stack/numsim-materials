#include <gtest/gtest.h>
#include <memory>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/isotropize.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/eshelby_spheroid.h"
#include "numsim-materials/materials/isotropic_moduli.h"
#include "numsim-materials/materials/isotropic_tangent.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/mean_field_composite_global.h"
#include "numsim-materials/materials/phase.h"
#include "numsim-materials/materials/short_fibre_composite.h"
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

constexpr T Km{3.0}, Gm{1.0}, Kf{60.0}, Gf{30.0}, c{0.2}, ar{10.0};
std::vector<T> const aligned{1, 0, 0, 0, 0, 0}, random3d{1.0 / 3, 1.0 / 3, 1.0 / 3, 0, 0, 0}, mostly{0.8, 0.1, 0.1, 0, 0, 0};

void constant(ctx_type &ctx, std::string const &name, T value) {
  param_type p;
  p.insert<std::string>("name", name);
  p.insert<T>("value", value);
  ctx.create<nm::constant_scalar<policy>>(p);
}

std::unique_ptr<ctx_type> build(std::vector<T> const &orientation, std::string const &averaging = "concentration",
                                std::size_t precision = 15) {
  auto owner{std::make_unique<ctx_type>()};
  auto &ctx{*owner};
  param_type p;
  p.insert<std::string>("name", "strain");
  ctx.create<nm::external_strain_source<policy>>(p);
  for (auto const &[name, K, G] : {std::tuple{"matrix", Km, Gm}, std::tuple{"fibre", Kf, Gf}}) {
    constant(ctx, std::string{name} + "_K", K);
    constant(ctx, std::string{name} + "_G", G);
    p.clear();
    p.insert<std::string>("name", std::string{name} + "_C");
    p.insert<std::string>("K_source", std::string{name} + "_K");
    p.insert<std::string>("G_source", std::string{name} + "_G");
    ctx.create<nm::isotropic_tangent<policy>>(p);
  }
  // the fibre shape along x1, shared by the unidirectional reference (a global Mori-Tanaka composite) and the composite
  p.clear();
  p.insert<std::string>("name", "moduli");
  p.insert<std::string>("tangent_source", "matrix_C");
  ctx.create<nm::isotropic_moduli<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "shape");
  p.insert<std::string>("nu_source", "moduli");
  p.insert<std::string>("nu_property", "poisson_ratio");
  p.insert<T>("aspect_ratio", ar);
  p.insert<std::vector<T>>("direction", {1, 0, 0});
  ctx.create<nm::eshelby_spheroid<policy>>(p);
  constant(ctx, "c", c);
  constant(ctx, "cm", 1 - c);
  p.clear();
  p.insert<std::string>("name", "spheroid");
  p.insert<std::string>("tangent_source", "fibre_C");
  p.insert<std::string>("eshelby_source", "shape");
  p.insert<std::string>("fraction_source", "c");
  ctx.create<nm::phase<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "m");
  p.insert<std::string>("tangent_source", "matrix_C");
  p.insert<std::string>("fraction_source", "cm");
  ctx.create<nm::phase<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "ud");
  p.insert<std::vector<std::string>>("phases", {"m", "spheroid"});
  p.insert<nm::object_spec<param_type>>("kernel", {"mori_tanaka", {}});
  ctx.create<nm::mean_field_composite_global<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "sf");
  p.insert<std::string>("matrix", "m");
  p.insert<std::string>("fibre", "spheroid");
  p.insert<std::vector<T>>("orientation", orientation);
  p.insert<std::string>("averaging", averaging);
  p.insert<int>("precision", static_cast<int>(precision));
  ctx.create<nm::short_fibre_composite<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "composite");
  p.insert<std::string>("tangent_source", "sf");
  p.insert<std::string>("strain_source", "strain");
  ctx.create<nm::linear_stress<policy>>(p);
  ctx.finalize();
  ctx.update();
  return owner;
}

T diff(tensor4 const &a, tensor4 const &b) { return tmech::norm(tensor4{a - b}) / tmech::norm(a); }

TEST(ShortFibre, AlignedEqualsTheUnidirectionalMoriTanaka) {
  for (auto const *averaging : {"concentration", "stiffness"}) {
    auto ctx{build(aligned, averaging)};
    EXPECT_NEAR(diff(ctx->get<tensor4>("sf", "tangent"), ctx->get<tensor4>("ud", "tangent")), 0, 1e-12) << averaging;
    EXPECT_NEAR(diff(ctx->get<tensor4>("sf", "concentration_fibre"), ctx->get<tensor4>("ud", "concentration_spheroid")), 0, 1e-12);
  }
}

TEST(ShortFibre, RandomOrientationIsIsotropic) {
  for (auto const *averaging : {"concentration", "stiffness"}) {
    auto ctx{build(random3d, averaging, 23)};
    auto const &C{ctx->get<tensor4>("sf", "tangent")};
    EXPECT_NEAR(diff(C, nm::isotropic_part<T, 3>(C)), 0, 1e-6) << averaging;
    auto coarse{build(random3d, averaging, 7)};
    EXPECT_LE(diff(C, nm::isotropic_part<T, 3>(C)), diff(coarse->get<tensor4>("sf", "tangent"), nm::isotropic_part<T, 3>(coarse->get<tensor4>("sf", "tangent"))) + 1e-14);
  }
}

TEST(ShortFibre, ConcentrationsAverageToIdentityAndStressIsOnTheTangent) {
  for (auto const *averaging : {"concentration", "stiffness"}) {
    auto ctx{build(mostly, averaging)};
    tensor4 const A_sum{(1 - c) * ctx->get<tensor4>("sf", "concentration_matrix") + c * ctx->get<tensor4>("sf", "concentration_fibre")};
    EXPECT_NEAR(tmech::norm(tensor4{A_sum - nm::sym_identity<T, 3>()}), 0, 1e-10) << averaging;
    tensor2 eps;
    eps(0, 0) = 1e-3;
    eps(0, 2) = eps(2, 0) = 4e-4;
    dynamic_cast<nm::external_strain_source<policy> *>(ctx->find("strain"))->bind(tensor2{}, eps);
    ctx->update();
    tensor2 const expected{tmech::dcontract(ctx->get<tensor4>("sf", "tangent"), eps)};
    EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("composite", "stress") - expected}), 0, 1e-12) << averaging;
  }
}

TEST(ShortFibre, StiffnessOrdering) {
  auto al{build(aligned)}, mo{build(mostly)}, ra{build(random3d)};
  auto const &Ca{al->get<tensor4>("sf", "tangent")};
  auto const &Cm{mo->get<tensor4>("sf", "tangent")};
  auto const &Cr{ra->get<tensor4>("sf", "tangent")};
  EXPECT_GT(Ca(0, 0, 0, 0), Cm(0, 0, 0, 0)); // along the fibres
  EXPECT_GT(Cm(0, 0, 0, 0), Cr(0, 0, 0, 0));
  EXPECT_GT(Cr(1, 1, 1, 1), Ca(1, 1, 1, 1)); // across them
  EXPECT_NEAR(Cm(1, 1, 1, 1), Cm(2, 2, 2, 2), 1e-12); // transverse isotropy of diag(0.8, 0.1, 0.1)
}

TEST(ShortFibre, RejectsBadInput) {
  EXPECT_THROW(build({0.5, 0.5, 0.5, 0, 0, 0}), std::invalid_argument); // trace 1.5
  EXPECT_THROW(build(aligned, "voigt"), std::invalid_argument);
  EXPECT_THROW(build(aligned, "concentration", 4), std::invalid_argument);
}

TEST(ShortFibre, FromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "constant_scalar", "name": "Km", "value": 3.0}, {"type": "constant_scalar", "name": "Gm", "value": 1.0},
    {"type": "constant_scalar", "name": "Kf", "value": 60.0}, {"type": "constant_scalar", "name": "Gf", "value": 30.0},
    {"type": "isotropic_tangent", "name": "Cm", "K_source": "Km", "G_source": "Gm"},
    {"type": "isotropic_tangent", "name": "Cf", "K_source": "Kf", "G_source": "Gf"},
    {"type": "isotropic_moduli", "name": "moduli", "tangent_source": "Cm"},
    {"type": "eshelby_spheroid", "name": "shape", "nu_source": "moduli", "nu_property": "poisson_ratio", "aspect_ratio": 10.0},
    {"type": "constant_scalar", "name": "c", "value": 0.2},
    {"type": "constant_scalar", "name": "cm", "value": 0.8},
    {"type": "phase", "name": "fibres", "tangent_source": "Cf", "eshelby_source": "shape", "fraction_source": "c"},
    {"type": "phase", "name": "matrix", "tangent_source": "Cm", "fraction_source": "cm"},
    {"type": "short_fibre_composite", "name": "sf", "matrix": "matrix", "fibre": "fibres", "orientation": [0.8, 0.1, 0.1, 0, 0, 0]},
    {"type": "linear_stress", "name": "composite", "tangent_source": "sf", "strain_source": "strain"}]})")(ctx, {});
  ctx.update();
  auto const &C{ctx.get<tensor4>("sf", "tangent")};
  EXPECT_GT(C(0, 0, 0, 0), C(1, 1, 1, 1));
}

} // namespace
