#include <gtest/gtest.h>
#include <memory>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/isotropize.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/materials/inclusion.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/mean_field_composite.h"
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

std::unique_ptr<ctx_type> build(std::vector<T> const &orientation, std::string const &averaging = "concentration",
                                std::size_t precision = 15) {
  auto owner{std::make_unique<ctx_type>()};
  auto &ctx{*owner};
  param_type p;
  p.insert<std::string>("name", "strain");
  ctx.create<nm::external_strain_source<policy>>(p);
  for (auto const &[name, K, G] : {std::tuple{"matrix", Km, Gm}, std::tuple{"fibre", Kf, Gf}}) {
    p.clear();
    p.insert<std::string>("name", name);
    p.insert<std::string>("strain_producer_name", "strain");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    ctx.create<nm::linear_elasticity<policy>>(p);
  }
  // the unidirectional reference: a spheroid along x1 in a mean_field_composite
  p.clear();
  p.insert<std::string>("name", "spheroid");
  p.insert<std::string>("material", "fibre");
  p.insert<std::string>("shape", "spheroid");
  p.insert<T>("volume_fraction", c);
  p.insert<T>("aspect_ratio", ar);
  p.insert<std::vector<T>>("direction", {1, 0, 0});
  ctx.create<nm::inclusion<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "ud");
  p.insert<std::string>("matrix", "matrix");
  p.insert<std::vector<std::string>>("inclusions", {"spheroid"});
  p.insert<std::string>("kernel", "mori_tanaka");
  p.insert<std::string>("strain_source", "strain");
  ctx.create<nm::mean_field_composite<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "sf");
  p.insert<std::string>("matrix", "matrix");
  p.insert<std::string>("fibre", "fibre");
  p.insert<T>("volume_fraction", c);
  p.insert<T>("aspect_ratio", ar);
  p.insert<std::vector<T>>("orientation", orientation);
  p.insert<std::string>("averaging", averaging);
  p.insert<int>("precision", static_cast<int>(precision));
  p.insert<std::string>("strain_source", "strain");
  ctx.create<nm::short_fibre_composite<policy>>(p);
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
    EXPECT_LE(diff(C, nm::isotropic_part<T, 3>(C)), diff(coarse->get<tensor4>("sf", "tangent"), nm::isotropic_part<T, 3>(coarse->get<tensor4>("sf", "tangent"))) + 1e-15);
  }
}

TEST(ShortFibre, ConcentrationsAverageToIdentityAndStressIsConsistent) {
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
    EXPECT_NEAR(tmech::norm(tensor2{ctx->get<tensor2>("sf", "stress") - expected}), 0, 1e-12) << averaging;
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
    {"type": "linear_elasticity", "name": "m", "strain_producer_name": "strain", "K": 3.0, "G": 1.0},
    {"type": "linear_elasticity", "name": "f", "strain_producer_name": "strain", "K": 60.0, "G": 30.0},
    {"type": "short_fibre_composite", "name": "sf", "matrix": "m", "fibre": "f", "volume_fraction": 0.2, "aspect_ratio": 10.0,
     "orientation": [0.8, 0.1, 0.1, 0, 0, 0], "strain_source": "strain"}]})")(ctx, {});
  ctx.update();
  auto const &C{ctx.get<tensor4>("sf", "tangent")};
  EXPECT_GT(C(0, 0, 0, 0), C(1, 1, 1, 1));
}

} // namespace
