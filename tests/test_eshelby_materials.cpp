#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/eshelby_cylinder.h"
#include "numsim-materials/materials/eshelby_ellipsoid.h"
#include "numsim-materials/materials/eshelby_sphere.h"
#include "numsim-materials/materials/eshelby_spheroid.h"
#include "numsim-materials/materials/isotropic_moduli.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/poisson_ratio_from_moduli.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
namespace es = numsim::materials::eshelby;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor4 = tmech::tensor<T, 3, 4>;
using vector = tmech::tensor<T, 3, 1>;

constexpr T K{100}, G{40};
T const nu{nm::poisson_ratio(K, G)};

T diff(tensor4 const &a, tensor4 const &b) { return tmech::norm(tensor4{a - b}) / tmech::norm(a); }

/// A fixed Poisson's ratio and the Eshelby nodes consuming it.
struct point {
  std::unique_ptr<ctx_type> owner{std::make_unique<ctx_type>()};
  ctx_type &ctx{*owner};
  param_type p;

  point() {
    p.insert<std::string>("name", "nu");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    ctx.create<nm::poisson_ratio_from_moduli<policy>>(p);
    p.clear();
    p.insert<std::string>("nu_source", "nu");
  }
  template <typename M>
  point &add(std::string const &name) {
    p.insert<std::string>("name", name);
    ctx.create<M>(p);
    return *this;
  }
  tensor4 const &tensor(std::string const &name) {
    ctx.finalize();
    ctx.update();
    return ctx.get<tensor4>(name, "tensor");
  }
};

TEST(EshelbyMaterials, SphereConsumesTheProducedPoissonsRatio) {
  point pt;
  pt.add<nm::eshelby_sphere<policy>>("sphere");
  EXPECT_NEAR(diff(pt.tensor("sphere"), es::sphere(nu)), 0, 1e-14);
  EXPECT_NEAR(pt.ctx.get<T>("nu", "value"), nu, 1e-15);
}

TEST(EshelbyMaterials, CylinderAlongADirection) {
  point pt;
  pt.p.insert<std::vector<T>>("direction", {0, 2, 0}); // normalised by the material
  pt.add<nm::eshelby_cylinder<policy>>("fibre");
  EXPECT_NEAR(diff(pt.tensor("fibre"), es::cylinder(nu, std::size_t{1})), 0, 1e-14);
}

TEST(EshelbyMaterials, SpheroidIsTheRotatedProlateEllipsoid) {
  point pt;
  pt.p.insert<T>("aspect_ratio", 5.0);
  pt.p.insert<std::vector<T>>("direction", {1, 1, 1});
  pt.add<nm::eshelby_spheroid<policy>>("spheroid");
  vector d;
  d(0) = d(1) = d(2) = 1 / std::sqrt(3.0);
  tensor4 const expected{nm::rotate(es::ellipsoid(nu, 5.0, 1.0, 1.0), nm::rotation_to(d))};
  EXPECT_NEAR(diff(pt.tensor("spheroid"), expected), 0, 1e-13);
}

TEST(EshelbyMaterials, EllipsoidAlongTheGlobalAxes) {
  point pt;
  pt.p.insert<std::vector<T>>("semi_axes", {3, 2, 1});
  pt.add<nm::eshelby_ellipsoid<policy>>("ellipsoid");
  EXPECT_NEAR(diff(pt.tensor("ellipsoid"), es::ellipsoid(nu, 3.0, 2.0, 1.0)), 0, 1e-14);
}

// the node is a consumer: when its Poisson's ratio changes, the engine re-evaluates it
TEST(EshelbyMaterials, FollowsAChangingPoissonsRatio) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "nu_in");
  ctx.create<nm::external_scalar_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "sphere");
  p.insert<std::string>("nu_source", "nu_in");
  p.insert<std::string>("nu_property", "state");
  ctx.create<nm::eshelby_sphere<policy>>(p);
  ctx.finalize();
  auto &source{*dynamic_cast<nm::external_scalar_source<policy> *>(ctx.find("nu_in"))};
  for (T const value : {0.2, 0.35, 0.1}) {
    source.bind(value, value);
    ctx.update();
    EXPECT_NEAR(diff(ctx.get<tensor4>("sphere", "tensor"), es::sphere(value)), 0, 1e-14) << value;
  }
}

// a matrix tangent that is itself a product: isotropic_moduli turns it into K, G and nu
TEST(EshelbyMaterials, PoissonsRatioFromAnIsotropicTangent) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "strain");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "matrix");
  p.insert<std::string>("strain_producer_name", "strain");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  ctx.create<nm::linear_elasticity<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "moduli");
  p.insert<std::string>("tangent_source", "matrix");
  ctx.create<nm::isotropic_moduli<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "sphere");
  p.insert<std::string>("nu_source", "moduli");
  p.insert<std::string>("nu_property", "poisson_ratio");
  ctx.create<nm::eshelby_sphere<policy>>(p);
  ctx.finalize();
  ctx.update();
  EXPECT_NEAR(ctx.get<T>("moduli", "K"), K, 1e-12);
  EXPECT_NEAR(ctx.get<T>("moduli", "G"), G, 1e-12);
  EXPECT_NEAR(diff(ctx.get<tensor4>("sphere", "tensor"), es::sphere(nu)), 0, 1e-14);
}

TEST(EshelbyMaterials, RejectsInvalidGeometry) {
  {
    point pt;
    pt.p.insert<T>("aspect_ratio", 0.0);
    EXPECT_THROW(pt.add<nm::eshelby_spheroid<policy>>("s"), std::invalid_argument);
  }
  {
    point pt;
    pt.p.insert<std::vector<T>>("direction", {0, 0, 0});
    EXPECT_THROW(pt.add<nm::eshelby_cylinder<policy>>("c"), std::invalid_argument);
  }
  {
    point pt;
    pt.p.insert<std::vector<T>>("semi_axes", {1, 2});
    EXPECT_THROW(pt.add<nm::eshelby_ellipsoid<policy>>("e"), std::invalid_argument);
  }
}

TEST(EshelbyMaterials, FromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "poisson_ratio_from_moduli", "name": "nu", "K": 100, "G": 40},
    {"type": "eshelby_sphere", "name": "sphere", "nu_source": "nu"},
    {"type": "eshelby_cylinder", "name": "cylinder", "nu_source": "nu", "direction": [0, 0, 1]},
    {"type": "eshelby_spheroid", "name": "spheroid", "nu_source": "nu", "aspect_ratio": 10, "direction": [1, 0, 0]},
    {"type": "eshelby_ellipsoid", "name": "ellipsoid", "nu_source": "nu", "semi_axes": [1, 2, 3]}]})")(ctx, {});
  ctx.update();
  EXPECT_NEAR(diff(ctx.get<tensor4>("sphere", "tensor"), es::sphere(nu)), 0, 1e-14);
  EXPECT_NEAR(diff(ctx.get<tensor4>("cylinder", "tensor"), es::cylinder(nu, std::size_t{2})), 0, 1e-14);
  EXPECT_NEAR(diff(ctx.get<tensor4>("spheroid", "tensor"), es::ellipsoid(nu, 10.0, 1.0, 1.0)), 0, 1e-13);
  EXPECT_NEAR(diff(ctx.get<tensor4>("ellipsoid", "tensor"), es::ellipsoid(nu, 1.0, 2.0, 3.0)), 0, 1e-14);
}

} // namespace
