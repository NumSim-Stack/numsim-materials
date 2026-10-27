#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/depolarisation_cylinder.h"
#include "numsim-materials/materials/depolarisation_ellipsoid.h"
#include "numsim-materials/materials/depolarisation_sphere.h"
#include "numsim-materials/materials/depolarisation_spheroid.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
namespace es = numsim::materials::eshelby;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using vector = tmech::tensor<T, 3, 1>;

T diff(tensor2 const &a, tensor2 const &b) { return tmech::norm(tensor2{a - b}) / tmech::norm(a); }

template <typename M>
tensor2 const &make(ctx_type &ctx, param_type p, std::string const &name) {
  p.insert<std::string>("name", name);
  ctx.create<M>(p);
  ctx.finalize();
  ctx.update();
  return ctx.get<tensor2>(name, "tensor");
}

TEST(DepolarisationMaterials, SphereIsAThirdOfTheIdentity) {
  ctx_type ctx;
  EXPECT_NEAR(diff(make<nm::depolarisation_sphere<policy>>(ctx, {}, "sphere"), tmech::eye<T, 3, 2>() / 3), 0, 1e-15);
}

TEST(DepolarisationMaterials, CylinderIsTransverseToItsDirection) {
  ctx_type ctx;
  param_type p;
  p.insert<std::vector<T>>("direction", {0, 0, 2});
  vector d;
  d(2) = 1;
  EXPECT_NEAR(diff(make<nm::depolarisation_cylinder<policy>>(ctx, p, "fibre"), tensor2{(tmech::eye<T, 3, 2>() - tmech::otimes(d, d)) / 2}), 0, 1e-15);
}

TEST(DepolarisationMaterials, SpheroidIsTheRotatedClosedForm) {
  ctx_type ctx;
  param_type p;
  p.insert<T>("aspect_ratio", 3.0);
  p.insert<std::vector<T>>("direction", {1, 1, 1});
  vector d;
  d(0) = d(1) = d(2) = 1 / std::sqrt(3.0);
  tensor2 const expected{nm::rotate(es::depolarisation(3.0, 1.0, 1.0), nm::rotation_to(d))};
  tensor2 const N{make<nm::depolarisation_spheroid<policy>>(ctx, p, "spheroid")};
  EXPECT_NEAR(diff(N, expected), 0, 1e-14);
  EXPECT_NEAR(tmech::trace(N), 1, 1e-14);
}

TEST(DepolarisationMaterials, EllipsoidAlongTheGlobalAxes) {
  ctx_type ctx;
  param_type p;
  p.insert<std::vector<T>>("semi_axes", {3, 2, 1});
  tensor2 const N{make<nm::depolarisation_ellipsoid<policy>>(ctx, p, "ellipsoid")};
  EXPECT_NEAR(diff(N, es::depolarisation(3.0, 2.0, 1.0)), 0, 1e-15);
}

TEST(DepolarisationMaterials, RejectsInvalidGeometry) {
  {
    ctx_type ctx;
    param_type p;
    p.insert<T>("aspect_ratio", -1.0);
    EXPECT_THROW(make<nm::depolarisation_spheroid<policy>>(ctx, p, "s"), std::invalid_argument);
  }
  {
    ctx_type ctx;
    param_type p;
    p.insert<std::vector<T>>("semi_axes", {1, 1});
    EXPECT_THROW(make<nm::depolarisation_ellipsoid<policy>>(ctx, p, "e"), std::invalid_argument);
  }
}

TEST(DepolarisationMaterials, FromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "depolarisation_sphere", "name": "sphere"},
    {"type": "depolarisation_cylinder", "name": "cylinder", "direction": [0, 1, 0]},
    {"type": "depolarisation_spheroid", "name": "spheroid", "aspect_ratio": 5, "direction": [1, 0, 0]},
    {"type": "depolarisation_ellipsoid", "name": "ellipsoid", "semi_axes": [1, 2, 3]}]})")(ctx, {});
  ctx.update();
  EXPECT_NEAR(ctx.get<tensor2>("cylinder", "tensor")(1, 1), 0, 1e-15);
  EXPECT_NEAR(diff(ctx.get<tensor2>("spheroid", "tensor"), es::depolarisation(5.0, 1.0, 1.0)), 0, 1e-14);
  EXPECT_NEAR(tmech::trace(ctx.get<tensor2>("ellipsoid", "tensor")), 1, 1e-14);
}

} // namespace
