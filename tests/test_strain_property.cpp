#include <gtest/gtest.h>
#include <stdexcept>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/von_mises_state_function.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;

constexpr T K{100}, G{40};

/// A driven strain and an elastic material "a" on it; "a::stress" serves as a strain under another name.
struct point {
  ctx_type ctx;
  param_type p;
  tensor2 eps;

  point() {
    p.insert<std::string>("name", "strain");
    ctx.create<nm::external_strain_source<policy>>(p);
    elastic("a", "strain", "strain");
    eps(0, 1) = eps(1, 0) = 0.01; // deviatoric: a::stress = 2G eps
  }
  void elastic(std::string const &name, std::string const &source, std::string const &property) {
    p.clear();
    p.insert<std::string>("name", name);
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    p.insert<std::string>("strain_producer_name", source);
    if (property != "strain")
      p.insert<std::string>("strain_property", property);
    ctx.create<nm::linear_elasticity<policy>>(p);
  }
  void run() {
    ctx.finalize();
    dynamic_cast<nm::external_strain_source<policy> &>(*ctx.find("strain")).bind(tensor2{}, eps);
    ctx.update();
  }
  T diff(tensor2 const &x, tensor2 const &y) const { return tmech::norm(tensor2{x - y}) / tmech::norm(x); }
};

TEST(StrainProperty, LinearElasticityReadsTheNamedProperty) {
  point pt;
  pt.elastic("b", "a", "stress");
  pt.run();
  tensor2 const expected{4 * G * G * pt.eps}; // C : (2G eps) with eps deviatoric
  EXPECT_NEAR(pt.diff(pt.ctx.get<tensor2>("b", "stress"), expected), 0, 1e-14);
  EXPECT_NEAR(pt.diff(pt.ctx.get<tensor2>("a", "stress"), tensor2{2 * G * pt.eps}), 0, 1e-14); // default unchanged
}

TEST(StrainProperty, LinearStressReadsTheNamedProperty) {
  point pt;
  pt.p.clear();
  pt.p.insert<std::string>("name", "b");
  pt.p.insert<std::string>("tangent_source", "a");
  pt.p.insert<std::string>("strain_source", "a");
  pt.p.insert<std::string>("strain_property", "stress");
  pt.ctx.create<nm::linear_stress<policy>>(pt.p);
  pt.run();
  EXPECT_NEAR(pt.diff(pt.ctx.get<tensor2>("b", "stress"), tensor2{4 * G * G * pt.eps}), 0, 1e-14);
}

// a state function is homogeneous of degree one: on 2G eps it gives 2G times its value on eps
TEST(StrainProperty, StateFunctionReadsTheNamedProperty) {
  point pt;
  for (auto const &[name, source, property] :
       {std::tuple{"on_strain", "strain", "strain"}, std::tuple{"on_stress", "a", "stress"}}) {
    pt.p.clear();
    pt.p.insert<std::string>("name", name);
    pt.p.insert<std::string>("strain_source", source);
    pt.p.insert<std::string>("strain_property", property);
    pt.ctx.create<nm::von_mises_state_function<policy>>(pt.p);
  }
  pt.run();
  T const on_strain{pt.ctx.get<T>("on_strain", "equivalent_strain")};
  ASSERT_GT(on_strain, 0);
  EXPECT_NEAR(pt.ctx.get<T>("on_stress", "equivalent_strain") / on_strain, 2 * G, 1e-10);
}

TEST(StrainProperty, RejectsAMissingProperty) {
  point pt;
  pt.elastic("b", "a", "no_such_property");
  EXPECT_THROW(pt.ctx.finalize(), std::runtime_error);
}

TEST(StrainProperty, FromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "linear_elasticity", "name": "a", "strain_producer_name": "strain", "K": 100, "G": 40},
    {"type": "linear_elasticity", "name": "b", "strain_producer_name": "a", "strain_property": "stress", "K": 100, "G": 40}]})")(ctx, {});
  tensor2 eps;
  eps(0, 1) = eps(1, 0) = 0.01;
  dynamic_cast<nm::external_strain_source<policy> &>(*ctx.find("strain")).bind(tensor2{}, eps);
  ctx.update();
  EXPECT_NEAR(tmech::norm(tensor2{ctx.get<tensor2>("b", "stress") - 4 * G * G * eps}), 0, 1e-12);
}

} // namespace
