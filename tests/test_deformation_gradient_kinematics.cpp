#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/materials/deformation_gradient_kinematics.h"
#include "numsim-materials/materials/neo_hooke.h"
#include "numsim-materials/umat/statev_map.h"

/// deformation_gradient_kinematics: F^-1 and det F of a deformation gradient, produced in the graph.

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;

struct kinematics {
  ctx_type ctx;
  nm::external_deformation_gradient_source<policy> *F{nullptr};

  kinematics() {
    param_type p;
    p.insert<std::string>("name", "F");
    F = &ctx.create<nm::external_deformation_gradient_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "kinematics");
    p.insert<std::string>("deformation_gradient_source", "F");
    ctx.create<nm::deformation_gradient_kinematics<policy>>(p);
    ctx.finalize();
  }

  tensor2 const &inverse() { return ctx.get<tensor2>("kinematics", "inverse"); }
  T determinant() { return ctx.get<T>("kinematics", "determinant"); }
};

tensor2 deformation() {
  tensor2 F{tmech::eye<T, 3, 2>()};
  F(0, 1) = 0.3;
  F(1, 1) = 1.1;
  F(2, 0) = -0.2;
  F(2, 2) = 0.95;
  return F;
}

TEST(DeformationGradientKinematics, TheInverseAndTheDeterminantOfTheBoundF) {
  kinematics k;
  k.ctx.update();
  EXPECT_EQ(k.determinant(), 1.0) << "the source starts at the identity";
  EXPECT_EQ(tmech::norm(tensor2{k.inverse() - tmech::eye<T, 3, 2>()}), 0.0);
  tensor2 const F{deformation()};
  k.F->bind(tmech::eye<T, 3, 2>(), F);
  k.ctx.update();
  EXPECT_DOUBLE_EQ(k.determinant(), tmech::det(F));
  EXPECT_LT(tmech::norm(tensor2{k.inverse() * F - tmech::eye<T, 3, 2>()}), 1e-15);
}

TEST(DeformationGradientKinematics, FollowAnFWrittenWithoutBind) {
  kinematics k;
  k.ctx.update();
  tensor2 const F{deformation()};
  k.ctx.get_mutable<tensor2>("F", "deformation_gradient") = F;
  k.ctx.update();
  EXPECT_DOUBLE_EQ(k.determinant(), tmech::det(F)) << "a value of the graph, not of bind()";
  EXPECT_LT(tmech::norm(tensor2{k.inverse() * F - tmech::eye<T, 3, 2>()}), 1e-15);
}

TEST(DeformationGradientKinematics, ASingularFHasANaNInverse) {
  kinematics k;
  tensor2 F{tmech::eye<T, 3, 2>()};
  F(2, 2) = 0;
  k.F->bind(tmech::eye<T, 3, 2>(), F);
  k.ctx.update();
  EXPECT_EQ(k.determinant(), 0.0) << "neo_hooke checks det F";
  EXPECT_TRUE(std::isnan(k.inverse()(0, 0)));
}

TEST(DeformationGradientKinematics, AreNotHistoryAndTheSourceHoldsOnlyF) {
  kinematics k;
  EXPECT_EQ(k.ctx.find_property("F", "inverse"), nullptr);
  EXPECT_EQ(k.ctx.find_property("F", "determinant"), nullptr);
  nm::umat::statev_map<policy> const map{k.ctx, {}};
  ASSERT_EQ(map.describe(), std::vector<std::string>{"0..8  F::deformation_gradient  [full] [two-point]"})
      << "plain properties of the current F: nothing in STATEV";
}

/// F, its kinematics and a neo_hooke reading both.
struct hyperelastic {
  ctx_type ctx;
  nm::external_deformation_gradient_source<policy> *F{nullptr};

  hyperelastic() {
    param_type p;
    p.insert<std::string>("name", "F");
    F = &ctx.create<nm::external_deformation_gradient_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "kinematics");
    p.insert<std::string>("deformation_gradient_source", "F");
    ctx.create<nm::deformation_gradient_kinematics<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "nh");
    p.insert<std::string>("deformation_gradient_source", "F");
    p.insert<std::string>("kinematics_source", "kinematics");
    p.insert<T>("K", T{0.833});
    p.insert<T>("G", T{0.386});
    ctx.create<nm::neo_hooke<policy>>(p);
    ctx.finalize();
  }

  tensor2 const &stress() { return ctx.get<tensor2>("nh", "stress"); }
};

TEST(DeformationGradientKinematics, AConsumerFollowsAnFWrittenWithoutBind) {
  hyperelastic bound, written;
  bound.F->bind(tmech::eye<T, 3, 2>(), deformation());
  bound.ctx.update();
  written.ctx.update();
  written.ctx.get_mutable<tensor2>("F", "deformation_gradient") = deformation();
  written.ctx.update();
  EXPECT_EQ(tmech::norm(tensor2{written.stress() - bound.stress()}), 0.0);
}

} // namespace
