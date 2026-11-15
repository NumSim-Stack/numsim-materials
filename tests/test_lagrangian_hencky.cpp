#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <string>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/driver/deformation_gradient_driver.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/elastic_strain.h"
#include "numsim-materials/materials/isotropic_tangent.h"
#include "numsim-materials/materials/kirchhoff_to_first_piola.h"
#include "numsim-materials/materials/lagrangian_log_strain.h"
#include "numsim-materials/materials/lagrangian_stress_to_first_piola.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/log_strain.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor1 = tmech::tensor<T, 3, 1>;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using driver = nm::deformation_gradient_driver<policy>;
using control = nm::deformation_control<T>;

constexpr T K{166.67}, G{76.92}, fibre{300.0};

/// A constant transversely isotropic stiffness: isotropic plus a fibre term along x1 (test only).
template <typename Traits>
class fibre_tangent final : public nm::material_base<fibre_tangent<Traits>, Traits> {
public:
  using base = nm::material_base<fibre_tangent<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit fibre_tangent(Args &&...args) : base(std::forward<Args>(args)...), m_C(base::template add_output<tensor4>("tangent")) {
    base::add_callback(&fibre_tangent::compute, m_C);
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() {
    tensor1 a;
    a(0) = 1;
    tensor2 const A{tmech::otimes(a, a)};
    m_C = nm::isotropic_stiffness<T, 3>(K, G) + fibre * tmech::otimes(A, A);
  }

private:
  tensor4 &m_C;
};

param_type named(std::string const &name) {
  param_type p;
  p.insert<std::string>("name", name);
  return p;
}

/// F -> strain -> linear_stress on `tangent` -> pull-back ("P"), built by hand. `lagrangian` picks
/// lagrangian_log_strain + lagrangian_stress_to_first_piola, otherwise log_strain + kirchhoff_to_first_piola.
std::unique_ptr<ctx_type> chain(bool lagrangian, bool anisotropic) {
  auto ctx{std::make_unique<ctx_type>()};
  auto pF{named("F")};
  ctx->create<nm::external_deformation_gradient_source<policy>>(pF);
  if (anisotropic) {
    auto pC{named("stiffness")};
    ctx->create<fibre_tangent<policy>>(pC);
  } else {
    auto pK{named("K")}, pG{named("G")}, pC{named("stiffness")};
    pK.insert<T>("value", K);
    pG.insert<T>("value", G);
    pC.insert<std::string>("K_source", "K");
    pC.insert<std::string>("G_source", "G");
    ctx->create<nm::constant_scalar<policy>>(pK);
    ctx->create<nm::constant_scalar<policy>>(pG);
    ctx->create<nm::isotropic_tangent<policy>>(pC);
  }
  auto ps{named("strain")};
  ps.insert<std::string>("deformation_gradient_source", "F");
  if (lagrangian) {
    ctx->create<nm::lagrangian_log_strain<policy>>(ps);
  } else {
    ctx->create<nm::log_strain<policy>>(ps);
  }
  auto pt{named("stress")};
  pt.insert<std::string>("tangent_source", "stiffness");
  pt.insert<std::string>("strain_source", "strain");
  ctx->create<nm::linear_stress<policy>>(pt);
  auto pp{named("P")};
  pp.insert<std::string>("stress_source", "stress");
  pp.insert<std::string>("tangent_source", "stiffness");
  pp.insert<std::string>("strain_source", "strain");
  pp.insert<std::string>("deformation_gradient_source", "F");
  if (lagrangian) {
    ctx->create<nm::lagrangian_stress_to_first_piola<policy>>(pp);
  } else {
    ctx->create<nm::kirchhoff_to_first_piola<policy>>(pp);
  }
  ctx->finalize();
  return ctx;
}

/// The isotropic Lagrangian chain as a JSON deck, with an elastic_strain between strain and stress.
std::unique_ptr<ctx_type> deck() {
  auto owner{std::make_unique<ctx_type>()};
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_deformation_gradient_source", "name": "F"},
    {"type": "constant_scalar", "name": "K", "value": 166.67},
    {"type": "constant_scalar", "name": "G", "value": 76.92},
    {"type": "isotropic_tangent", "name": "stiffness", "K_source": "K", "G_source": "G"},
    {"type": "lagrangian_log_strain", "name": "strain", "deformation_gradient_source": "F"},
    {"type": "elastic_strain", "name": "elastic_strain", "strain_source": "strain", "eigenstrain_sources": []},
    {"type": "linear_stress", "name": "stress", "tangent_source": "stiffness", "strain_source": "elastic_strain"},
    {"type": "lagrangian_stress_to_first_piola", "name": "P", "stress_source": "stress", "tangent_source": "stiffness",
     "strain_source": "strain", "deformation_gradient_source": "F"}]})")(*owner, {});
  return owner;
}

param_type point() {
  param_type p;
  p.insert<std::string>("deformation_gradient_source", "F");
  p.insert<std::string>("stress_source", "P");
  return p;
}

void bind(ctx_type &ctx, tensor2 const &F) {
  dynamic_cast<nm::external_deformation_gradient_source<policy> *>(ctx.find("F"))->bind(tmech::eye<T, 3, 2>(), F);
  ctx.update();
}

tensor2 stretch(T l1, T l2 = 1, T l3 = 1) {
  tensor2 F;
  F(0, 0) = l1;
  F(1, 1) = l2;
  F(2, 2) = l3;
  return F;
}

tensor2 rotation() {
  tensor1 axis;
  axis(0) = 1, axis(1) = 2, axis(2) = 2;
  return nm::rotation(tensor1{axis / 3}, T{0.7});
}

/// A general F: rotated, sheared, with three distinct stretches.
tensor2 general_F() {
  tensor2 U{stretch(1.25, 0.9, 1.1)};
  U(0, 1) = U(1, 0) = 0.15;
  tensor2 F{rotation() * U};
  F(0, 1) += 0.2;
  return F;
}

T rel(tensor2 const &a, tensor2 const &b) { return tmech::norm(tensor2{a - b}) / tmech::norm(b); }

TEST(LagrangianHencky, StrainIsHalfTheLogOfC) {
  auto ctx{chain(true, false)};
  tensor2 const F{general_F()};
  bind(*ctx, F);
  tensor2 const C{tmech::trans(F) * F};
  tensor2 const E{T{0.5} * tmech::log(C)};
  EXPECT_LE(tmech::norm(tensor2{ctx->get<tensor2>("strain", "strain") - E}), 1e-14);
}

// for an isotropic law ln U and ln V share eigenvalues and T = R^T tau R: both chains give the same P and dP/dF
TEST(LagrangianHencky, IsotropicMatchesTheEulerianChain) {
  auto lag{deck()};
  auto eul{chain(false, false)};
  for (auto const &F : {general_F(), stretch(1.2, 0.95, 1.05), stretch(1.15, 1.15, 1.0)}) {
    bind(*lag, F);
    bind(*eul, F);
    EXPECT_LE(rel(lag->get<tensor2>("P", "stress"), eul->get<tensor2>("P", "stress")), 1e-12);
    tensor4 const dA{lag->get<tensor4>("P", "tangent") - eul->get<tensor4>("P", "tangent")};
    EXPECT_LE(tmech::norm(dA) / tmech::norm(eul->get<tensor4>("P", "tangent")), 1e-10);
  }
}

TEST(LagrangianHencky, TangentMatchesFiniteDifferences) {
  for (bool const anisotropic : {false, true}) {
    auto ctx{chain(true, anisotropic)};
    auto p{point()};
    p.insert<T>("fd_step", 1e-5); // the default 1e-7 splits an exact pair and loses eps/h: 1e-4 at a rotated pair
    driver d(*ctx, p);
    // general, repeated (the second derivative of tmech#41) and all-equal stretches
    for (auto const &F : {general_F(), stretch(1.15, 1.15, 1.0), tensor2{rotation() * stretch(1.15, 1.15, 1.0)}, stretch(1.2, 1.2, 1.2),
                          stretch(1.15, 1.15 + 1e-9, 1.0)}) {
      auto const res{d.step(F, tensor2{}, control::deformation(), 0, 1)};
      ASSERT_TRUE(res.converged);
      EXPECT_LT(d.tangent_error(res), 1e-6) << anisotropic << " " << F(0, 0) << " " << F(1, 1) << " " << F(0, 1);
    }
  }
}

TEST(LagrangianHencky, AnisotropicIsObjective) {
  auto ctx{chain(true, true)};
  tensor2 U{stretch(1.3, 0.9, 1.1)};
  U(0, 1) = U(1, 0) = 0.1;
  bind(*ctx, U);
  tensor2 const P_U{ctx->get<tensor2>("P", "stress")};
  bind(*ctx, tensor2{rotation() * U});
  EXPECT_LE(rel(ctx->get<tensor2>("P", "stress"), tensor2{rotation() * P_U}), 1e-12); // P(R U) = R P(U)
  bind(*ctx, rotation());
  EXPECT_LE(tmech::norm(ctx->get<tensor2>("P", "stress")), 1e-10 * K); // a rotation is stress free
}

// why the Lagrangian chain exists: the Eulerian one with the same anisotropic C is not objective
TEST(LagrangianHencky, AnisotropicEulerianChainIsNotObjective) {
  auto ctx{chain(false, true)};
  tensor2 U{stretch(1.3, 0.9, 1.1)};
  U(0, 1) = U(1, 0) = 0.1;
  bind(*ctx, U);
  tensor2 const P_U{ctx->get<tensor2>("P", "stress")};
  bind(*ctx, tensor2{rotation() * U});
  EXPECT_GT(rel(ctx->get<tensor2>("P", "stress"), tensor2{rotation() * P_U}), 1e-2);
}

TEST(LagrangianHencky, AnisotropicSmallStrainLimitIsLinearElasticity) {
  auto ctx{chain(true, true)};
  tensor2 F{tmech::eye<T, 3, 2>()};
  F(0, 0) = 1 + 1e-6;
  F(0, 1) = 2e-6;
  F(1, 2) = -1e-6;
  bind(*ctx, F);
  tensor1 a;
  a(0) = 1;
  tensor2 const A{tmech::otimes(a, a)};
  tensor4 const C{nm::isotropic_stiffness<T, 3>(K, G) + fibre * tmech::otimes(A, A)};
  tensor2 const sigma{tmech::dcontract(C, tensor2{tmech::sym(tensor2{F - tmech::eye<T, 3, 2>()})})};
  EXPECT_LE(rel(ctx->get<tensor2>("P", "stress"), sigma), 1e-5);
}

} // namespace
