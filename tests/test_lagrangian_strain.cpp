#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/driver/deformation_gradient_driver.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/elastic_strain.h"
#include "numsim-materials/materials/isotropic_tangent.h"
#include "numsim-materials/materials/deformation_gradient_kinematics.h"
#include "numsim-materials/materials/kirchhoff_to_first_piola.h"
#include "numsim-materials/materials/lagrangian_stress_to_first_piola.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/log_strain.h"
#include "numsim-materials/materials/saint_venant_kirchhoff.h"
#include "numsim-materials/materials/seth_hill_strain.h"
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
using tensor6 = tmech::tensor<T, 3, 6>;
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

enum class pull_back { none, lagrangian, eulerian };

/// F -> strain -> linear_stress on `stiffness` -> "P". lagrangian: seth_hill_strain(m) + lagrangian_stress_to_first_piola;
/// eulerian: log_strain + kirchhoff_to_first_piola; none: the strain node alone.
std::unique_ptr<ctx_type> chain(T m, pull_back kind, bool anisotropic = false) {
  auto ctx{std::make_unique<ctx_type>()};
  auto pF{named("F")};
  ctx->create<nm::external_deformation_gradient_source<policy>>(pF);
  auto ps{named("strain")};
  ps.insert<std::string>("deformation_gradient_source", "F");
  if (kind == pull_back::eulerian) {
    ctx->create<nm::log_strain<policy>>(ps);
  } else {
    ps.insert<T>("m", m);
    ctx->create<nm::seth_hill_strain<policy>>(ps);
  }
  if (kind != pull_back::none) {
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
    auto pt{named("stress")};
    pt.insert<std::string>("tangent_source", "stiffness");
    pt.insert<std::string>("strain_source", "strain");
    ctx->create<nm::linear_stress<policy>>(pt);
    auto pp{named("P")};
    pp.insert<std::string>("stress_source", "stress");
    pp.insert<std::string>("tangent_source", "stiffness");
    pp.insert<std::string>("strain_source", "strain");
    if (kind == pull_back::lagrangian) {
      pp.insert<std::string>("deformation_gradient_source", "F");
      ctx->create<nm::lagrangian_stress_to_first_piola<policy>>(pp);
    } else {
      auto pk{named("kinematics")};
      pk.insert<std::string>("deformation_gradient_source", "F");
      ctx->create<nm::deformation_gradient_kinematics<policy>>(pk);
      pp.insert<std::string>("kinematics_source", "kinematics");
      ctx->create<nm::kirchhoff_to_first_piola<policy>>(pp);
    }
  }
  ctx->finalize();
  return ctx;
}

/// The isotropic Hencky chain as a JSON deck, with an elastic_strain between strain and stress.
std::unique_ptr<ctx_type> deck() {
  auto owner{std::make_unique<ctx_type>()};
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_deformation_gradient_source", "name": "F"},
    {"type": "constant_scalar", "name": "K", "value": 166.67},
    {"type": "constant_scalar", "name": "G", "value": 76.92},
    {"type": "isotropic_tangent", "name": "stiffness", "K_source": "K", "G_source": "G"},
    {"type": "seth_hill_strain", "name": "strain", "m": 0.0, "deformation_gradient_source": "F"},
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

template <typename A>
T rel(A const &a, A const &b) {
  return tmech::norm(A{a - b}) / tmech::norm(b);
}

constexpr T measures[]{0.0, 1.0, 2.0, -2.0, 0.5};

TEST(LagrangianStrain, StrainIsTheSethHillMeasure) {
  tensor2 const F{general_F()}, C{tmech::trans(F) * F}, I{tmech::eye<T, 3, 2>()};
  tensor2 const expected[]{T{0.5} * tmech::log(C), tensor2{tmech::sqrt(C)} - I, T{0.5} * (C - I), T{0.5} * (I - tmech::inv(C)),
                           T{2} * (tensor2{tmech::sqrt(tensor2{tmech::sqrt(C)})} - I)};
  for (std::size_t i{0}; i < std::size(measures); ++i) {
    auto ctx{chain(measures[i], pull_back::none)};
    bind(*ctx, F);
    EXPECT_LE(rel(ctx->get<tensor2>("strain", "strain"), expected[i]), 1e-13) << measures[i];
  }
}

// central differences in a symmetric direction of C, through F = sqrt(C); a step splitting the repeated pair loses
// eps/h in dE/dC, so h = 1e-4 there
TEST(LagrangianStrain, StrainDerivativesMatchFiniteDifferences) {
  for (T const m : measures) {
    auto ctx{chain(m, pull_back::lagrangian)};
    for (auto const &[C0, h] : {std::pair{tensor2{tmech::trans(general_F()) * general_F()}, T{1e-6}}, std::pair{tensor2{stretch(1.3, 1.3, 0.9)}, T{1e-4}}}) {
      auto const at{[&](tensor2 const &C) {
        bind(*ctx, tensor2{tmech::sqrt(C)});
        return std::pair{ctx->get<tensor2>("strain", "strain"), ctx->get<tensor4>("strain", "d_strain_d_right_cauchy_green")};
      }};
      bind(*ctx, tensor2{tmech::sqrt(C0)});
      tensor4 const dE{ctx->get<tensor4>("strain", "d_strain_d_right_cauchy_green")};
      tensor6 const d2E{ctx->get<tensor6>("strain", "d2_strain_d_right_cauchy_green2")};
      T err1{0}, err2{0}, scale2{0};
      for (std::size_t k{0}; k < 3; ++k)
        for (std::size_t l{0}; l < 3; ++l) {
          tensor2 dC;
          dC(k, l) += h / 2;
          dC(l, k) += h / 2;
          auto const [Ep, dEp] = at(tensor2{C0 + dC});
          auto const [Em, dEm] = at(tensor2{C0 - dC});
          tensor2 const fd1{(Ep - Em) / (2 * h)};
          tensor4 const fd2{(dEp - dEm) / (2 * h)};
          for (std::size_t a{0}; a < 3; ++a)
            for (std::size_t b{0}; b < 3; ++b) {
              err1 = std::max(err1, std::abs(dE(a, b, k, l) - fd1(a, b)));
              for (std::size_t c{0}; c < 3; ++c)
                for (std::size_t d{0}; d < 3; ++d) {
                  err2 = std::max(err2, std::abs(d2E(a, b, c, d, k, l) - fd2(a, b, c, d)));
                  scale2 = std::max(scale2, std::abs(fd2(a, b, c, d)));
                }
            }
        }
      EXPECT_LT(err1, 1e-8) << m << " " << h;
      EXPECT_LT(err2, 1e-6 * std::max(scale2, T{1})) << m << " " << h;
    }
  }
}

TEST(LagrangianStrain, SecondDerivativeOnlyWhenRead) {
  auto ctx{chain(1.0, pull_back::none)};
  bind(*ctx, general_F());
  EXPECT_THROW(ctx->get<tensor6>("strain", "d2_strain_d_right_cauchy_green2"), std::logic_error);
  auto read{chain(1.0, pull_back::lagrangian)};
  bind(*read, general_F());
  EXPECT_GT(tmech::norm(read->get<tensor6>("strain", "d2_strain_d_right_cauchy_green2")), 0.1);
}

// for an isotropic law ln U and ln V share eigenvalues and T = R^T tau R: both chains give the same P and dP/dF
TEST(LagrangianStrain, HenckyMatchesTheEulerianChain) {
  auto lag{deck()};
  auto eul{chain(0, pull_back::eulerian)};
  for (auto const &F : {general_F(), stretch(1.2, 0.95, 1.05), stretch(1.15, 1.15, 1.0)}) {
    bind(*lag, F);
    bind(*eul, F);
    EXPECT_LE(rel(lag->get<tensor2>("P", "stress"), eul->get<tensor2>("P", "stress")), 1e-12);
    EXPECT_LE(rel(lag->get<tensor4>("P", "tangent"), eul->get<tensor4>("P", "tangent")), 1e-10);
  }
}

TEST(LagrangianStrain, GreenLagrangeIsSaintVenantKirchhoff) {
  auto lag{chain(2.0, pull_back::lagrangian)};
  ctx_type svk;
  auto pF{named("F")}, ps{named("P")};
  svk.create<nm::external_deformation_gradient_source<policy>>(pF);
  ps.insert<T>("K", K);
  ps.insert<T>("G", G);
  ps.insert<std::string>("deformation_gradient_source", "F");
  svk.create<nm::saint_venant_kirchhoff<policy>>(ps);
  svk.finalize();
  for (auto const &F : {general_F(), stretch(1.2, 0.95, 1.05)}) {
    bind(*lag, F);
    bind(svk, F);
    EXPECT_LE(rel(lag->get<tensor2>("P", "stress"), svk.get<tensor2>("P", "stress")), 1e-13);
    EXPECT_LE(rel(lag->get<tensor4>("P", "tangent"), svk.get<tensor4>("P", "tangent")), 1e-13);
  }
}

TEST(LagrangianStrain, TangentMatchesFiniteDifferences) {
  for (T const m : {0.0, 1.0, -2.0, 2.0})
    for (bool const anisotropic : {false, true}) {
      auto ctx{chain(m, pull_back::lagrangian, anisotropic)};
      auto p{point()};
      p.insert<T>("fd_step", 1e-5); // the default 1e-7 splits an exact pair and loses eps/h: 1e-4 at a rotated pair
      driver d(*ctx, p);
      // general, repeated (the second derivative of tmech#41) and all-equal stretches
      for (auto const &F : {general_F(), stretch(1.15, 1.15, 1.0), tensor2{rotation() * stretch(1.15, 1.15, 1.0)},
                            stretch(1.2, 1.2, 1.2), stretch(1.15, 1.15 + 1e-9, 1.0)}) {
        auto const res{d.step(F, tensor2{}, control::deformation(), 0, 1)};
        ASSERT_TRUE(res.converged);
        EXPECT_LT(d.tangent_error(res), 1e-6) << m << " " << anisotropic << " " << F(0, 0) << " " << F(1, 1) << " " << F(0, 1);
      }
    }
}

TEST(LagrangianStrain, AnisotropicIsObjective) {
  for (T const m : {0.0, 1.0, -2.0}) {
    auto ctx{chain(m, pull_back::lagrangian, true)};
    tensor2 U{stretch(1.3, 0.9, 1.1)};
    U(0, 1) = U(1, 0) = 0.1;
    bind(*ctx, U);
    tensor2 const P_U{ctx->get<tensor2>("P", "stress")};
    bind(*ctx, tensor2{rotation() * U});
    EXPECT_LE(rel(ctx->get<tensor2>("P", "stress"), tensor2{rotation() * P_U}), 1e-12) << m; // P(R U) = R P(U)
    bind(*ctx, rotation());
    EXPECT_LE(tmech::norm(ctx->get<tensor2>("P", "stress")), 1e-10 * K) << m; // a rotation is stress free
  }
}

// why the Lagrangian chain exists: the Eulerian one with the same anisotropic C is not objective
TEST(LagrangianStrain, AnisotropicEulerianChainIsNotObjective) {
  auto ctx{chain(0, pull_back::eulerian, true)};
  tensor2 U{stretch(1.3, 0.9, 1.1)};
  U(0, 1) = U(1, 0) = 0.1;
  bind(*ctx, U);
  tensor2 const P_U{ctx->get<tensor2>("P", "stress")};
  bind(*ctx, tensor2{rotation() * U});
  EXPECT_GT(rel(ctx->get<tensor2>("P", "stress"), tensor2{rotation() * P_U}), 1e-2);
}

// every measure is sym(F - I) to first order
TEST(LagrangianStrain, SmallStrainLimitIsLinearElasticity) {
  tensor2 F{tmech::eye<T, 3, 2>()};
  F(0, 0) = 1 + 1e-6;
  F(0, 1) = 2e-6;
  F(1, 2) = -1e-6;
  tensor1 a;
  a(0) = 1;
  tensor2 const A{tmech::otimes(a, a)};
  tensor4 const C{nm::isotropic_stiffness<T, 3>(K, G) + fibre * tmech::otimes(A, A)};
  tensor2 const sigma{tmech::dcontract(C, tensor2{tmech::sym(tensor2{F - tmech::eye<T, 3, 2>()})})};
  for (T const m : measures) {
    auto ctx{chain(m, pull_back::lagrangian, true)};
    bind(*ctx, F);
    EXPECT_LE(rel(ctx->get<tensor2>("P", "stress"), sigma), 1e-5) << m;
  }
}

} // namespace
