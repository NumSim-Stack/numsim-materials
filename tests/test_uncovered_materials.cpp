#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/partial_check.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/materials/exponential_isotropic_hardening.h"
#include "numsim-materials/materials/linear_damage_law.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/scalar_complement_weight.h"
#include "numsim-materials/materials/scalar_stepper.h"
#include "numsim-materials/materials/strain_energy_state_function.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/materials/vector_strain_state_function.h"
#include "numsim-materials/materials/von_mises_state_function.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/json_model.h"

/// Materials that had no test at all. Three of the five are the SECOND of a
/// pair whose first is well covered -- exponential hardening beside linear,
/// the linear damage law beside the exponential one, the complement weight
/// beside the identity weight -- which is the pattern that produced the gap.
namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

/// Publishes one scalar under a caller-chosen property name, so a material can
/// be exercised in isolation instead of behind a full plasticity model. The
/// value is fixed per context; tests that need a sweep build one context per
/// point, which also proves the material is a pure function of its input.
template <typename Traits>
class scalar_source_stub final
    : public nm::material_base<scalar_source_stub<Traits>, Traits> {
public:
  using base = nm::material_base<scalar_source_stub<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit scalar_source_stub(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_value(base::template add_output<value_type>(
            base::m_parameter_handler.template get<std::string>("property"))) {
    m_value = base::template get_parameter<value_type>("value");
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("property")
        .template add<nm::is_required>();
    para.template insert<value_type>("value").template add<nm::is_required>();
    return para;
  }

private:
  value_type& m_value;
};

/// A context holding just the stub and the material under test.
template <typename Material>
std::unique_ptr<ctx_type> with_scalar_source(const std::string& property,
                                             T value, param_type mat) {
  auto ctx = std::make_unique<ctx_type>();
  param_type p;
  p.insert<std::string>("name", "src");
  p.insert<std::string>("property", property);
  p.insert<T>("value", value);
  ctx->template create<scalar_source_stub<policy>>(p);
  ctx->template create<Material>(mat);
  ctx->finalize();
  ctx->update();
  return ctx;
}

/// A scalar_stepper driving `name`, so a scalar consumer has a live source.
void add_driver(ctx_type& ctx, const std::string& name, T increment) {
  param_type p;
  p.insert<std::string>("name", name);
  p.insert<T>("increment", increment);
  ctx.create<nm::scalar_stepper<policy>>(p);
}

// ---------------------------------------------------------------------------

/// H = K_inf (1 - e^{-delta k}),  dH = K_inf delta e^{-delta k}.
/// Saturating, unlike the linear variant: the point of the material is that H
/// approaches K_inf and dH decays, so both ends are asserted.
TEST(ExponentialIsotropicHardening, MatchesTheClosedForm) {
  constexpr T K_inf = 250.0, delta = 8.0;
  auto at = [&](T kappa) {
    param_type m;
    m.insert<std::string>("name", "hardening");
    m.insert<std::string>("source", "src");
    m.insert<T>("K_inf", K_inf);
    m.insert<T>("delta", delta);
    return with_scalar_source<nm::exponential_isotropic_hardening<policy>>(
        "equivalent_plastic_strain", kappa, m);
  };

  for (const T kappa : {0.0, 0.01, 0.05, 0.2, 0.5}) {
    const auto ctx = at(kappa);
    const T e = std::exp(-delta * kappa);
    // hardening_stress is H; hardening_modulus is dH/dkappa.
    EXPECT_NEAR(ctx->get<T>("hardening", "hardening_stress"),
                K_inf * (1.0 - e), 1e-10) << "kappa = " << kappa;
    EXPECT_NEAR(ctx->get<T>("hardening", "hardening_modulus"),
                K_inf * delta * e, 1e-10) << "kappa = " << kappa;
  }

  // Saturation is what distinguishes this from linear_isotropic_hardening:
  // H approaches K_inf rather than growing without bound.
  // Strictly below K_inf while the exponential is still representable...
  const auto mid = at(1.0);
  EXPECT_LT(mid->get<T>("hardening", "hardening_stress"), K_inf);
  EXPECT_GT(mid->get<T>("hardening", "hardening_stress"), 0.99 * K_inf);
  // ...and indistinguishable from it once exp(-delta*kappa) underflows, which
  // is the saturation that distinguishes this from linear_isotropic_hardening.
  const auto far = at(20.0);
  EXPECT_NEAR(far->get<T>("hardening", "hardening_stress"), K_inf, 1e-12);
  EXPECT_NEAR(far->get<T>("hardening", "hardening_modulus"), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------

/// Three branches: below kappa_0 undamaged, above kappa_f capped at 0.999,
/// linear in between. The cap exists so (1-d) never reaches zero; asserting it
/// pins a deliberate choice that reads like a magic number otherwise.
TEST(LinearDamageLaw, HasThreeBranchesIncludingTheCap) {
  constexpr T k0 = 0.10, kf = 0.30;
  auto at = [&](T kappa) {
    param_type m;
    m.insert<std::string>("name", "damage");
    m.insert<std::string>("yield_source", "src");
    m.insert<T>("kappa_0", k0);
    m.insert<T>("kappa_f", kf);
    return with_scalar_source<nm::linear_damage_law<policy>>("kappa", kappa, m);
  };

  // below kappa_0 — undamaged
  for (const T kappa : {0.0, 0.05, 0.10}) {
    const auto c = at(kappa);
    EXPECT_NEAR(c->get<T>("damage", "damage"), 0.0, 1e-12) << kappa;
    EXPECT_NEAR(c->get<T>("damage", "d_damage"), 0.0, 1e-12) << kappa;
  }
  // between — linear ramp with constant slope
  for (const T kappa : {0.15, 0.20, 0.25}) {
    const auto c = at(kappa);
    EXPECT_NEAR(c->get<T>("damage", "damage"), (kappa - k0) / (kf - k0), 1e-10);
    EXPECT_NEAR(c->get<T>("damage", "d_damage"), 1.0 / (kf - k0), 1e-10);
  }
  // at and beyond kappa_f — capped, so (1-d) never reaches zero. Pinning the
  // cap keeps a deliberate choice from reading as a magic number.
  for (const T kappa : {0.30, 0.50, 5.0}) {
    const auto c = at(kappa);
    EXPECT_NEAR(c->get<T>("damage", "damage"), 0.999, 1e-12) << kappa;
    EXPECT_NEAR(c->get<T>("damage", "d_damage"), 0.0, 1e-12) << kappa;
  }
}

/// A degenerate range must not divide by zero.
TEST(LinearDamageLaw, ZeroRangeIsUndamagedRatherThanInfinite) {
  param_type m;
  m.insert<std::string>("name", "damage");
  m.insert<std::string>("yield_source", "src");
  m.insert<T>("kappa_0", 0.2);
  m.insert<T>("kappa_f", 0.2);
  const auto c =
      with_scalar_source<nm::linear_damage_law<policy>>("kappa", 0.5, m);
  EXPECT_TRUE(std::isfinite(c->get<T>("damage", "damage")));
  EXPECT_TRUE(std::isfinite(c->get<T>("damage", "d_damage")));
  EXPECT_NEAR(c->get<T>("damage", "d_damage"), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------

/// value = 1 - source. Trivial, and its partner scalar_identity_weight is
/// tested; the pair is exactly where an untested second variant hides.
TEST(ScalarComplementWeight, IsOneMinusItsSource) {
  constexpr T inc = 0.1;
  ctx_type ctx;
  add_driver(ctx, "frac", inc);
  param_type p;
  p.insert<std::string>("name", "complement");
  p.insert<std::string>("source", "frac::state");
  ctx.create<nm::scalar_complement_weight<policy>>(p);
  ctx.finalize();

  for (int step = 1; step <= 8; ++step) {
    ctx.update();
    EXPECT_NEAR(ctx.get<T>("complement", "value"), 1.0 - inc * step, 1e-12)
        << "step " << step;
    ctx.commit();
  }
}

// ---------------------------------------------------------------------------

/// eps_eq = sqrt(2 W / E) is scalar; the material's job is its DERIVATIVE
/// with respect to strain, which drives damage. Checked against a central
/// difference of eps_eq, which is the property that has to hold.
TEST(StrainEnergyStateFunction, DerivativeMatchesAFiniteDifference) {
  constexpr T K = 166.67, G = 76.92, E_mod = 9 * K * G / (3 * K + G);
  auto build = [&](T e11) {
    auto ctx = std::make_unique<ctx_type>();
    param_type p;
    p.insert<std::string>("name", "stepper");
    p.insert<T>("increment", e11);
    p.insert<std::vector<std::size_t>>("indices", {0, 0});
    ctx->create<nm::tensor_component_stepper<2, policy>>(p);
    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "stepper");
    p.insert<T>("K", K); p.insert<T>("G", G);
    ctx->create<nm::linear_elasticity<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "state");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<std::string>("tangent_source", "elastic");
    p.insert<T>("youngs_modulus", E_mod);
    ctx->create<nm::strain_energy_state_function<policy>>(p);
    ctx->finalize();
    ctx->update();
    return ctx;
  };

  constexpr T e0 = 0.01, h = 1e-6;
  const auto plus = build(e0 + h);
  const auto minus = build(e0 - h);
  const auto at = build(e0);

  const T fd = (plus->get<T>("state", "equivalent_strain") -
                minus->get<T>("state", "equivalent_strain")) / (2 * h);
  const auto& d = at->get<tensor2>("state", "d_equivalent_strain");
  EXPECT_NEAR(d(0, 0), fd, 1e-5)
      << "d(equivalent_strain)/d(eps_11) must match a central difference";
  EXPECT_GT(at->get<T>("state", "equivalent_strain"), 0.0);
}

// ---------------------------------------------------------------------------

/// Publishes a fixed tangent that is minor-symmetric but not major-symmetric (C != C^T), as a hosted law can produce.
template <typename Traits>
class asymmetric_tangent_stub final : public nm::material_base<asymmetric_tangent_stub<Traits>, Traits> {
public:
  using base = nm::material_base<asymmetric_tangent_stub<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit asymmetric_tangent_stub(Args &&...args)
      : base(std::forward<Args>(args)...), m_tangent(base::template add_output<tensor4>("tangent")) {
    tensor2 A, B;
    A(0, 0) = 1, A(1, 1) = -0.5, A(2, 2) = 0.25, A(0, 1) = A(1, 0) = 0.4, A(1, 2) = A(2, 1) = -0.3;
    B(0, 0) = -0.2, B(1, 1) = 0.7, B(2, 2) = 0.1, B(0, 2) = B(2, 0) = 0.5, B(0, 1) = B(1, 0) = 0.2;
    m_tangent = tensor4{2 * 76.92 * nm::sym_identity<T, 3>() + 166.67 * tmech::otimes(tmech::eye<T, 3, 2>(), tmech::eye<T, 3, 2>()) +
                        60.0 * tmech::otimes(A, B)};
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

private:
  tensor4 &m_tangent;
};

tensor2 sym_tensor(T a, T b, T c, T d, T e, T f) {
  tensor2 x;
  x(0, 0) = a, x(1, 1) = b, x(2, 2) = c, x(1, 2) = x(2, 1) = d, x(0, 2) = x(2, 0) = e, x(0, 1) = x(1, 0) = f;
  return x;
}

std::unique_ptr<ctx_type> asymmetric_energy_model(tensor2 const &strain) {
  auto ctx = std::make_unique<ctx_type>();
  param_type p;
  p.insert<std::string>("name", "strain");
  ctx->create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "elastic");
  ctx->create<asymmetric_tangent_stub<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "state");
  p.insert<std::string>("strain_source", "strain");
  p.insert<std::string>("tangent_source", "elastic");
  p.insert<T>("youngs_modulus", 200.0);
  ctx->create<nm::strain_energy_state_function<policy>>(p);
  ctx->finalize();
  dynamic_cast<nm::external_strain_source<policy> *>(ctx->find("strain"))->bind(tensor2{}, strain);
  ctx->update();
  return ctx;
}

/// d(eps : C : eps) = (C + C^T) : eps : d(eps) for a tangent without major symmetry.
TEST(StrainEnergyStateFunction, AsymmetricTangentGivesTheFiniteDifferenceGradient) {
  tensor2 const eps{sym_tensor(0.01, -0.004, 0.002, 0.003, -0.001, 0.005)};
  auto at = asymmetric_energy_model(eps);
  ASSERT_GT(tmech::norm(tensor4{tmech::trans(at->get<tensor4>("elastic", "tangent")) - at->get<tensor4>("elastic", "tangent")}), 1.0);

  auto const& d = at->get<tensor2>("state", "d_equivalent_strain");
  constexpr T h = 1e-6;
  for (auto const& direction : {sym_tensor(1, 0, 0, 0, 0, 0), sym_tensor(0, 1, 0, 0, 0, 0), sym_tensor(0, 0, 0, 0, 0, 1),
                                sym_tensor(0.3, -0.5, 0.8, 0.2, -0.6, 0.4)}) {
    auto const plus = asymmetric_energy_model(tensor2{eps + h * direction});
    auto const minus = asymmetric_energy_model(tensor2{eps - h * direction});
    T const fd = (plus->get<T>("state", "equivalent_strain") - minus->get<T>("state", "equivalent_strain")) / (2 * h);
    EXPECT_NEAR(tmech::dcontract(d, direction), fd, 1e-7 * std::abs(fd) + 1e-9);
  }
  EXPECT_TRUE(nm::check_partials(*at, "state").empty()) << "the declared partials of the equivalent strain wrt strain and tangent";
}

// ---------------------------------------------------------------------------

/// eps_eq = |n . eps . m|, so the derivative is the symmetrised n (x) m, and
/// the absolute value must flip its sign under a sign change of the strain.
TEST(VectorStrainStateFunction, PicksOneComponentAndIsSignSymmetric) {
  auto build = [](T e11) {
    auto ctx = std::make_unique<ctx_type>();
    param_type p;
    p.insert<std::string>("name", "stepper");
    p.insert<T>("increment", e11);
    p.insert<std::vector<std::size_t>>("indices", {0, 0});
    ctx->create<nm::tensor_component_stepper<2, policy>>(p);
    p.clear();
    p.insert<std::string>("name", "state");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<std::vector<std::size_t>>("n_direction", {0});
    p.insert<std::vector<std::size_t>>("m_direction", {0});
    ctx->create<nm::vector_strain_state_function<policy>>(p);
    ctx->finalize();
    ctx->update();
    return ctx;
  };

  const auto pos = build(0.02);
  const auto neg = build(-0.02);

  // n = m = e_0, so the measure is |eps_11| either way.
  EXPECT_NEAR(pos->get<T>("state", "equivalent_strain"), 0.02, 1e-12);
  EXPECT_NEAR(neg->get<T>("state", "equivalent_strain"), 0.02, 1e-12);

  // The derivative carries the sign, so it flips while the measure does not.
  const auto& dp = pos->get<tensor2>("state", "d_equivalent_strain");
  const auto& dn = neg->get<tensor2>("state", "d_equivalent_strain");
  EXPECT_NEAR(dp(0, 0), 1.0, 1e-12);
  EXPECT_NEAR(dn(0, 0), -1.0, 1e-12);
  // and it is confined to the selected component
  EXPECT_NEAR(dp(1, 1), 0.0, 1e-12);
  EXPECT_NEAR(dp(0, 1), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------

/// The equivalent strain accumulated from the declared partial is the hand-written d_equivalent_strain.
TEST(StateFunctions, AccumulatedDerivativeIsTheHandWrittenOne) {
  for (const char *state : {
           R"({"type": "von_mises_state_function", "name": "state", "strain_source": "strain"})",
           R"({"type": "strain_energy_state_function", "name": "state", "strain_source": "strain", "tangent_source": "elastic", "youngs_modulus": 200.0})",
           R"({"type": "vector_strain_state_function", "name": "state", "strain_source": "strain", "m_direction": [0], "n_direction": [1]})"}) {
    ctx_type ctx;
    nm::umat::make_json_builder<policy>(std::string{R"({"materials": [
      {"type": "external_strain_source", "name": "strain"},
      {"type": "linear_elasticity", "name": "elastic", "strain_producer_name": "strain", "K": 166.67, "G": 76.92},
      )"} + state + "]}")(ctx, {});
    auto const &derivative{ctx.derivative<tensor2>("state", "equivalent_strain", "strain", "strain")};
    tensor2 direction;
    direction(0, 0) = 1, direction(1, 1) = -0.3, direction(0, 1) = direction(1, 0) = 0.4, direction(0, 2) = direction(2, 0) = -0.2;
    for (T const load : {0.004, 0.01, -0.006}) {
      dynamic_cast<nm::external_strain_source<policy> *>(ctx.find("strain"))->bind(tensor2{}, tensor2{load * direction});
      ctx.update();
      auto const &expected{ctx.get<tensor2>("state", "d_equivalent_strain")};
      EXPECT_LE(tmech::norm(tensor2{derivative - expected}), 1e-12 * tmech::norm(expected)) << state << " " << load;
      // independent of both: a central difference of the equivalent strain along the load direction
      auto const eps_eq{[&](T value) {
        dynamic_cast<nm::external_strain_source<policy> *>(ctx.find("strain"))->bind(tensor2{}, tensor2{value * direction});
        ctx.update();
        return ctx.get<T>("state", "equivalent_strain");
      }};
      constexpr T h{1e-7};
      T const fd{(eps_eq(load + h) - eps_eq(load - h)) / (2 * h)};
      eps_eq(load);
      EXPECT_NEAR(tmech::dcontract(derivative, direction), fd, 1e-6 * std::abs(fd) + 1e-9) << state << " " << load;
      ctx.commit();
    }
  }
}

}  // namespace
