#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/exponential_damage_law.h"
#include "numsim-materials/materials/linear_damage_law.h"
#include "numsim-materials/materials/peerlings_damage_law.h"
#include "numsim-materials/materials/power_damage_law.h"
#include "numsim-materials/materials/saturation_damage_law.h"
#include "numsim-materials/materials/strain_threshold_yield.h"
#include "numsim-materials/materials/von_mises_state_function.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;

constexpr T kappa_0{0.002};

/// strain -> von Mises state -> threshold yield (kappa) -> the law under test.
template <typename Law>
std::unique_ptr<ctx_type> build(param_type law_parameters) {
  auto owner{std::make_unique<ctx_type>()};
  auto &ctx{*owner};
  param_type p;
  p.insert<std::string>("name", "strain");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "state");
  p.insert<std::string>("strain_source", "strain");
  ctx.create<nm::von_mises_state_function<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "yield");
  p.insert<std::string>("state_source", "state");
  p.insert<T>("kappa_critical", kappa_0);
  ctx.create<nm::strain_threshold_yield<policy>>(p);
  law_parameters.insert<std::string>("name", "law");
  law_parameters.insert<std::string>("yield_source", "yield");
  ctx.create<Law>(law_parameters);
  ctx.finalize();
  return owner;
}

/// Uniaxial strain e11 = x; kappa follows from the graph (kappa_at), whatever the state function's convention.
std::pair<T, T> law_at(ctx_type &ctx, T x) {
  tensor2 eps;
  eps(0, 0) = x;
  dynamic_cast<nm::external_strain_source<policy> *>(ctx.find("strain"))->bind(tensor2{}, eps);
  ctx.update();
  ctx.commit();
  return {ctx.get<T>("law", "damage"), ctx.get<T>("law", "d_damage")};
}

T kappa_at(ctx_type &ctx) { return ctx.get<T>("yield", "kappa"); }

template <typename Law>
void expect_derivative_matches(param_type const &p, T x) {
  auto a{build<Law>(p)}, b{build<Law>(p)}, c{build<Law>(p)};
  T const h{1e-7};
  auto const [d, dd]{law_at(*a, x)};
  T const kappa{kappa_at(*a)};
  auto const [dp, _1]{law_at(*b, x + h)};
  auto const [dm, _2]{law_at(*c, x - h)};
  T const dkappa{kappa_at(*b) - kappa_at(*c)};
  EXPECT_NEAR(dd, (dp - dm) / dkappa, 1e-5 * std::abs(dd) + 1e-12) << "x = " << x << " kappa = " << kappa;
}

TEST(PowerDamageLaw, FormulaDerivativeThresholdAndCap) {
  param_type p;
  p.insert<T>("K", 200.0);
  p.insert<T>("m", 1.5);
  p.insert<T>("kappa_0", kappa_0);
  auto ctx{build<nm::power_damage_law<policy>>(p)};
  EXPECT_EQ(law_at(*ctx, 0.001).first, 0); // below the threshold
  auto const [d, dd]{law_at(*ctx, 0.01)};
  T const kappa{kappa_at(*ctx)};
  EXPECT_NEAR(d, 200.0 * std::pow(kappa - kappa_0, 1.5), 1e-12);
  EXPECT_NEAR(dd, 200.0 * 1.5 * std::pow(kappa - kappa_0, 0.5), 1e-12);
  expect_derivative_matches<nm::power_damage_law<policy>>(p, 0.01);
  auto const [capped, slope]{law_at(*ctx, 0.5)}; // at the cap the damage stops and so does its derivative
  EXPECT_DOUBLE_EQ(capped, 0.999);
  EXPECT_EQ(slope, 0);
}

TEST(SaturationDamageLaw, FormulaAndDerivative) {
  param_type p;
  p.insert<T>("K", 0.9);
  p.insert<T>("m", 300.0);
  p.insert<T>("kappa_0", kappa_0);
  auto ctx{build<nm::saturation_damage_law<policy>>(p)};
  auto const [d, dd]{law_at(*ctx, 0.008)};
  T const kappa{kappa_at(*ctx)};
  EXPECT_NEAR(d, 0.9 * (1 - std::exp(-300.0 * (kappa - kappa_0))), 1e-12);
  EXPECT_NEAR(dd, 0.9 * 300.0 * std::exp(-300.0 * (kappa - kappa_0)), 1e-12);
  expect_derivative_matches<nm::saturation_damage_law<policy>>(p, 0.008);
  EXPECT_LT(law_at(*ctx, 1.0).first, 0.9 + 1e-12); // saturates at K
}

TEST(PeerlingsDamageLaw, FormulaDerivativeAndResidual) {
  param_type p;
  p.insert<T>("kappa_0", kappa_0);
  p.insert<T>("alpha", 0.95);
  p.insert<T>("beta", 200.0);
  auto ctx{build<nm::peerlings_damage_law<policy>>(p)};
  auto const [d, dd]{law_at(*ctx, 0.01)};
  T const kappa{kappa_at(*ctx)};
  T const expected{1 - kappa_0 / kappa * (1 - 0.95 + 0.95 * std::exp(200.0 * (kappa_0 - kappa)))};
  EXPECT_NEAR(d, expected, 1e-12);
  expect_derivative_matches<nm::peerlings_damage_law<policy>>(p, 0.01);
  // for large kappa the damage tends to 1 - (1 - alpha) kappa_0 / kappa -> 1, capped below one
  EXPECT_LT(law_at(*ctx, 10.0).first, 1.0);
  EXPECT_GT(law_at(*ctx, 10.0).first, 0.99);
}

template <typename Law>
void expect_rejected(param_type p, char const *key, T value) {
  p.insert<T>(key, value);
  EXPECT_THROW(build<Law>(p), std::invalid_argument) << key << " = " << value;
}

TEST(DamageLaws, RejectInvalidParameters) {
  param_type power;
  power.insert<T>("K", 200.0);
  power.insert<T>("m", 1.5);
  expect_rejected<nm::power_damage_law<policy>>(power, "K", -1.0);
  expect_rejected<nm::power_damage_law<policy>>(power, "m", 0.0);
  expect_rejected<nm::power_damage_law<policy>>(power, "d_max", 1.5);
  expect_rejected<nm::power_damage_law<policy>>(power, "kappa_0", -0.1);
  param_type saturation;
  saturation.insert<T>("K", 0.9);
  saturation.insert<T>("m", 300.0);
  expect_rejected<nm::saturation_damage_law<policy>>(saturation, "K", 1.2); // K > 1 would give negative stiffness
  expect_rejected<nm::saturation_damage_law<policy>>(saturation, "m", -3.0);
  param_type peerlings;
  peerlings.insert<T>("kappa_0", kappa_0);
  peerlings.insert<T>("alpha", 0.95);
  peerlings.insert<T>("beta", 200.0);
  expect_rejected<nm::peerlings_damage_law<policy>>(peerlings, "alpha", 1.5);
  expect_rejected<nm::peerlings_damage_law<policy>>(peerlings, "beta", -1.0);
  expect_rejected<nm::peerlings_damage_law<policy>>(peerlings, "kappa_0", 0.0);
  param_type exponential;
  exponential.insert<T>("kappa_0", kappa_0);
  exponential.insert<T>("beta", 100.0);
  expect_rejected<nm::exponential_damage_law<policy>>(exponential, "beta", -1.0);
  param_type linear;
  linear.insert<T>("kappa_0", kappa_0);
  linear.insert<T>("kappa_f", 0.01);
  expect_rejected<nm::linear_damage_law<policy>>(linear, "kappa_f", kappa_0 / 2); // kappa_f == kappa_0 stays legal: full damage past the threshold
}

// the older laws share the base: the same threshold, the same cap with a zero slope
TEST(DamageLaws, OlderLawsShareTheCap) {
  param_type linear;
  linear.insert<T>("kappa_0", kappa_0);
  linear.insert<T>("kappa_f", 0.004);
  auto ctx{build<nm::linear_damage_law<policy>>(linear)};
  EXPECT_EQ(law_at(*ctx, 0.001).first, 0);
  auto const [d, dd]{law_at(*ctx, 0.01)};
  EXPECT_DOUBLE_EQ(d, 0.999);
  EXPECT_EQ(dd, 0);
  param_type exponential;
  exponential.insert<T>("kappa_0", kappa_0);
  exponential.insert<T>("beta", 100.0);
  exponential.insert<T>("d_max", 0.5);
  auto ctx2{build<nm::exponential_damage_law<policy>>(exponential)};
  auto const [d2, dd2]{law_at(*ctx2, 0.5)};
  EXPECT_DOUBLE_EQ(d2, 0.5);
  EXPECT_EQ(dd2, 0);
}

/// d damage / d strain accumulated from the partials of state, yield and law against the chain written by hand
/// (is_yielding d_damage d_equivalent_strain), loading, unloading and reloading into the cap.
template <typename Law>
void expect_accumulated_chain(param_type const &p) {
  std::unique_ptr<ctx_type> const ctx{build<Law>(p)};
  auto const &derivative{ctx->derivative<tensor2>("law", "damage", "strain", "strain")};
  int damaging{0}, unloading{0};
  for (T const x : {0.002, 0.004, 0.006, 0.008, 0.005, 0.001, 0.007, 0.01, 0.014, 0.02}) {
    law_at(*ctx, x); // commits, the values stay
    bool const yielding{ctx->get<int>("yield", "is_yielding") != 0};
    damaging += yielding;
    unloading += !yielding && ctx->get<T>("law", "damage") > 0;
    tensor2 const expected{(yielding ? ctx->get<T>("law", "d_damage") : T{0}) * ctx->get<tensor2>("state", "d_equivalent_strain")};
    EXPECT_LE(tmech::norm(tensor2{derivative - expected}), 1e-12 * tmech::norm(expected)) << Law::type_name() << " x = " << x;
  }
  EXPECT_GT(damaging, 3) << Law::type_name();
  EXPECT_GT(unloading, 1) << Law::type_name();
}

TEST(DamageLaws, AccumulatedDerivativeIsTheHandWrittenChain) {
  param_type power, saturation, peerlings, exponential, linear;
  for (auto *p : {&power, &saturation, &peerlings, &exponential, &linear})
    p->insert<T>("kappa_0", kappa_0);
  power.insert<T>("K", 200.0);
  power.insert<T>("m", 1.5);
  saturation.insert<T>("K", 0.9);
  saturation.insert<T>("m", 300.0);
  peerlings.insert<T>("alpha", 0.95);
  peerlings.insert<T>("beta", 200.0);
  exponential.insert<T>("beta", 100.0);
  linear.insert<T>("kappa_f", 0.008); // capped at the end of the path
  expect_accumulated_chain<nm::power_damage_law<policy>>(power);
  expect_accumulated_chain<nm::saturation_damage_law<policy>>(saturation);
  expect_accumulated_chain<nm::peerlings_damage_law<policy>>(peerlings);
  expect_accumulated_chain<nm::exponential_damage_law<policy>>(exponential);
  expect_accumulated_chain<nm::linear_damage_law<policy>>(linear);
}

TEST(DamageLaws, FromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "von_mises_state_function", "name": "state", "strain_source": "strain"},
    {"type": "strain_threshold_yield", "name": "yield", "state_source": "state", "kappa_critical": 0.002},
    {"type": "power_damage_law", "name": "pow", "yield_source": "yield", "K": 200.0, "m": 1.5, "kappa_0": 0.002},
    {"type": "saturation_damage_law", "name": "sat", "yield_source": "yield", "K": 0.9, "m": 300.0, "kappa_0": 0.002},
    {"type": "peerlings_damage_law", "name": "peer", "yield_source": "yield", "kappa_0": 0.002, "alpha": 0.95, "beta": 200.0}]})")(ctx, {});
  EXPECT_NE(ctx.find("peer"), nullptr);
}

} // namespace
