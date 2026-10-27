#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/curing_rate.h"
#include "numsim-materials/materials/kamal_sourour_rate.h"
#include "numsim-materials/materials/scalar_stepper.h"
#include "numsim-materials/solvers/rk_integrator.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

constexpr T R{8.31446261815324};
constexpr T A1{2e5}, E1{60000}, A2{1e6}, E2{50000}, m{0.7}, n{1.3};

/// temperature (scalar_stepper) and a state source feeding the rate law `type`.
std::unique_ptr<ctx_type> build(std::string const &type, T A2_ = A2, T dz = 0, T m_ = m) {
  auto owner{std::make_unique<ctx_type>()};
  auto &ctx{*owner};
  param_type p;
  p.insert<std::string>("name", "temperature");
  p.insert<T>("increment", T{0});
  ctx.create<nm::scalar_stepper<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "state_source");
  p.insert<T>("increment", dz);
  ctx.create<nm::scalar_stepper<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "rate");
  p.insert<std::string>("integrator_source", "state_source");
  if (type == "kamal") {
    p.insert<T>("A1", A1);
    p.insert<T>("E1", E1);
    p.insert<T>("A2", A2_);
    p.insert<T>("E2", E2);
    p.insert<T>("m", m_);
    p.insert<T>("n", n);
    ctx.create<nm::kamal_sourour_rate<policy>>(p);
  } else {
    p.insert<T>("A", A1);
    p.insert<T>("E", E1);
    p.insert<T>("m", T{0});
    p.insert<T>("n", n);
    ctx.create<nm::curing_rate<policy>>(p);
  }
  ctx.finalize();
  return owner;
}

T rate_at(ctx_type &ctx, T z, T theta) {
  ctx.get_mutable<T>("temperature", "state") = theta;
  ctx.get_mutable<T>("state_source", "state") = z;
  ctx.update();
  return ctx.get<T>("rate", "rate");
}

TEST(KamalSourour, RateAndDerivativeMatchTheFormula) {
  auto ctx{build("kamal")};
  T const theta{80}, z{0.3};
  T const k1{A1 * std::exp(-E1 / (R * (273.15 + theta)))}, k2{A2 * std::exp(-E2 / (R * (273.15 + theta)))};
  EXPECT_NEAR(rate_at(*ctx, z, theta), (k1 + k2 * std::pow(z, m)) * std::pow(1 - z, n), 1e-12 * k2);
  T const h{1e-6};
  T const fd{(rate_at(*ctx, z + h, theta) - rate_at(*ctx, z - h, theta)) / (2 * h)};
  rate_at(*ctx, z, theta);
  EXPECT_NEAR(ctx->get<T>("rate", "rate_derivative"), fd, 1e-6 * std::abs(fd));
}

// curing_rate returns zero at z = 0 exactly (its lower clamp), so the identity holds for z > 0.
TEST(KamalSourour, WithoutTheAutocatalyticTermItIsTheNthOrderLaw) {
  auto kamal{build("kamal", 0.0)}, nth{build("nth")};
  for (T const z : {1e-6, 0.2, 0.9})
    EXPECT_NEAR(rate_at(*kamal, z, 60), rate_at(*nth, z, 60), 1e-15);
}

// z^0 = 1: with m = 0 the autocatalytic term is a constant and the reaction starts at k1 + k2
TEST(KamalSourour, ZeroExponentKeepsTheAutocatalyticTermAtZeroCure) {
  auto ctx{build("kamal", A2, 0, 0.0)};
  T const theta{80};
  T const k1{A1 * std::exp(-E1 / (R * (273.15 + theta)))}, k2{A2 * std::exp(-E2 / (R * (273.15 + theta)))};
  EXPECT_NEAR(rate_at(*ctx, 0.0, theta), k1 + k2, 1e-12 * (k1 + k2));
  EXPECT_NEAR(ctx->get<T>("rate", "rate_derivative"), -n * (k1 + k2), 1e-12 * (k1 + k2)); // d/dz (1 - z)^n at 0
}

// m = 1: d/dz [(k1 + k2 z)(1 - z)^n] at z = 0 is k2 - n k1 (the z^(m-1) factor is one, not zero)
TEST(KamalSourour, DerivativeAtZeroCureForUnitExponent) {
  auto ctx{build("kamal", A2, 0, 1.0)};
  T const theta{80};
  T const k1{A1 * std::exp(-E1 / (R * (273.15 + theta)))}, k2{A2 * std::exp(-E2 / (R * (273.15 + theta)))};
  rate_at(*ctx, 0.0, theta);
  EXPECT_NEAR(ctx->get<T>("rate", "rate_derivative"), k2 - n * k1, 1e-12 * k2);
}

TEST(KamalSourour, RejectsNegativeParameters) {
  for (auto const *key : {"A1", "E1", "A2", "E2", "m", "n"}) {
    ctx_type ctx;
    param_type p;
    p.insert<std::string>("name", "temperature");
    p.insert<T>("increment", T{0});
    ctx.create<nm::scalar_stepper<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "rate");
    p.insert<std::string>("integrator_source", "temperature");
    for (auto const &[k, v] : {std::pair{"A1", A1}, std::pair{"E1", E1}, std::pair{"A2", A2}, std::pair{"E2", E2}, std::pair{"m", m}, std::pair{"n", n}})
      p.insert<T>(k, std::string{k} == key ? T{-1} : v);
    EXPECT_THROW(ctx.create<nm::kamal_sourour_rate<policy>>(p), std::invalid_argument) << key;
  }
}

TEST(KamalSourour, RateVanishesAtFullCure) {
  auto ctx{build("kamal")};
  EXPECT_NEAR(rate_at(*ctx, 1.0, 100), 0, 1e-15);
  EXPECT_GT(rate_at(*ctx, 0.0, 100), 0); // the nth-order term starts the reaction
}

TEST(KamalSourour, IntegratesToFullCureWithRungeKutta) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "temperature");
  p.insert<T>("increment", T{0});
  ctx.create<nm::scalar_stepper<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "integrator");
  p.insert<std::string>("function", "rate");
  p.insert<T>("step_size", T{1});
  p.insert<std::string>("tableau", "rk4");
  ctx.create<nm::rk_integrator<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "rate");
  p.insert<std::string>("integrator_source", "integrator");
  p.insert<T>("A1", A1);
  p.insert<T>("E1", E1);
  p.insert<T>("A2", A2);
  p.insert<T>("E2", E2);
  p.insert<T>("m", m);
  p.insert<T>("n", n);
  ctx.create<nm::kamal_sourour_rate<policy>>(p);
  ctx.finalize();
  ctx.get_mutable<T>("temperature", "state") = 120;
  T previous{0};
  for (int i{0}; i < 600; ++i) {
    ctx.update();
    T const z{ctx.get<T>("integrator", "state")};
    EXPECT_GE(z, previous - 1e-14);
    previous = z;
    ctx.commit();
  }
  EXPECT_GT(previous, 0.99);
  EXPECT_LE(previous, 1.0 + 1e-12);
}

TEST(KamalSourour, FromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "scalar_stepper", "name": "temperature", "increment": 0.0},
    {"type": "scalar_stepper", "name": "z", "increment": 0.0},
    {"type": "kamal_sourour_rate", "name": "rate", "integrator_source": "z", "A1": 2e5, "E1": 60000.0, "A2": 1e6, "E2": 50000.0, "m": 0.7, "n": 1.3}]})")(ctx, {});
  EXPECT_NE(ctx.find("rate"), nullptr);
}

} // namespace
