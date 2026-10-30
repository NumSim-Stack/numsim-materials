#include <gtest/gtest.h>
#include <memory>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/driver/material_point_driver.h"
#include "numsim-materials/materials/incremental_linear_elasticity.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/umat/json_model.h"
#include "numsim-materials/umat/material_point_evaluator.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

constexpr T K{100}, G{40};

/// The driver on "strain" -> "inc".
nm::material_point_driver<policy> driver_for(ctx_type &ctx) {
  param_type p;
  p.insert<std::string>("strain_source", "strain");
  p.insert<std::string>("stress_source", "inc");
  return {ctx, p};
}

/// strain -> linear_elasticity "total" and incremental_linear_elasticity "inc" reading total's tangent.
std::unique_ptr<ctx_type> build() {
  auto owner{std::make_unique<ctx_type>()};
  auto &ctx{*owner};
  param_type p;
  p.insert<std::string>("name", "strain");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "total");
  p.insert<std::string>("strain_producer_name", "strain");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  ctx.create<nm::linear_elasticity<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "inc");
  p.insert<std::string>("strain_source", "strain");
  p.insert<std::string>("tangent_source", "total");
  ctx.create<nm::incremental_linear_elasticity<policy>>(p);
  ctx.finalize();
  return owner;
}

TEST(IncrementalElasticity, EqualsTheTotalFormAlongAPathAndClosesACycle) {
  auto ctx{build()};
  auto d{driver_for(*ctx)};
  tensor2 eps;
  for (int n{1}; n <= 20; ++n) {
    T const load{n <= 10 ? n * 1e-3 : (20 - n) * 1e-3};
    eps(0, 0) = load;
    eps(0, 1) = eps(1, 0) = 0.5 * load;
    auto const res{d.step(eps, tensor2{}, nm::load_control<T>::strain(), n - 1, 1)};
    ASSERT_TRUE(res.converged);
    d.accept();
    tensor2 const expected{tmech::dcontract(ctx->get<tensor4>("total", "tangent"), eps)};
    EXPECT_NEAR(tmech::norm(tensor2{res.stress - expected}), 0, 1e-12) << n;
    EXPECT_NEAR(tmech::norm(tensor4{res.tangent - ctx->get<tensor4>("total", "tangent")}), 0, 1e-12);
  }
  EXPECT_NEAR(tmech::norm(d.accepted_strain()), 0, 1e-15); // back at zero
  EXPECT_EQ(d.point().history_width(), 6u); // the stress is history (the strain source is the host's)
}

// Under stress control the stress passes exactly through zero at the reversal; the round-off
// of the incremental sum must still count as converged (the driver scales with |C| |d eps|).
TEST(IncrementalElasticity, UniaxialStressCycleThroughZeroConverges) {
  auto ctx{build()};
  auto d{driver_for(*ctx)};
  for (int n{1}; n <= 40; ++n) {
    tensor2 eps;
    eps(0, 0) = (n <= 10 ? n : 20 - n) * 1e-3; // 0 -> 1e-2 -> -2e-2
    auto const res{d.step(eps, tensor2{}, nm::load_control<T>::uniaxial_stress(), n - 1, 1)};
    EXPECT_TRUE(res.converged) << n;
    d.accept();
  }
}

TEST(IncrementalElasticity, StressIsHistoryAndRepeatedTrialsDoNotAccumulate) {
  auto ctx{build()};
  u::material_point_evaluator<policy> ev(*ctx, {.strain_source = "strain", .stress_source = "inc"});
  std::vector<T> statev(ev.nstatv(), 0);
  T const old6[6]{}, new6[6]{1e-3, 0, 0, 0, 0, 0};
  T sig6[6], tan36[36];
  ev.evaluate_canonical(statev.data(), old6, new6, 0, 1, sig6, tan36);
  T const first{sig6[0]};
  ev.evaluate_canonical(statev.data(), old6, new6, 0, 1, sig6, tan36); // same trial again from the same state
  EXPECT_DOUBLE_EQ(sig6[0], first);
  EXPECT_NEAR(first, (K + 4 * G / 3) * 1e-3, 1e-12);
}

// the strain may be any history property: a second node reads the first one's stress as its strain
TEST(IncrementalElasticity, ReadsTheNamedHistoryProperty) {
  param_type p;
  p.insert<std::string>("name", "inc2");
  p.insert<std::string>("strain_source", "inc");
  p.insert<std::string>("strain_property", "stress");
  p.insert<std::string>("tangent_source", "total");
  ctx_type fresh;
  param_type source;
  source.insert<std::string>("name", "strain");
  fresh.create<nm::external_strain_source<policy>>(source);
  for (auto const *m : {"total", "inc"}) {
    param_type q;
    q.insert<std::string>("name", m);
    if (std::string{m} == "total") {
      q.insert<std::string>("strain_producer_name", "strain");
      q.insert<T>("K", K);
      q.insert<T>("G", G);
      fresh.create<nm::linear_elasticity<policy>>(q);
    } else {
      q.insert<std::string>("strain_source", "strain");
      q.insert<std::string>("tangent_source", "total");
      fresh.create<nm::incremental_linear_elasticity<policy>>(q);
    }
  }
  fresh.create<nm::incremental_linear_elasticity<policy>>(p);
  fresh.finalize();
  tensor2 eps;
  eps(0, 1) = eps(1, 0) = 1e-3; // deviatoric: sigma = 2G eps, and inc2 integrates C : d(sigma) = (2G)^2 eps
  dynamic_cast<nm::external_strain_source<policy> *>(fresh.find("strain"))->bind(tensor2{}, eps);
  fresh.update();
  EXPECT_NEAR(tmech::norm(tensor2{fresh.get<tensor2>("inc2", "stress") - 4 * G * G * eps}), 0, 1e-12);
  // a plain property is not a history and cannot be read incrementally
  ctx_type bad;
  bad.create<nm::external_strain_source<policy>>(source);
  param_type q;
  q.insert<std::string>("name", "total");
  q.insert<std::string>("strain_producer_name", "strain");
  q.insert<T>("K", K);
  q.insert<T>("G", G);
  bad.create<nm::linear_elasticity<policy>>(q);
  p.clear();
  p.insert<std::string>("name", "inc");
  p.insert<std::string>("strain_source", "total");
  p.insert<std::string>("strain_property", "stress");
  p.insert<std::string>("tangent_source", "total");
  bad.create<nm::incremental_linear_elasticity<policy>>(p);
  EXPECT_ANY_THROW(bad.finalize());
}

TEST(IncrementalElasticity, FromJson) {
  ctx_type ctx;
  nm::umat::make_json_builder<policy>(R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "linear_elasticity", "name": "total", "strain_producer_name": "strain", "K": 100.0, "G": 40.0},
    {"type": "incremental_linear_elasticity", "name": "inc", "strain_source": "strain", "tangent_source": "total"}]})")(ctx, {});
  EXPECT_NE(ctx.find("inc"), nullptr);
}

} // namespace
