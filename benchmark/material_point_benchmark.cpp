// One material point: a context set up once, then bind the strain and update() per iteration (no commit, so
// every iteration repeats the same increment). Arg 0: elastic, 1: plastic.
#include "j2_residual_system.h"
#include "numsim-materials/materials/j2_plasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/solvers/local_newton.h"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <string>

namespace {

namespace nm = numsim::materials;
namespace bm = numsim::materials::benchmark;
using policy = bm::system_policy;
using context = bm::system_context;
using t2 = tmech::tensor<double, 3, 2>;
using t4 = tmech::tensor<double, 3, 4>;

constexpr double K{166.67}, G{76.92}, sigma_0{50.0}, H{1000.0}, alpha{1e-5}, T_ref{20}, T_now{320};

enum class model { shipped, return_node, thermal, thermal_dT, split };

/// j2_plasticity with its local Newton: the closed-form tangent is an output.
void shipped_build(context &ctx) {
  policy::ParameterHandler p;
  p.insert<std::string>("name", "strain");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<double>("tolerance", 1e-8);
  p.insert<int>("max_iter", 100);
  ctx.create<nm::local_newton<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "j2");
  p.insert<double>("K", H);
  ctx.create<nm::linear_isotropic_hardening<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "j2");
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "strain");
  p.insert<std::string>("strain_property", "strain");
  p.insert<std::string>("solver_source", "solver");
  p.insert<double>("K", K);
  p.insert<double>("G", G);
  p.insert<double>("sigma_0", sigma_0);
  ctx.create<nm::j2_plasticity<policy>>(p);
}

/// Shear-dominated; 0.001 stays elastic, 0.4 yields.
t2 strain_of(std::int64_t plastic) {
  double const g{plastic ? 0.4 : 0.001};
  t2 e;
  e(0, 1) = e(1, 0) = g;
  e(0, 0) = 0.3 * g;
  e(2, 2) = -0.1 * g;
  return e;
}

void material_point(benchmark::State &state, model m) {
  context ctx;
  switch (m) {
  case model::shipped: shipped_build(ctx); break;
  case model::return_node: bm::j2_return_build(K, G, sigma_0, H)(ctx); break;
  case model::thermal:
  case model::thermal_dT: bm::j2_return_build(K, G, sigma_0, H, alpha, T_ref)(ctx); break;
  case model::split: bm::j2_system_build(K, G, sigma_0, H)(ctx); break;
  }
  ctx.finalize();
  if (m != model::shipped)
    (void)ctx.derivative<t4>("j2", "stress", "strain", "strain");
  if (m == model::thermal_dT)
    (void)ctx.derivative<t2>("j2", "stress", "temperature", "state");
  if (m == model::thermal || m == model::thermal_dT)
    dynamic_cast<nm::external_scalar_source<policy> &>(*ctx.find("temperature")).bind(T_ref, T_now);
  auto &strain{dynamic_cast<nm::external_strain_source<policy> &>(*ctx.find("strain"))};
  t2 const eps{strain_of(state.range(0))}, zero{};
  for (auto _ : state) {
    strain.bind(zero, eps);
    ctx.update();
    benchmark::ClobberMemory();
  }
  state.SetLabel(state.range(0) ? "plastic" : "elastic");
}

} // namespace

BENCHMARK_CAPTURE(material_point, shipped_j2, model::shipped)->Arg(0)->Arg(1);
BENCHMARK_CAPTURE(material_point, return_node, model::return_node)->Arg(0)->Arg(1);
BENCHMARK_CAPTURE(material_point, return_node_thermal, model::thermal)->Arg(0)->Arg(1);
BENCHMARK_CAPTURE(material_point, return_node_thermal_dT, model::thermal_dT)->Arg(0)->Arg(1);
BENCHMARK_CAPTURE(material_point, separate_stress_node, model::split)->Arg(0)->Arg(1);
