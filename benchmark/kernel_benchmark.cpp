// The chain-rule kernels of a derivative plan in 3D: out (+)= P * in for d out / d in of ranks (a, b) applied to
// d in / d seed of rank s, the seed copy and the scalar fold. Args: a, b, s, assign.
#include "numsim-materials/core/derivative_kernels.h"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <string>

namespace {

namespace dv = numsim::materials::derivatives;

std::size_t arg(benchmark::State const &state, int i) { return static_cast<std::size_t>(state.range(i)); }

void step_kernel(benchmark::State &state) {
  auto const a{arg(state, 0)}, b{arg(state, 1)}, s{arg(state, 2)};
  bool const assign{state.range(3) != 0};
  auto P{dv::make_value<double, 3>(a + b)}, in{dv::make_value<double, 3>(b + s)}, out{dv::make_value<double, 3>(a + s)},
      scratch{dv::make_value<double, 3>(a + s)};
  P->fill(0.5);
  in->fill(0.25);
  auto const kernel{dv::step_kernel_for<double, 3>(a, b, s, assign)};
  void *const work{dv::needs_scratch(a, b, s, assign) ? scratch->object() : nullptr};
  for (auto _ : state) {
    kernel(P->object(), in->object(), out->object(), work);
    benchmark::DoNotOptimize(out->data());
    benchmark::ClobberMemory();
  }
  state.SetLabel(std::to_string(a) + "." + std::to_string(b) + "." + std::to_string(s) + (assign ? " =" : " +="));
}

void seed_kernel(benchmark::State &state) {
  auto const a{arg(state, 0)}, s{arg(state, 1)};
  auto P{dv::make_value<double, 3>(a + s)}, out{dv::make_value<double, 3>(a + s)};
  P->fill(0.5);
  auto const kernel{dv::seed_kernel_for<double, 3>(a, s, true)};
  for (auto _ : state) {
    kernel(P->object(), nullptr, out->object(), nullptr);
    benchmark::DoNotOptimize(out->data());
    benchmark::ClobberMemory();
  }
}

/// dx/dp = -(1 / J) dR/dp of a scalar system, dR/dp of rank `rank`.
void scalar_fold(benchmark::State &state) {
  auto const rank{arg(state, 0)};
  auto B{dv::make_value<double, 3>(rank)}, dx{dv::make_value<double, 3>(rank)};
  B->fill(0.5);
  auto const scale{dv::scale_kernel_for<double, 3>(rank)};
  double J{2.0};
  for (auto _ : state) {
    benchmark::DoNotOptimize(J);
    scale(-1 / J, B->object(), dx->object());
    benchmark::DoNotOptimize(dx->data());
    benchmark::ClobberMemory();
  }
}

} // namespace

// The J2 plans' products (fold 2.0.2 +=, C : d eps 2.2.2, d sigma / d T 2.2.0), then larger ranks.
BENCHMARK(step_kernel)
    ->Args({2, 0, 2, 0})
    ->Args({2, 0, 2, 1})
    ->Args({2, 2, 2, 1})
    ->Args({2, 2, 2, 0})
    ->Args({2, 2, 0, 1})
    ->Args({0, 2, 2, 1})
    ->Args({4, 2, 2, 0})
    ->Args({2, 4, 2, 0})
    ->Args({4, 2, 4, 0})
    ->Args({4, 4, 4, 0});
BENCHMARK(seed_kernel)->Args({2, 2})->Args({4, 2});
BENCHMARK(scalar_fold)->Arg(0)->Arg(2)->Arg(4);
