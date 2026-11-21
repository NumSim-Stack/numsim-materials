#include <fstream>
#include <numsim-core/print.h>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/drucker_prager_yield_function.h"
#include "numsim-materials/materials/drucker_prager_plasticity.h"
#include "numsim-materials/materials/j2_plasticity.h"
#include "numsim-materials/solvers/local_newton.h"
#include "numsim-materials/driver/external_state_source.h"

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using dp_yield = numsim::materials::drucker_prager_yield_function<T, 3>;
using dp_plasticity = numsim::materials::drucker_prager_plasticity<policy>;
using j2_plasticity = numsim::materials::j2_plasticity<policy>;

// Material constants (steel-like)
constexpr T K_val{166667.0};   // MPa
constexpr T G_val{76923.0};    // MPa
constexpr T sigma_0{250.0};    // MPa
constexpr T H_mod{1000.0};     // MPa

struct run_result {
  std::vector<int> step;
  std::vector<T> eps_load;     // driving strain component
  std::vector<T> sig_11, sig_22, sig_33, sig_12;
  std::vector<T> pressure;
  std::vector<T> alpha;
  std::vector<T> tangent_rel_error;
};

void record(run_result& r, int i, const tensor2& eps, const tensor2& sig,
            T al, T rel, std::size_t ci, std::size_t cj) {
  r.step.push_back(i);
  r.eps_load.push_back(eps(ci, cj));
  r.sig_11.push_back(sig(0, 0));
  r.sig_22.push_back(sig(1, 1));
  r.sig_33.push_back(sig(2, 2));
  r.sig_12.push_back(sig(0, 1));
  r.pressure.push_back(tmech::trace(sig) / T{3});
  r.alpha.push_back(al);
  r.tangent_rel_error.push_back(rel);
}

/// The stepper's path, bound by the host: component (ci, cj) grows by increment, then one evaluation.
void step(ctx_type& ctx, numsim::materials::external_strain_source<policy>& strain, T increment, std::size_t ci,
          std::size_t cj) {
  tensor2 e{strain.strain().old_value()};
  e(ci, cj) += increment;
  strain.bind(strain.strain().old_value(), e);
  ctx.update();
}

/// Relative error of `material`'s tangent, differentiated along the host path.
T tangent_error(ctx_type& ctx, std::string const& material) {
  param_type p;
  p.insert<std::string>("output", material + "::stress");
  p.insert<std::string>("input", "stepper::strain");
  p.insert<std::string>("analytical", material + "::tangent");
  return ctx.check_derivative(std::move(p)).rel_error;
}

run_result run_j2(T increment, int steps,
                  std::size_t ci, std::size_t cj) {
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "stepper");
  auto& strain{ctx.create<numsim::materials::external_strain_source<policy>>(p)};

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", K_val);
  p.insert<T>("G", G_val);
  ctx.create<numsim::materials::linear_elasticity<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "solver");
  ctx.create<numsim::materials::local_newton<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "j2");
  p.insert<T>("K", H_mod);
  ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "j2");
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<std::string>("solver_source", "solver");
  p.insert<T>("K", K_val);
  p.insert<T>("G", G_val);
  p.insert<T>("sigma_0", sigma_0);
  ctx.create<j2_plasticity>(p);


  ctx.finalize();

  run_result r;
  for (int i = 0; i < steps; ++i) {
    step(ctx, strain, increment, ci, cj);
    record(r, i,
           strain.strain().new_value(),
           ctx.get<tensor2>("j2", "stress"),
           ctx.get<T>("j2", "equivalent_plastic_strain"),
           tangent_error(ctx, "j2"),
           ci, cj);
    ctx.commit();
  }
  return r;
}

run_result run_dp(T increment, int steps,
                  std::size_t ci, std::size_t cj) {
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "stepper");
  auto& strain{ctx.create<numsim::materials::external_strain_source<policy>>(p)};

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", K_val);
  p.insert<T>("G", G_val);
  ctx.create<numsim::materials::linear_elasticity<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "solver");
  ctx.create<numsim::materials::local_newton<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "dp");
  p.insert<T>("K", H_mod);
  ctx.create<numsim::materials::linear_isotropic_hardening<policy>>(p);


  p.clear();
  p.insert<std::string>("name", "dp");
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<std::string>("solver_source", "solver");
  p.insert<T>("G", G_val);
  p.insert<T>("sigma_0", sigma_0);
  p.insert<T>("eta", T{0.3});
  p.insert<T>("beta", T{0.15});
  p.insert<T>("K_bulk", K_val);
  ctx.create<dp_plasticity>(p);


  ctx.finalize();

  run_result r;
  for (int i = 0; i < steps; ++i) {
    step(ctx, strain, increment, ci, cj);
    record(r, i,
           strain.strain().new_value(),
           ctx.get<tensor2>("dp", "stress"),
           ctx.get<T>("dp", "equivalent_plastic_strain"),
           tangent_error(ctx, "dp"),
           ci, cj);
    ctx.commit();
  }
  return r;
}

void write_csv(const std::string& path, const run_result& r) {
  std::ofstream f(path);
  f << "step,eps_load,sig_11,sig_22,sig_33,sig_12,pressure,alpha,tangent_rel_error\n";
  for (std::size_t i = 0; i < r.step.size(); ++i) {
    numsim_core::print(f, "{},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e}\n",
               r.step[i], r.eps_load[i],
               r.sig_11[i], r.sig_22[i], r.sig_33[i], r.sig_12[i],
               r.pressure[i], r.alpha[i], r.tangent_rel_error[i]);
  }
  numsim_core::println("Wrote {}", path);
}

int main() {
  struct load_case {
    std::size_t i, j;
    T increment;
    int steps;
    std::string tag;
  };

  std::vector<load_case> cases = {
    {0, 0, T{0.0005}, 60, "eps_11"},  // uniaxial strain 11
    {1, 1, T{0.0005}, 60, "eps_22"},  // uniaxial strain 22
    {0, 1, T{0.001},  60, "eps_12"},  // pure shear 12
  };

  for (const auto& lc : cases) {
    numsim_core::println("=== Load case: {} (increment={}) ===", lc.tag, lc.increment);

    auto j2 = run_j2(lc.increment, lc.steps, lc.i, lc.j);
    write_csv("j2_" + lc.tag + ".csv", j2);

    auto dp = run_dp(lc.increment, lc.steps, lc.i, lc.j);
    write_csv("dp_" + lc.tag + ".csv", dp);
  }

  return 0;
}
