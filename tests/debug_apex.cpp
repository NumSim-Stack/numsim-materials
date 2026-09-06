#include <print>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/drucker_prager_yield_function.h"
#include "numsim-materials/materials/small_strain_plasticity.h"
#include "numsim-materials/solvers/backward_euler.h"

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using dp_yield = numsim::materials::drucker_prager_yield_function<T, 3>;

int main() {
  // Reproduce the oscillation with manual computation
  const T K{166667.0}, G{76923.0}, sigma_0{250.0}, H_mod{1000.0};
  const T eta{0.3}, beta{0.15};
  const T lambda = K - T{2}*G/T{3};  // 115385
  const T G_eff = G + K*eta*beta;   // 84423

  dp_yield yf(eta, beta, K);

  std::println("Elastic constants: lambda={:.1f}, G={:.1f}, K={:.1f}", lambda, G, K);
  std::println("G_eff = {:.1f}", G_eff);
  std::println("");

  // Simulate a few steps manually
  tensor2 eps_p{};   // zero initially
  T alpha_eq = 0;
  const T deps = 0.0005;
  const auto I = tmech::eye<T, 3, 2>();

  for (int step = 0; step < 25; ++step) {
    T eps11 = (step + 1) * deps;

    // Total strain: only eps_11
    tensor2 eps{};
    eps(0, 0) = eps11;

    // Trial stress
    tensor2 sig_trial{};
    sig_trial(0, 0) = (lambda + 2*G) * eps11 - (lambda + 2*G) * eps_p(0, 0)
                     - lambda * eps_p(1, 1) - lambda * eps_p(2, 2);
    sig_trial(1, 1) = lambda * eps11 - lambda * eps_p(0, 0)
                     - (lambda + 2*G) * eps_p(1, 1) - lambda * eps_p(2, 2);
    sig_trial(2, 2) = lambda * eps11 - lambda * eps_p(0, 0)
                     - lambda * eps_p(1, 1) - (lambda + 2*G) * eps_p(2, 2);

    T I1_trial = sig_trial(0, 0) + sig_trial(1, 1) + sig_trial(2, 2);
    T p_trial = I1_trial / T{3};
    tensor2 s_trial = sig_trial - p_trial * I;

    T J2_trial = T{0.5} * tmech::dcontract(s_trial, s_trial);
    T sqrt_j2_trial = std::sqrt(J2_trial);

    T modified_sig_eq = sqrt_j2_trial + eta * p_trial;
    T H_val = H_mod * alpha_eq;
    T F = modified_sig_eq - sigma_0 - H_val;

    if (F <= 0) {
      std::println("step {:2d}: ELASTIC  eps11={:.4f} sig11={:.1f} p={:.1f}",
                   step, eps11, sig_trial(0, 0), p_trial);
      continue;
    }

    // Newton for dlambda
    T dlambda = 0;
    for (int iter = 0; iter < 20; ++iter) {
      T H_iter = H_mod * (alpha_eq + dlambda);
      T r = modified_sig_eq - G_eff * dlambda - sigma_0 - H_iter;
      T dr = -G_eff - H_mod;
      dlambda -= r / dr;
    }

    T sqrt_j2_corrected = sqrt_j2_trial - G * dlambda;
    bool apex = (G * dlambda >= sqrt_j2_trial);

    std::println("step {:2d}: eps11={:.4f}  sqrt_J2_trial={:.2f}  G*dl={:.2f}  "
                 "sqrt_J2_corr={:.2f}  {}",
                 step, eps11, sqrt_j2_trial, G*dlambda, sqrt_j2_corrected,
                 apex ? "*** APEX OVERSHOOT ***" : "ok");

    // Update plastic strain (standard, buggy for apex)
    tensor2 N_trial{};
    if (sqrt_j2_trial > 1e-30) {
      N_trial = s_trial / (T{2} * sqrt_j2_trial) + (beta / T{3}) * I;
    }
    eps_p = eps_p + dlambda * N_trial;
    alpha_eq += dlambda;
  }
}
