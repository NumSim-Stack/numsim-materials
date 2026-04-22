#ifndef NUMSIM_MATERIALS_PLASTICITY_UTILS_H
#define NUMSIM_MATERIALS_PLASTICITY_UTILS_H

#include <tmech/tmech.h>

namespace numsim::materials::plasticity_detail {

/// Trial state computed from current strain and plastic strain.
template<typename T, std::size_t Dim>
struct trial_state {
  using tensor2 = tmech::tensor<T, Dim, 2>;
  tensor2 sig_trial;
  tensor2 sig_dev;
  tensor2 N;
  T sig_eq;
  bool yielding;
};

/// Compute the trial stress state and check yield.
template<typename T, std::size_t Dim, typename YieldFunction>
trial_state<T, Dim> compute_trial(
    const tmech::tensor<T, Dim, 2>& eps,
    const tmech::tensor<T, Dim, 2>& eps_p_old,
    const tmech::tensor<T, Dim, 4>& C_e,
    T sigma_0, T H_val)
{
  using tensor2 = tmech::tensor<T, Dim, 2>;
  const auto I = tmech::eye<T, Dim, 2>();

  trial_state<T, Dim> ts;
  ts.sig_trial = tmech::dcontract(C_e, eps - eps_p_old);

  const auto trace_sig = tmech::trace(ts.sig_trial);
  ts.sig_dev = ts.sig_trial - (trace_sig / T{Dim}) * I;
  ts.sig_eq = YieldFunction::equivalent_stress(ts.sig_dev);

  const auto F = YieldFunction::trial_yield(ts.sig_eq, sigma_0, H_val);
  ts.yielding = F > T{0};

  if (ts.sig_eq > T{1e-30})
    ts.N = YieldFunction::flow_normal(ts.sig_dev, ts.sig_eq);
  else
    ts.N = tensor2{};

  return ts;
}

/// Compute the algorithmic tangent via implicit function theorem.
/// Uses TRIAL sig_eq and N — correct for radial return (J2).
template<typename T, std::size_t Dim, typename YieldFunction>
tmech::tensor<T, Dim, 4> compute_tangent(
    const tmech::tensor<T, Dim, 2>& N_trial,
    T sig_eq_trial,
    T total_dlambda,
    T G, T dH_val,
    const tmech::tensor<T, Dim, 4>& C_e)
{
  using tensor2 = tmech::tensor<T, Dim, 2>;
  using tensor4 = tmech::tensor<T, Dim, 4>;

  const auto dr_ddlambda = YieldFunction::jacobian(G, dH_val);
  const tensor2 dr_deps{tmech::dcontract(N_trial, C_e)};
  const tensor2 dlambda_deps{-dr_deps / dr_ddlambda};

  const tensor4 dN_dsig{YieldFunction::flow_normal_stress_derivative(N_trial, sig_eq_trial)};
  const tensor4 dN_deps{tmech::dcontract(dN_dsig, C_e)};
  const tensor4 dsig_deps{C_e - T{2} * G * total_dlambda * dN_deps};
  const tensor2 dsig_ddlambda{-T{2} * G * N_trial};

  return dsig_deps + tmech::otimes(dsig_ddlambda, dlambda_deps);
}

} // namespace numsim::materials::plasticity_detail

#endif // NUMSIM_MATERIALS_PLASTICITY_UTILS_H
