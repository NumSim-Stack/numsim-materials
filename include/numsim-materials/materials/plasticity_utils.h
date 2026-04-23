#ifndef NUMSIM_MATERIALS_PLASTICITY_UTILS_H
#define NUMSIM_MATERIALS_PLASTICITY_UTILS_H

#include <tmech/tmech.h>

namespace numsim::materials::plasticity_detail {

/// Stress state evaluation at a given (ε_p, α) state.
template<typename T, std::size_t Dim>
struct state_eval {
  using tensor2 = tmech::tensor<T, Dim, 2>;
  tensor2 sig;
  tensor2 sig_dev;
  tensor2 N;
  T sig_eq;
  T F;
};

/// Evaluate stress, deviatoric, equivalent stress, flow normal, and yield
/// function at a given state.
template<typename T, std::size_t Dim, typename YieldFunction>
state_eval<T, Dim> evaluate_at_state(
    const YieldFunction& yf,
    const tmech::tensor<T, Dim, 2>& eps,
    const tmech::tensor<T, Dim, 2>& eps_p,
    const tmech::tensor<T, Dim, 4>& C_e,
    T sigma_0, T H_val)
{
  using tensor2 = tmech::tensor<T, Dim, 2>;
  const auto I = tmech::eye<T, Dim, 2>();

  state_eval<T, Dim> se;
  se.sig = tmech::dcontract(C_e, eps - eps_p);

  const auto trace_sig = tmech::trace(se.sig);
  se.sig_dev = se.sig - (trace_sig / T{Dim}) * I;
  se.sig_eq = yf.equivalent_stress(se.sig_dev);
  se.F = yf.trial_yield(se.sig, se.sig_eq, sigma_0, H_val);

  if (se.sig_eq > T{1e-30})
    se.N = yf.flow_normal(se.sig_dev, se.sig_eq);
  else
    se.N = tensor2{};

  return se;
}

/// Trial state — wraps state_eval with a yield check.
template<typename T, std::size_t Dim>
struct trial_state {
  state_eval<T, Dim> eval;
  bool yielding;
};

/// Compute trial stress state and check yield.
template<typename T, std::size_t Dim, typename YieldFunction>
trial_state<T, Dim> compute_trial(
    const YieldFunction& yf,
    const tmech::tensor<T, Dim, 2>& eps,
    const tmech::tensor<T, Dim, 2>& eps_p_old,
    const tmech::tensor<T, Dim, 4>& C_e,
    T sigma_0, T H_val)
{
  trial_state<T, Dim> ts;
  ts.eval = evaluate_at_state<T, Dim>(yf, eps, eps_p_old, C_e, sigma_0, H_val);
  ts.yielding = ts.eval.F > T{0};
  return ts;
}

/// Compute the algorithmic tangent via implicit function theorem.
///
/// For non-associative flow (DP), the yield normal (dF/dsigma) differs
/// from the flow normal (dG/dsigma = N). The residual gradient uses
/// the yield normal: dr/deps = (dF/dsigma) : C_e.
/// The stress correction uses the flow normal: dsigma/ddlambda = -2G*N.
template<typename T, std::size_t Dim, typename YieldFunction>
tmech::tensor<T, Dim, 4> compute_tangent(
    const YieldFunction& yf,
    const tmech::tensor<T, Dim, 2>& sig_dev,
    const tmech::tensor<T, Dim, 2>& N,
    T sig_eq,
    T total_dlambda,
    T G, T dH_val,
    const tmech::tensor<T, Dim, 4>& C_e)
{
  using tensor2 = tmech::tensor<T, Dim, 2>;
  using tensor4 = tmech::tensor<T, Dim, 4>;

  // Yield normal (dF/dsigma) — may differ from flow normal N for non-associative
  const tensor2 M{yf.yield_normal(sig_dev, sig_eq)};

  const auto dr_ddlambda = yf.jacobian(G, dH_val);
  // dr/deps uses YIELD normal M, not flow normal N
  const tensor2 dr_deps{tmech::dcontract(M, C_e)};
  const tensor2 dlambda_deps{-dr_deps / dr_ddlambda};

  // dsigma/deps uses FLOW normal N
  const tensor4 dN_dsig{yf.flow_normal_stress_derivative(N, sig_eq)};
  const tensor4 dN_deps{tmech::dcontract(dN_dsig, C_e)};
  const tensor4 dsig_deps{C_e - T{2} * G * total_dlambda * dN_deps};
  const tensor2 dsig_ddlambda{-T{2} * G * N};

  return dsig_deps + tmech::otimes(dsig_ddlambda, dlambda_deps);
}

} // namespace numsim::materials::plasticity_detail

#endif // NUMSIM_MATERIALS_PLASTICITY_UTILS_H
