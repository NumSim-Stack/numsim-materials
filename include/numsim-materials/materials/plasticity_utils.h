#ifndef NUMSIM_MATERIALS_PLASTICITY_UTILS_H
#define NUMSIM_MATERIALS_PLASTICITY_UTILS_H

#include <cassert>
#include <limits>
#include <tmech/tmech.h>

namespace numsim::materials::plasticity_detail {

/// Construct the fourth-order deviatoric projector IIdev = IIsym - 1/Dim * I⊗I.
/// Shared by J2 and DP flow normal derivatives.
template<typename T, std::size_t Dim>
tmech::tensor<T, Dim, 4> make_IIdev() {
  const auto I = tmech::eye<T, Dim, 2>();
  const auto IIsym = (tmech::otimesu(I, I) + tmech::otimesl(I, I)) * T{0.5};
  const auto IIvol = tmech::otimes(I, I) / T{Dim};
  return tmech::tensor<T, Dim, 4>{IIsym - IIvol};
}

/// Isotropic elastic tangent C = K*I⊗I + 2G*IIdev.
///
/// The bulk term is K*I⊗I, NOT 3K*IIvol. Those are equal only at Dim = 3,
/// because IIvol = I⊗I/Dim. All three plasticity materials wrote the 3K*IIvol
/// form while their fast paths -- do_smooth_return's 2G*dev(N) + K*tr(N)*I and
/// drucker_prager_yield_function::apex_tangent's K*H'/(...)*I⊗I -- assumed the
/// K*I⊗I one, so at Dim = 2 a material disagreed with itself by 30% on
/// C:x against its own shortcut. Building it here keeps the two consistent by
/// construction, at every dimension.
///
/// linear_elasticity keeps its own 3K*IIvol spelling: it is a separate working
/// material with no such shortcut, and plasticity no longer sources a tangent
/// from it.
template<typename T, std::size_t Dim>
tmech::tensor<T, Dim, 4> make_isotropic_tangent(T K, T G) {
  const auto I = tmech::eye<T, Dim, 2>();
  return tmech::tensor<T, Dim, 4>{K * tmech::otimes(I, I) +
                                  T{2} * G * make_IIdev<T, Dim>()};
}

/// Stress state evaluation at a given (ε_p, κ) state.
template<typename T, std::size_t Dim>
struct state_eval {
  using tensor2 = tmech::tensor<T, Dim, 2>;
  tensor2 sig;
  tensor2 sig_dev;
  tensor2 N;
  T sig_eq;
  T modified_sig_eq;  // includes pressure term for DP
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

  // Guard threshold: below this equivalent stress, flow normal is undefined
  // and downstream derivatives (1/J2 ~ 1/sig_eq^2) would overflow into
  // subnormals. Scaled to the yield stress so it stays meaningful at any
  // stress magnitude.
  const auto sig_eq_min = T{1e-10} * std::abs(sigma_0);

  state_eval<T, Dim> se;
  se.sig = tmech::dcontract(C_e, eps - eps_p);

  const auto trace_sig = tmech::trace(se.sig);
  se.sig_dev = se.sig - (trace_sig / T{Dim}) * I;
  se.sig_eq = yf.equivalent_stress(se.sig_dev);
  se.modified_sig_eq = yf.modified_equivalent_stress(se.sig, se.sig_eq);
  se.F = yf.trial_yield(se.sig, se.sig_eq, sigma_0, H_val);

  if (se.sig_eq > sig_eq_min)
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
/// σ = C : (ε - ε_p(Δλ))  where  ε_p = ε_p_old + Δλ · N_trial(ε)
/// r(Δλ, ε) = σ̃_trial(ε) - G_eff·Δλ - Y₀ - H(κ_n + Δλ) = 0
///
/// A = ∂σ/∂ε|_{Δλ} = C - Δλ · C : (dN/dσ) : C
///
/// dΔλ/dε = -(∂r/∂Δλ)⁻¹ · (∂r/∂ε)
///   ∂r/∂ε = M : C    (trial quantities depend on ε only through σ_trial = C:ε)
///   ∂r/∂Δλ = -(G_eff + H') = -(M:C:N + H')
///
/// C_ep = A + (∂σ/∂Δλ) ⊗ (dΔλ/dε)
///   ∂σ/∂Δλ = -C : N
///
/// All quantities are evaluated at the trial state.
/// Precondition: sig_eq > 0 (smooth return, not apex).
template<typename T, std::size_t Dim, typename YieldFunction>
tmech::tensor<T, Dim, 4> compute_tangent(
    const YieldFunction& yf,
    const tmech::tensor<T, Dim, 2>& sig_dev,
    const tmech::tensor<T, Dim, 2>& N,
    T sig_eq,
    T total_dlambda,
    T dH_val,
    const tmech::tensor<T, Dim, 4>& C_e,
    T G)
{
  using tensor2 = tmech::tensor<T, Dim, 2>;
  using tensor4 = tmech::tensor<T, Dim, 4>;

  assert(sig_eq > T{0} && "compute_tangent requires sig_eq > 0 (use apex tangent at q=0)");

  // Yield normal M = dF/dsigma (differs from N for non-associative)
  const tensor2 M{yf.yield_normal(sig_dev, sig_eq)};

  // C : N
  const tensor2 C_N{tmech::dcontract(C_e, N)};

  // dr/ddlambda = M : (-C:N) - dH = -(M:C:N + dH)
  const auto dr_ddlambda = -(tmech::dcontract(M, C_N) + dH_val);

  // dr/deps = M : C (trial quantities don't depend on dlambda)
  const tensor2 dr_deps{tmech::dcontract(M, C_e)};

  // dlambda/deps
  const tensor2 dlambda_deps{-dr_deps / dr_ddlambda};

  // dsigma/ddlambda = -C : N
  const tensor2 dsig_ddlambda{-C_N};

  // A = dsigma/deps|_{dlambda} = C - dlambda * C : (dN/dsig) : C
  // flow_normal_stress_derivative takes (sig_dev, sig_eq) to avoid
  // cancellation error from reconstructing s from N.
  const tensor4 dN_dsig{yf.flow_normal_stress_derivative(sig_dev, sig_eq)};

  // C_e : (dN/dsig) : C_e  ==  4G^2 (dN/dsig).
  //
  // Every flow normal here is deviatoric plus a constant volumetric part, so
  // dN/dsig is deviatoric in BOTH index pairs. For isotropic C_e that makes
  // C_e : X = 2G X and X : C_e = 2G X, and the two rank-4 x rank-4
  // contractions collapse to a scalar multiply. Verified to 2.5e-16 against
  // the explicit form for both the J2 and Drucker-Prager derivatives.
  //
  // Isotropy is not newly assumed: effective_modulus() above already requires
  // it (3G for J2, G + K*eta*beta for DP).
  const tensor4 C_dN_C{T{4} * G * G * dN_dsig};
  const tensor4 A{C_e - total_dlambda * C_dN_C};

  return A + tmech::otimes(dsig_ddlambda, dlambda_deps);
}

} // namespace numsim::materials::plasticity_detail

#endif // NUMSIM_MATERIALS_PLASTICITY_UTILS_H
