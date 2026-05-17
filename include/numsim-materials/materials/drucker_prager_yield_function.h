#ifndef NUMSIM_MATERIALS_DRUCKER_PRAGER_YIELD_FUNCTION_H
#define NUMSIM_MATERIALS_DRUCKER_PRAGER_YIELD_FUNCTION_H

#include <cmath>
#include <tmech/tmech.h>
#include "numsim-materials/materials/plasticity_utils.h"

namespace numsim::materials {

/// Drucker-Prager yield function policy.
///
/// Yield: F = q + eta * p - k - H(kappa)  where q = sqrt(J2), p = I1/3
/// Flow:  N = dG/dsigma = s/(2*q) + beta/3 * I  (non-associative)
///
/// eta  = pressure/friction coefficient in the yield function
/// beta = dilatancy coefficient in the plastic potential
/// When eta = beta: associative. When eta = beta = 0: von Mises (up to normalization).
///
/// Normalization convention: equivalent_stress returns q = √J₂ (not √(3J₂)).
/// The effective modulus (G + K·η·β), residual, and flow normal are all
/// consistent with this choice. See the documentation in small_strain_plasticity.md
/// for the full consistency requirements and the relationship to J2.
template<typename T, std::size_t Dim>
struct drucker_prager_yield_function {
  using tensor2 = tmech::tensor<T, Dim, 2>;
  using tensor4 = tmech::tensor<T, Dim, 4>;

  T eta{0};     // pressure/friction coefficient
  T beta{0};    // dilatancy coefficient
  T K_bulk{0};  // bulk modulus, needed for volumetric coupling

  drucker_prager_yield_function() = default;
  drucker_prager_yield_function(T eta_, T beta_, T K_bulk_ = T{0})
      : eta(eta_), beta(beta_), K_bulk(K_bulk_) {}

  /// Effective modulus for DP: G + K*eta*beta
  /// Accounts for volumetric-deviatoric coupling in return mapping.
  /// tr(N) = beta, so Δp = -K*beta*Δλ, and the pressure term in F
  /// decreases by eta*K*beta*Δλ per increment.
  T effective_modulus(T G) const {
    return G + K_bulk * eta * beta;
  }

  /// sqrt(J2) from deviatoric stress
  T equivalent_stress(const tensor2& sig_dev) const {
    return std::sqrt(T{0.5} * tmech::dcontract(sig_dev, sig_dev));
  }

  /// Modified equivalent stress including pressure: q + eta*p
  /// where p = I1/3 = tr(sigma)/3 (mean stress).
  /// This is the quantity that decreases by G_eff*Δλ during return mapping.
  T modified_equivalent_stress(const tensor2& sig, T sqrt_j2) const {
    return sqrt_j2 + eta * tmech::trace(sig) / T{3};
  }

  /// F = q + eta*p - k - H
  T trial_yield(const tensor2& sig, T sqrt_j2, T k, T H) const {
    return modified_equivalent_stress(sig, sqrt_j2) - k - H;
  }

  /// Residual for return mapping.
  /// F_corrected = (q_trial - G*Δλ) + eta*(p_trial - K*beta*Δλ) - k - H
  /// = modified_sig_eq - (G + K*eta*beta)*Δλ - k - H
  /// modified_sig_eq = q_trial + eta*p_trial is passed as sig_eq.
  /// G_eff is the EFFECTIVE modulus G + K*eta*beta.
  T residual(T modified_sig_eq, T dlambda, T G_eff, T k, T H) const {
    return modified_sig_eq - G_eff * dlambda - k - H;
  }

  T jacobian(T G_eff, T dH) const {
    return -G_eff - dH;
  }

  /// Yield normal: dF/dsigma = s/(2*q) + eta/3 * I  (factor 1/3 from p = I1/3)
  /// Different from flow normal when non-associative (eta != beta).
  tensor2 yield_normal(const tensor2& sig_dev, T sqrt_j2) const {
    const auto I = tmech::eye<T, Dim, 2>();
    return sig_dev / (T{2} * sqrt_j2) + (eta / T{3}) * I;
  }

  /// Flow normal: N = dG/dsigma = s/(2*q) + beta/3 * I (non-associative)
  tensor2 flow_normal(const tensor2& sig_dev, T sqrt_j2) const {
    const auto I = tmech::eye<T, Dim, 2>();
    return sig_dev / (T{2} * sqrt_j2) + (beta / T{3}) * I;
  }

  /// Check if the standard return overshoots the DP cone apex.
  /// When G_shear*Δλ ≥ q_trial, the deviatoric correction flips direction.
  /// G_shear is the plain shear modulus (not G_eff).
  bool needs_apex_return(T G_shear, T dlambda, T sqrt_j2) const {
    return G_shear * dlambda >= sqrt_j2;
  }

  /// Apex return: only pressure term in modified_sig_eq (q = 0 at apex).
  T apex_modified_sig_eq(const tensor2& sig) const {
    return eta * tmech::trace(sig) / T{3};
  }

  /// Effective modulus at the apex (only volumetric coupling, no G).
  T apex_effective_modulus() const {
    return K_bulk * eta * beta;
  }

  /// Compute plastic strain for apex return.
  /// At apex: s = 0, so dev(ε_p) = dev(ε). Volumetric: tr(ε_p) += β·Δκ.
  ///
  /// This is a projection algorithm: the deviatoric plastic strain is set
  /// directly to enforce q = 0, while Δκ is determined by the apex
  /// consistency condition η·p = k + H(κ_n + Δκ). It differs from a
  /// classical single-multiplier flow-rule update where deviatoric and
  /// volumetric corrections would both be tied strictly to one Δλ.
  tensor2 apex_plastic_strain(
      const tensor2& eps, const tensor2& eps_p_old, T dkappa) const
  {
    const auto I = tmech::eye<T, Dim, 2>();
    const auto trace_eps = tmech::trace(eps);
    const auto eps_dev = eps - (trace_eps / T{Dim}) * I;
    const auto trace_eps_p_old = tmech::trace(eps_p_old);
    return eps_dev + ((trace_eps_p_old + beta * dkappa) / T{Dim}) * I;
  }

  /// Apex tangent: C_ep = K*H'/(K*η*β + H') · I⊗I  (purely volumetric).
  ///
  /// This is a branch tangent: it is the algorithmic tangent for
  /// perturbations that remain on the active apex branch. The return map is
  /// nonsmooth at the apex, so this is not a unique classical derivative.
  /// Perturbations that leave the apex back to the smooth cone follow the
  /// smooth-branch tangent.
  ///
  /// For perfectly plastic (H'=0) with K*η*β = 0, returns zero tangent —
  /// the apex branch is rate-indifferent and has no stiffness. Callers
  /// should be aware that a zero tangent makes the global stiffness singular.
  tensor4 apex_tangent(T dH_val) const {
    const auto I = tmech::eye<T, Dim, 2>();
    const auto Kab = K_bulk * eta * beta;
    const auto denom = Kab + dH_val;
    // Scale-relative threshold: treat denom as zero if both contributions
    // are at machine-epsilon level relative to their magnitudes.
    const auto scale = std::abs(Kab) + std::abs(dH_val);
    if (std::abs(denom) <= std::numeric_limits<T>::epsilon() * scale)
      return tensor4{};
    return (K_bulk * dH_val / denom) * tmech::otimes(I, I);
  }

  /// dN/dσ — only the deviatoric part contributes (β/3·I is constant w.r.t. σ).
  ///
  /// dN/dσ = d/dσ[s/(2q)] = (IIdev - s⊗s/(2J₂)) / (2q)
  ///
  /// Takes sig_dev directly (not N) to avoid cancellation error from
  /// reconstructing s = (N - β/3·I)·2q when q is small.
  tensor4 flow_normal_stress_derivative(const tensor2& sig_dev, T sqrt_j2) const {
    const tensor4 IIdev{plasticity_detail::make_IIdev<T, Dim>()};
    const auto j2 = sqrt_j2 * sqrt_j2;
    return (IIdev - tmech::otimes(sig_dev, sig_dev) / (T{2} * j2)) / (T{2} * sqrt_j2);
  }
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DRUCKER_PRAGER_YIELD_FUNCTION_H
