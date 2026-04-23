#ifndef NUMSIM_MATERIALS_DRUCKER_PRAGER_YIELD_FUNCTION_H
#define NUMSIM_MATERIALS_DRUCKER_PRAGER_YIELD_FUNCTION_H

#include <cmath>
#include <tmech/tmech.h>

namespace numsim::materials {

/// Drucker-Prager yield function policy.
///
/// Yield: F = sqrt(J2) + alpha * I1 - k - H(alpha_eq)
/// Flow:  N = dG/dsigma = s/(2*sqrt(J2)) + beta/3 * I  (non-associative)
///
/// Unlike J2, this policy is STATEFUL — it holds alpha (friction) and
/// beta (dilatancy) parameters. Constructed per material instance.
///
/// When alpha = beta, flow is associative.
/// When alpha = beta = 0, reduces to von Mises.
template<typename T, std::size_t Dim>
struct drucker_prager_yield_function {
  using tensor2 = tmech::tensor<T, Dim, 2>;
  using tensor4 = tmech::tensor<T, Dim, 4>;

  T alpha;  // friction parameter (yield surface shape)
  T beta;   // dilatancy parameter (flow direction)

  drucker_prager_yield_function(T alpha_, T beta_)
      : alpha(alpha_), beta(beta_) {}

  /// sqrt(J2) from deviatoric stress
  T equivalent_stress(const tensor2& sig_dev) const {
    return std::sqrt(T{0.5} * tmech::dcontract(sig_dev, sig_dev));
  }

  /// Yield function: F = sqrt(J2) + alpha*I1 - k - H
  /// I1 must be passed separately (not available from sig_dev alone).
  T trial_yield_with_pressure(T sqrt_j2, T I1, T k, T H) const {
    return sqrt_j2 + alpha * I1 - k - H;
  }

  /// For interface compatibility: without pressure term.
  /// Caller must add alpha*I1 to sqrt_j2 before calling.
  T trial_yield(T modified_sig_eq, T sigma_0, T H) const {
    return modified_sig_eq - sigma_0 - H;
  }

  /// Residual for return mapping.
  /// modified_sig_eq = sqrt(J2) + alpha*I1 at trial state.
  /// During Newton: pressure changes as I1_trial - 3*K*beta*dlambda (volumetric)
  T residual(T modified_sig_eq, T dlambda, T G, T sigma_0, T H) const {
    // For DP: the residual accounts for both deviatoric and volumetric return.
    // sqrt(J2) decreases by G*dlambda, I1 decreases by 9*K*alpha*beta*dlambda
    // Simplified: r = modified_sig_eq - (G + 9*K*alpha*beta)*dlambda - sigma_0 - H
    // But we don't have K here. Use the standard form for now.
    return modified_sig_eq - T{3} * G * dlambda - sigma_0 - H;
  }

  T jacobian(T G, T dH) const {
    return -T{3} * G - dH;
  }

  /// Flow normal: N = s/(2*sqrt(J2)) + beta/3 * I
  /// NON-ASSOCIATIVE when alpha != beta.
  tensor2 flow_normal(const tensor2& sig_dev, T sqrt_j2) const {
    const auto I = tmech::eye<T, Dim, 2>();
    return sig_dev / (T{2} * sqrt_j2) + (beta / T{3}) * I;
  }

  /// dN/dsigma
  tensor4 flow_normal_stress_derivative(const tensor2& N, T sqrt_j2) const {
    const auto I = tmech::eye<T, Dim, 2>();
    const auto IIsym = (tmech::otimesu(I, I) + tmech::otimesl(I, I)) * T{0.5};
    const auto IIvol = tmech::otimes(I, I) / T{Dim};
    const tensor4 IIdev{IIsym - IIvol};

    // Recover s from N: s = (N - beta/3*I) * 2*sqrt(J2)
    const tensor2 s{(N - (beta / T{3}) * I) * (T{2} * sqrt_j2)};

    // d[s/(2*sqrt(J2))]/dsigma = (IIdev - s⊗s/(2*J2)) / (2*sqrt(J2))
    const auto j2 = sqrt_j2 * sqrt_j2;
    return (IIdev - tmech::otimes(s, s) / (T{2} * j2)) / (T{2} * sqrt_j2);
  }
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DRUCKER_PRAGER_YIELD_FUNCTION_H
