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
/// alpha = friction parameter, beta = dilatancy parameter.
/// When alpha = beta: associative. When alpha = beta = 0: von Mises.
template<typename T, std::size_t Dim>
struct drucker_prager_yield_function {
  using tensor2 = tmech::tensor<T, Dim, 2>;
  using tensor4 = tmech::tensor<T, Dim, 4>;

  T alpha{0};
  T beta{0};

  drucker_prager_yield_function() = default;
  drucker_prager_yield_function(T alpha_, T beta_)
      : alpha(alpha_), beta(beta_) {}

  /// sqrt(J2) from deviatoric stress
  T equivalent_stress(const tensor2& sig_dev) const {
    return std::sqrt(T{0.5} * tmech::dcontract(sig_dev, sig_dev));
  }

  /// F = sqrt(J2) + alpha*I1 - k - H
  T trial_yield(const tensor2& sig, T sqrt_j2, T k, T H) const {
    const auto I1 = tmech::trace(sig);
    return sqrt_j2 + alpha * I1 - k - H;
  }

  /// Residual: at trial state with correction.
  /// During return mapping, sqrt(J2) decreases by G*dlambda,
  /// I1 decreases by 9*K*alpha*beta*dlambda (volumetric).
  /// For simplicity, use the shear-only form (exact for incompressible).
  T residual(T sqrt_j2, T dlambda, T G, T k, T H) const {
    return sqrt_j2 - G * dlambda - k - H;
  }

  T jacobian(T G, T dH) const {
    return -G - dH;
  }

  /// Flow normal: N = s/(2*sqrt(J2)) + beta/3 * I (non-associative)
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

    // Recover s from N
    const tensor2 s{(N - (beta / T{3}) * I) * (T{2} * sqrt_j2)};
    const auto j2 = sqrt_j2 * sqrt_j2;

    return (IIdev - tmech::otimes(s, s) / (T{2} * j2)) / (T{2} * sqrt_j2);
  }
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DRUCKER_PRAGER_YIELD_FUNCTION_H
