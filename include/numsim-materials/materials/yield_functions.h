#ifndef NUMSIM_MATERIALS_YIELD_FUNCTIONS_H
#define NUMSIM_MATERIALS_YIELD_FUNCTIONS_H

#include <cmath>
#include <tmech/tmech.h>

namespace numsim::materials {

/// J2 (von Mises) yield function policy.
///
/// F = σ_eq - σ_0 - H(α)
/// Associative flow rule: N = 3/2 · dev(σ) / σ_eq
///
/// Required interface for small_strain_plasticity:
///   equivalent_stress(sig_dev) → σ_eq
///   trial_yield(σ_eq, σ_0, H) → F
///   residual(σ_eq, Δλ, G, σ_0, H) → r
///   jacobian(G, dH) → dr/dΔλ
///   flow_normal(sig_dev, σ_eq) → N
///   flow_normal_stress_derivative(N, σ_eq) → ∂N/∂σ (tensor4)
template<typename T, std::size_t Dim>
struct j2_yield_function {
  using tensor2 = tmech::tensor<T, Dim, 2>;
  using tensor4 = tmech::tensor<T, Dim, 4>;

  static T equivalent_stress(const tensor2& sig_dev) {
    return std::sqrt(T{1.5} * tmech::dcontract(sig_dev, sig_dev));
  }

  static T trial_yield(T sig_eq, T sigma_0, T H) {
    return sig_eq - sigma_0 - H;
  }

  static T residual(T sig_eq, T dlambda, T G, T sigma_0, T H) {
    return sig_eq - T{3} * G * dlambda - sigma_0 - H;
  }

  static T jacobian(T G, T dH) {
    return -T{3} * G - dH;
  }

  static tensor2 flow_normal(const tensor2& sig_dev, T sig_eq) {
    return T{1.5} * sig_dev / sig_eq;
  }

  /// ∂N/∂σ — derivative of flow normal w.r.t. stress tensor.
  /// For J2: ∂N_ij/∂σ_mn = 1/σ_eq · (3/2 · IIdev_ijmn - N_ij · N_mn)
  static tensor4 flow_normal_stress_derivative(const tensor2& N, T sig_eq) {
    const auto I = tmech::eye<T, Dim, 2>();
    const auto IIsym = (tmech::otimesu(I, I) + tmech::otimesl(I, I)) * T{0.5};
    const auto IIvol = tmech::otimes(I, I) / T{Dim};
    const tensor4 IIdev{IIsym - IIvol};

    return (T{1.5} * IIdev - tmech::otimes(N, N)) / sig_eq;
  }
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_YIELD_FUNCTIONS_H
