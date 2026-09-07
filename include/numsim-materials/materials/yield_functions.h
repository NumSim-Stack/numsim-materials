#ifndef YIELD_FUNCTIONS_H
#define YIELD_FUNCTIONS_H

#include <cmath>
#include <tmech/tmech.h>
#include "numsim-materials/materials/plasticity_utils.h"

namespace numsim::materials {

/// J2 (von Mises) yield function policy.
///
/// F = σ_eq - σ_0 - H(κ)
/// Associative flow rule: N = 3/2 · s / σ_eq
///
/// Normalization convention: equivalent_stress returns σ_eq = √(3J₂).
/// The effective modulus (3G), residual, and flow normal are all consistent
/// with this choice. See docs/plasticity-theory.md for the derivation and
/// section 15 there for the ways this consistency is broken in practice.
///
/// Stateless — default-constructible, all methods const.
template<typename T, std::size_t Dim>
struct j2_yield_function {
  using tensor2 = tmech::tensor<T, Dim, 2>;
  using tensor4 = tmech::tensor<T, Dim, 4>;

  /// Effective modulus for residual/jacobian. For J2: G_eff = 3G.
  T effective_modulus(T G) const { return T{3} * G; }

  T equivalent_stress(const tensor2& sig_dev) const {
    return std::sqrt(T{1.5} * tmech::dcontract(sig_dev, sig_dev));
  }

  /// For J2: modified = equivalent (no pressure coupling).
  T modified_equivalent_stress(const tensor2& /*sig*/, T sig_eq) const {
    return sig_eq;
  }

  T trial_yield(const tensor2& /*sig*/, T sig_eq, T sigma_0, T H) const {
    return sig_eq - sigma_0 - H;
  }

  /// Residual: G_eff is effective_modulus(G) = 3G for J2.
  T residual(T sig_eq, T dlambda, T G_eff, T sigma_0, T H) const {
    return sig_eq - G_eff * dlambda - sigma_0 - H;
  }

  T jacobian(T G_eff, T dH) const {
    return -G_eff - dH;
  }

  /// Yield normal = flow normal for associative J2.
  tensor2 yield_normal(const tensor2& sig_dev, T sig_eq) const {
    return flow_normal(sig_dev, sig_eq);
  }

  tensor2 flow_normal(const tensor2& sig_dev, T sig_eq) const {
    return T{1.5} * sig_dev / sig_eq;
  }

  /// J2 yield surface is a cylinder — no apex, never needs apex return.
  bool needs_apex_return(T, T, T) const { return false; }

  /// dN/dσ = (3/2 · IIdev - N⊗N) / σ_eq
  /// Takes sig_dev (not N) to match the DP interface and avoid
  /// needing to reconstruct s from N.
  tensor4 flow_normal_stress_derivative(const tensor2& sig_dev, T sig_eq) const {
    const tensor4 IIdev{plasticity_detail::make_IIdev<T, Dim>()};
    const tensor2 N{T{1.5} * sig_dev / sig_eq};
    return (T{1.5} * IIdev - tmech::otimes(N, N)) / sig_eq;
  }
};

} // namespace numsim::materials

#endif // YIELD_FUNCTIONS_H
