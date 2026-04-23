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
/// Stateless — default-constructible, all methods const.
template<typename T, std::size_t Dim>
struct j2_yield_function {
  using tensor2 = tmech::tensor<T, Dim, 2>;
  using tensor4 = tmech::tensor<T, Dim, 4>;

  T equivalent_stress(const tensor2& sig_dev) const {
    return std::sqrt(T{1.5} * tmech::dcontract(sig_dev, sig_dev));
  }

  T trial_yield(const tensor2& /*sig*/, T sig_eq, T sigma_0, T H) const {
    return sig_eq - sigma_0 - H;
  }

  T residual(T sig_eq, T dlambda, T G, T sigma_0, T H) const {
    return sig_eq - T{3} * G * dlambda - sigma_0 - H;
  }

  T jacobian(T G, T dH) const {
    return -T{3} * G - dH;
  }

  tensor2 flow_normal(const tensor2& sig_dev, T sig_eq) const {
    return T{1.5} * sig_dev / sig_eq;
  }

  tensor4 flow_normal_stress_derivative(const tensor2& N, T sig_eq) const {
    const auto I = tmech::eye<T, Dim, 2>();
    const auto IIsym = (tmech::otimesu(I, I) + tmech::otimesl(I, I)) * T{0.5};
    const auto IIvol = tmech::otimes(I, I) / T{Dim};
    const tensor4 IIdev{IIsym - IIvol};

    return (T{1.5} * IIdev - tmech::otimes(N, N)) / sig_eq;
  }
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_YIELD_FUNCTIONS_H
