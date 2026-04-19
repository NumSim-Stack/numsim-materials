#ifndef NUMSIM_MATERIALS_TANGENT_CHECKER_H
#define NUMSIM_MATERIALS_TANGENT_CHECKER_H

#include <cmath>
#include <print>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"

namespace numsim::materials {

/// Verifies a material's analytical tangent against numerical differentiation.
///
/// Uses tmech::num_diff_central to compute dstress/dstrain numerically,
/// then compares against the analytical tangent from the material.
///
/// Works by: perturbing the strain property → calling ctx.update_property()
/// to re-evaluate the full upstream chain → reading the stress result.
template<typename Traits>
struct tangent_check_result {
  typename Traits::value_type abs_error;
  typename Traits::value_type rel_error;
};

template<typename Traits>
tangent_check_result<Traits> check_tangent(
    material_context<Traits>& ctx,
    const std::string& strain_material,
    const std::string& strain_property,
    const std::string& stress_material,
    const std::string& stress_property,
    const std::string& tangent_property)
{
  using T = typename Traits::value_type;
  constexpr auto Dim = Traits::Dim;
  using tensor2 = tmech::tensor<T, Dim, 2>;
  using tensor4 = tmech::tensor<T, Dim, 4>;

  // Read current state
  auto& eps = ctx.template get_mutable<tensor2>(strain_material, strain_property);
  const tensor2 eps_saved{eps};
  const tensor4 C_analytical{ctx.template get<tensor4>(stress_material, tangent_property)};

  // Build stress function that updates the full chain
  auto stress_func = [&](const tensor2& trial_eps) -> tensor2 {
    eps = trial_eps;
    ctx.update_property(stress_material, stress_property);
    return ctx.template get<tensor2>(stress_material, stress_property);
  };

  // Numerical tangent via central differences
  tensor4 C_numerical = tmech::num_diff_central<tmech::sequence<1,2,3,4>>(stress_func, eps_saved);

  // Restore
  eps = eps_saved;
  ctx.update_property(stress_material, stress_property);

  // Frobenius norm
  auto frobenius = [](const tensor4& t) {
    T sum{0};
    for (std::size_t i = 0; i < Dim; ++i)
      for (std::size_t j = 0; j < Dim; ++j)
        for (std::size_t k = 0; k < Dim; ++k)
          for (std::size_t l = 0; l < Dim; ++l)
            sum += t(i,j,k,l) * t(i,j,k,l);
    return std::sqrt(sum);
  };

  const tensor4 diff{C_analytical - C_numerical};
  T abs_error = frobenius(diff);
  T norm = frobenius(C_analytical);
  T rel_error = (norm > T{1e-30}) ? abs_error / norm : abs_error;

  return {abs_error, rel_error};
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_TANGENT_CHECKER_H
