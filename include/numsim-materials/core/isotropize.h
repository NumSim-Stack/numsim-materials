#ifndef NUMSIM_MATERIALS_CORE_ISOTROPIZE_H
#define NUMSIM_MATERIALS_CORE_ISOTROPIZE_H

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/tensor_identities.h"

/// The closest isotropic tensor (Frobenius projection onto Pvol and Pdev) of a rank-4 tangent.
namespace numsim::materials {

template <typename T>
struct elastic_moduli {
  T K;
  T G;
};

/// K = Pvol :: C / Dim, G = Pdev :: C / (2 (Dim (Dim + 1) / 2 - 1)); exact for an isotropic C.
template <typename T, std::size_t Dim>
[[nodiscard]] elastic_moduli<T> isotropize(tmech::tensor<T, Dim, 4> const &C) noexcept {
  constexpr T pdev_norm_sq{T(Dim * (Dim + 1) / 2 - 1)}; // Pdev :: Pdev
  return {tmech::ddcontract(vol_projector<T, Dim>(), C) / T(Dim), tmech::ddcontract(dev_projector<T, Dim>(), C) / (2 * pdev_norm_sq)};
}

template <typename T, std::size_t Dim>
[[nodiscard]] tmech::tensor<T, Dim, 4> isotropic_part(tmech::tensor<T, Dim, 4> const &C) noexcept {
  auto const [K, G]{isotropize(C)};
  return isotropic_stiffness<T, Dim>(K, G);
}

/// The moduli of an isotropic C; throws naming `who` if C deviates from its projection by more than tol.
template <typename T, std::size_t Dim>
[[nodiscard]] elastic_moduli<T> require_isotropic(tmech::tensor<T, Dim, 4> const &C, std::string_view who, T tol = T{1e-8}) {
  auto const moduli{isotropize(C)};
  if (tmech::norm(tmech::tensor<T, Dim, 4>{C - isotropic_stiffness<T, Dim>(moduli.K, moduli.G)}) > tol * tmech::norm(C))
    throw std::invalid_argument(std::string{who} + ": the tangent must be isotropic");
  return moduli;
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_ISOTROPIZE_H
