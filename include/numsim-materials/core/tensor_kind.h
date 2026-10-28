#ifndef NUMSIM_MATERIALS_TENSOR_KIND_H
#define NUMSIM_MATERIALS_TENSOR_KIND_H

#include <tmech/tmech.h>

#include <array>
#include <cstddef>
#include <type_traits>

namespace numsim::materials {

/// Configuration an index lives in: a rotation acts on spatial legs only.
enum class leg : unsigned char { spatial, material };

/// STATEV storage: shear doubled (strains), shear as is, or all components (vectors, non-symmetric tensors).
enum class voigt : unsigned char { none, engineering, plain };

namespace detail {
template <voigt V, leg... Legs> consteval bool valid_kind() {
  constexpr std::array<leg, sizeof...(Legs)> legs{Legs...};
  if constexpr (sizeof...(Legs) == 1)
    return V == voigt::none;
  else if constexpr (sizeof...(Legs) == 2)
    return V == voigt::none || legs[0] == legs[1];
  else
    return false;
}
}  // namespace detail

/// A tensor whose type says how a host stores and rotates it; arithmetic is that of the tensor.
template <typename T, voigt V, leg... Legs> struct kinded : tmech::tensor<T, 3, sizeof...(Legs)> {
  static_assert(detail::valid_kind<V, Legs...>(),
                "kinded: a vector or rank-2 tensor; six-slot storage needs a rank-2 tensor with both legs alike");
  using base = tmech::tensor<T, 3, sizeof...(Legs)>;
  using base::base;
  using base::operator=;
  kinded() = default;
  kinded(base const &t) : base(t) {}
};

template <typename T> using stress_like = kinded<T, voigt::plain, leg::spatial, leg::spatial>;
template <typename T> using reference_strain = kinded<T, voigt::engineering, leg::material, leg::material>;
template <typename T> using two_point = kinded<T, voigt::none, leg::spatial, leg::material>;
template <typename T> using material_tensor = kinded<T, voigt::none, leg::material, leg::material>;
template <typename T> using spatial_vector = kinded<T, voigt::none, leg::spatial>;
template <typename T> using material_vector = kinded<T, voigt::none, leg::material>;
template <typename T> using spatial_moduli = kinded<T, voigt::plain, leg::spatial, leg::spatial>;
template <typename T> using material_moduli = kinded<T, voigt::plain, leg::material, leg::material>;

/// Applies @p R to every spatial leg of @p a: R·v, R·t·Rᵀ, R·t, t·Rᵀ; material legs are left alone.
template <leg... Legs, typename T>
tmech::tensor<T, 3, sizeof...(Legs)> rotate_legs(tmech::tensor<T, 3, sizeof...(Legs)> const &a,
                                                 tmech::tensor<T, 3, 2> const &R) {
  constexpr std::array<leg, sizeof...(Legs)> legs{Legs...};
  tmech::tensor<T, 3, sizeof...(Legs)> out;
  if constexpr (sizeof...(Legs) == 1) {
    if constexpr (legs[0] == leg::spatial)
      out = R * a;
    else
      out = a;
  } else if constexpr (legs[0] == leg::spatial && legs[1] == leg::spatial) {
    out = R * a * tmech::trans(R);
  } else if constexpr (legs[0] == leg::spatial) {
    out = R * a;
  } else if constexpr (legs[1] == leg::spatial) {
    out = a * tmech::trans(R);
  } else {
    out = a;
  }
  return out;
}

template <typename T, voigt V, leg... Legs>
kinded<T, V, Legs...> rotated(kinded<T, V, Legs...> const &a, tmech::tensor<T, 3, 2> const &R) {
  return rotate_legs<Legs...>(a, R);
}

}  // namespace numsim::materials

/// A kinded tensor takes part in tmech expressions like the tensor it derives from.
template <typename T, numsim::materials::voigt V, numsim::materials::leg... Legs>
struct tmech::is_tensor_type<numsim::materials::kinded<T, V, Legs...>> {
  using type = std::true_type;
  static constexpr bool value = true;
};

#endif  // NUMSIM_MATERIALS_TENSOR_KIND_H
