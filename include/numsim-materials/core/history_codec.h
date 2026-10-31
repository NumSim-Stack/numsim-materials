#ifndef NUMSIM_MATERIALS_CORE_HISTORY_CODEC_H
#define NUMSIM_MATERIALS_CORE_HISTORY_CODEC_H

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>

#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/errors.h"
#include "numsim-materials/core/voigt_buffer.h"

/// How one history value sits in a flat host buffer (STATEV) and turns under a rotation R.
/// A codec has width, read and write; tag and rotate are optional (no rotate: never rotated).
namespace numsim::materials {

/// Codec C stores a V in C::width slots of a T buffer; write names the history `who` in its errors.
template <class C, class V, class T>
concept history_codec_for = requires(V& v, V const& cv, T* b, T const* cb, std::string_view who) {
  { C::width } -> std::convertible_to<std::size_t>;
  C::read(v, cb);
  C::write(cv, b, who);
};

/// Six slots hold only a symmetric tensor: a skew part is reported, not dropped.
template <class T>
void require_symmetric(tmech::tensor<T, 3, 2> const& t, std::string_view who) {
  T scale{0};
  T skew{0};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      scale = std::max(scale, std::abs(t(i, j)));
      if (j > i) skew = std::max(skew, std::abs(t(i, j) - t(j, i)));
    }
  const T tol = T{1e-12} * std::max(scale, T{1});

  if (skew > tol)
    throw fatal_error(
        "statev_map: history property '" + std::string{who} +
        "' holds a non-symmetric rank-2 tensor, which cannot be stored in 6 "
        "STATEV slots without losing the skew part");
}

template <class T>
void require_minor_symmetric(tmech::tensor<T, 3, 4> const& c, std::string_view who) {
  if (!is_minor_symmetric(c))
    throw fatal_error(
        "statev_map: history property '" + std::string{who} +
        "' holds a rank-4 tensor without minor symmetry, which "
        "cannot be stored in 36 STATEV slots without loss");
}

namespace codecs {

/// One slot, frame-indifferent.
struct scalar {
  static constexpr std::size_t width = 1;
  template <class T> static void read(T& a, T const* b) { a = *b; }
  template <class T> static void write(T const& a, T* b, std::string_view) { *b = a; }
};

/// Cp, Ep: a symmetric reference-frame tensor with engineering shear, never rotated.
struct reference_strain {
  static constexpr std::size_t width = canonical_width;
  static constexpr std::string_view tag = "[engineering shear] [reference]";
  template <class T> static void read(tmech::tensor<T, 3, 2>& a, T const* b) { a = strain_from_buffer<T>(b); }
  template <class T> static void write(tmech::tensor<T, 3, 2> const& a, T* b, std::string_view who) {
    require_symmetric(a, who);
    strain_to_buffer<T>(a, b);
  }
};

/// εp, eigenstrains: R·a·Rᵀ.
struct strain : reference_strain {
  static constexpr std::string_view tag = "[engineering shear]";
  template <class T>
  static tmech::tensor<T, 3, 2> rotate(tmech::tensor<T, 3, 2> const& a, tmech::tensor<T, 3, 2> const& R) {
    return materials::rotate(a, R);
  }
};

/// Back stress and spatial moduli (conductivity): shear as is, R·a·Rᵀ.
struct stress {
  static constexpr std::size_t width = canonical_width;
  static constexpr std::string_view tag = "[stress]";
  template <class T> static void read(tmech::tensor<T, 3, 2>& a, T const* b) { a = stress_from_buffer<T>(b); }
  template <class T> static void write(tmech::tensor<T, 3, 2> const& a, T* b, std::string_view who) {
    require_symmetric(a, who);
    stress_to_buffer<T>(a, b);
  }
  template <class T>
  static tmech::tensor<T, 3, 2> rotate(tmech::tensor<T, 3, 2> const& a, tmech::tensor<T, 3, 2> const& R) {
    return materials::rotate(a, R);
  }
};

/// Fp: all nine components, never rotated.
struct reference_tensor {
  static constexpr std::size_t width = 9;
  static constexpr std::string_view tag = "[full] [reference]";
  template <class T> static void read(tmech::tensor<T, 3, 2>& a, T const* b) { a = full_from_buffer<T, 2>(b); }
  template <class T> static void write(tmech::tensor<T, 3, 2> const& a, T* b, std::string_view) {
    full_to_buffer<T, 2>(a, b);
  }
};

/// Fe: the first leg spatial, R·a.
struct two_point : reference_tensor {
  static constexpr std::string_view tag = "[full] [two-point]";
  template <class T>
  static tmech::tensor<T, 3, 2> rotate(tmech::tensor<T, 3, 2> const& a, tmech::tensor<T, 3, 2> const& R) {
    return tmech::tensor<T, 3, 2>{R * a};
  }
};

/// A reference-frame vector, never rotated.
struct reference_vector {
  static constexpr std::size_t width = 3;
  static constexpr std::string_view tag = "[vector] [reference]";
  template <class T> static void read(tmech::tensor<T, 3, 1>& v, T const* b) { v = full_from_buffer<T, 1>(b); }
  template <class T> static void write(tmech::tensor<T, 3, 1> const& v, T* b, std::string_view) {
    full_to_buffer<T, 1>(v, b);
  }
};

/// q, E, D, j: R·v.
struct vector : reference_vector {
  static constexpr std::string_view tag = "[vector]";
  template <class T>
  static tmech::tensor<T, 3, 1> rotate(tmech::tensor<T, 3, 1> const& v, tmech::tensor<T, 3, 2> const& R) {
    return tmech::tensor<T, 3, 1>{R * v};
  }
};

/// A minor-symmetric rank-4 tensor in 36 slots, all four legs rotated.
struct moduli {
  static constexpr std::size_t width = canonical_width * canonical_width;
  template <class T> static void read(tmech::tensor<T, 3, 4>& c, T const* b) { c = tangent_from_buffer<T>(b); }
  template <class T> static void write(tmech::tensor<T, 3, 4> const& c, T* b, std::string_view who) {
    require_minor_symmetric(c, who);
    tangent_to_buffer<T>(c, b);
  }
  template <class T>
  static tmech::tensor<T, 3, 4> rotate(tmech::tensor<T, 3, 4> const& c, tmech::tensor<T, 3, 2> const& R) {
    return materials::rotate(c, R);
  }
};

} // namespace codecs

namespace detail {
template <class V> struct default_codec { using type = void; };
template <class V> requires std::is_arithmetic_v<V> struct default_codec<V> { using type = codecs::scalar; };
template <class T> struct default_codec<tmech::tensor<T, 3, 1>> { using type = codecs::vector; };
template <class T> struct default_codec<tmech::tensor<T, 3, 2>> { using type = codecs::strain; };
template <class T> struct default_codec<tmech::tensor<T, 3, 4>> { using type = codecs::moduli; };
} // namespace detail

/// The codec of a history declared without one; void when the value type has none.
template <class V> using default_codec = typename detail::default_codec<V>::type;

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_HISTORY_CODEC_H
