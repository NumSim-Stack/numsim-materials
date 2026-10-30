#ifndef NUMSIM_MATERIALS_CORE_HISTORY_KIND_H
#define NUMSIM_MATERIALS_CORE_HISTORY_KIND_H

#include <tmech/tmech.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>

namespace numsim::materials {

/// Configuration an index lives in: a rotation acts on spatial legs only.
enum class leg : unsigned char { spatial, material };

/// STATEV storage: shear doubled (strains), shear as is, or all components (vectors, non-symmetric tensors).
enum class storage : unsigned char { engineering, plain, full };

/// How a host stores and rotates a vector or rank-2 history; given when the history is declared.
struct history_kind {
  storage store{storage::engineering};
  std::array<leg, 2> legs{leg::spatial, leg::spatial}; ///< a vector uses legs[0]

  constexpr bool operator==(history_kind const&) const = default;

  /// A vector takes all components; six slots need a rank-2 tensor with both legs alike.
  constexpr bool valid_for(std::size_t rank) const {
    if (rank == 1)
      return store == storage::full;
    if (rank == 2)
      return store == storage::full || legs[0] == legs[1];
    return false;
  }

  /// Whether a rotation acts on some leg a tensor of @p rank uses.
  constexpr bool rotates(std::size_t rank) const { return std::ranges::contains(used(rank), leg::spatial); }

  /// Q·a for a vector, Q₀·a·Q₁ᵀ for a rank-2 tensor, with Q = R on a spatial leg and I on a material one.
  template <typename T, std::size_t Rank>
  tmech::tensor<T, 3, Rank> rotated(tmech::tensor<T, 3, Rank> const& a, tmech::tensor<T, 3, 2> const& R) const {
    static_assert(Rank == 1 || Rank == 2, "history_kind: a vector or rank-2 tensor");
    tmech::tensor<T, 3, 2> const I = tmech::eye<T, 3, 2>();
    auto const Q = [&](leg l) -> tmech::tensor<T, 3, 2> const& { return l == leg::spatial ? R : I; };
    tmech::tensor<T, 3, Rank> out;
    if constexpr (Rank == 1)
      out = Q(legs[0]) * a;
    else
      out = Q(legs[0]) * a * tmech::trans(Q(legs[1]));
    return out;
  }

  /// Storage and leg marks for a STATEV layout, e.g. "[engineering shear] [reference]".
  std::string tags(std::size_t rank) const {
    std::string out = store == storage::engineering ? "[engineering shear]"
                      : store == storage::plain     ? "[stress]"
                      : rank == 1                   ? "[vector]"
                                                    : "[full]";
    if (std::ranges::all_of(used(rank), [](leg l) { return l == leg::material; }))
      out += " [reference]";
    else if (std::ranges::contains(used(rank), leg::material))
      out += " [two-point]";
    return out;
  }

private:
  constexpr std::span<leg const> used(std::size_t rank) const { return {legs.data(), std::min(rank, legs.size())}; }
};

/// The kind as an error message names it: {storage, leg, leg}.
inline std::string to_string(history_kind const& kind) {
  constexpr std::array<char const*, 3> stores{"engineering", "plain", "full"};
  constexpr std::array<char const*, 2> legs{"spatial", "material"};
  return std::string{"{"} + stores[std::to_underlying(kind.store)] + ", " + legs[std::to_underlying(kind.legs[0])] + ", " +
         legs[std::to_underlying(kind.legs[1])] + "}";
}

namespace history_kinds {
inline constexpr history_kind strain{storage::engineering, {leg::spatial, leg::spatial}};             // εp, eigenstrains
inline constexpr history_kind stress_like{storage::plain, {leg::spatial, leg::spatial}};              // back stress
inline constexpr history_kind reference_strain{storage::engineering, {leg::material, leg::material}}; // Cp, Ep
inline constexpr history_kind two_point{storage::full, {leg::spatial, leg::material}};                // Fe
inline constexpr history_kind material_tensor{storage::full, {leg::material, leg::material}};          // Fp
inline constexpr history_kind spatial_vector{storage::full, {leg::spatial, leg::spatial}};            // q, E, D, P, j
inline constexpr history_kind material_vector{storage::full, {leg::material, leg::material}};
inline constexpr history_kind spatial_moduli{storage::plain, {leg::spatial, leg::spatial}}; // conductivity, diffusivity
inline constexpr history_kind material_moduli{storage::plain, {leg::material, leg::material}};
} // namespace history_kinds

/// The rank a kind applies to: 1 for a vector, 2 for a rank-2 tensor, 0 for any other value.
template <typename V> inline constexpr std::size_t history_kind_rank{0};
template <typename T> inline constexpr std::size_t history_kind_rank<tmech::tensor<T, 3, 1>>{1};
template <typename T> inline constexpr std::size_t history_kind_rank<tmech::tensor<T, 3, 2>>{2};

/// The kind of a history declared without one: a spatial strain or a spatial vector; none for other values.
template <typename V> inline constexpr std::optional<history_kind> default_history_kind{};
template <typename T>
inline constexpr std::optional<history_kind> default_history_kind<tmech::tensor<T, 3, 2>>{history_kinds::strain};
template <typename T>
inline constexpr std::optional<history_kind> default_history_kind<tmech::tensor<T, 3, 1>>{history_kinds::spatial_vector};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_HISTORY_KIND_H
