#ifndef NUMSIM_MATERIALS_CORE_DERIVATIVE_KERNELS_H
#define NUMSIM_MATERIALS_CORE_DERIVATIVE_KERNELS_H

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <tmech/tmech.h>
#include "numsim-materials/core/derivatives.h"

/// The chain rule on the values themselves: d node / d seed is a scalar or a tmech tensor of rank
/// rank(node) + rank(seed), and a step d out / d in (ranks a, b) applied to it is a tmech product over the b
/// indices of the middle value: otimes (b = 0), dcontract (b = 2) or ddcontract (b = 4).
namespace numsim::materials::derivatives {

/// The value of rank `Rank`: T for 0, else a tmech tensor.
template <typename T, std::size_t Dim, std::size_t Rank>
using value_t = typename partial_value<T, Dim, Rank>::type;

/// out = P * in (assigning) or out += P * in, the values given by address: P of rank a + b, in of rank b + s,
/// out of rank a + s; `scratch`, of rank a + s, for an accumulated contraction.
template <typename T>
using step_kernel = void (*)(void const* P, void const* in, void* out, void* scratch) noexcept;

/// Whether a step needs a scratch value: tmech's contraction carries temporaries that `o += ` would copy onto the stack.
constexpr bool needs_scratch(std::size_t a, std::size_t b, std::size_t s, bool assign) noexcept { return !assign && b != 0 && a + s != 0; }

template <typename T, std::size_t Dim, std::size_t A, std::size_t B, std::size_t S, bool Assign>
void tensor_step(void const* P, void const* in, void* out, [[maybe_unused]] void* scratch) noexcept {
  auto const& p{*static_cast<value_t<T, Dim, A + B> const*>(P)};
  auto const& d{*static_cast<value_t<T, Dim, B + S> const*>(in)};
  auto& o{*static_cast<value_t<T, Dim, A + S>*>(out)};
  auto const product{[&] {
    if constexpr (B == 0 && (A == 0 || S == 0))
      return p * d;
    else if constexpr (B == 0)
      return tmech::otimes(p, d);
    else if constexpr (B == 2)
      return tmech::dcontract(p, d);
    else
      return tmech::ddcontract(p, d);
  }};
  if constexpr (Assign) {
    o = product();
  } else if constexpr (needs_scratch(A, B, S, Assign)) {
    auto& t{*static_cast<value_t<T, Dim, A + S>*>(scratch)};
    t = product();
    o += t;
  } else {
    o += product();
  }
}

/// From a seed, whose d seed / d seed is the identity: out = P or out += P.
template <typename T, std::size_t Dim, std::size_t Rank, bool Assign>
void seed_step(void const* P, void const*, void* out, void*) noexcept {
  auto const& p{*static_cast<value_t<T, Dim, Rank> const*>(P)};
  auto& o{*static_cast<value_t<T, Dim, Rank>*>(out)};
  if constexpr (Assign)
    o = p;
  else
    o += p;
}

namespace detail {
inline std::size_t rank_index(std::size_t rank) {
  if (rank != 0 && rank != 2 && rank != 4)
    throw std::logic_error("derivatives: no node value has rank " + std::to_string(rank));
  return rank / 2;
}
} // namespace detail

/// The kernel of a step d out / d in (ranks a, b) applied to d in / d seed (rank s); chosen once per plan.
template <typename T, std::size_t Dim>
step_kernel<T> step_kernel_for(std::size_t a, std::size_t b, std::size_t s, bool assign) {
  constexpr auto table{[]<std::size_t... k>(std::index_sequence<k...>) {
    return std::array<step_kernel<T>, 54>{&tensor_step<T, Dim, 2 * ((k / 9) % 3), 2 * ((k / 3) % 3), 2 * (k % 3), (k >= 27)>...};
  }(std::make_index_sequence<54>{})};
  return table[(assign ? 27 : 0) + detail::rank_index(a) * 9 + detail::rank_index(b) * 3 + detail::rank_index(s)];
}

/// The kernel of a step d out / d seed (ranks a, s) taken straight from the seed.
template <typename T, std::size_t Dim>
step_kernel<T> seed_kernel_for(std::size_t a, std::size_t s, bool assign) {
  constexpr auto table{[]<std::size_t... k>(std::index_sequence<k...>) {
    return std::array<step_kernel<T>, 18>{&seed_step<T, Dim, 2 * ((k / 3) % 3) + 2 * (k % 3), (k >= 9)>...};
  }(std::make_index_sequence<18>{})};
  return table[(assign ? 9 : 0) + detail::rank_index(a) * 3 + detail::rank_index(s)];
}

/// out = f * in for a value of rank `Rank`: the scalar fold dx/dp = -(1 / J) dR/dp.
template <typename T>
using scale_kernel = void (*)(T f, void const* in, void* out) noexcept;

template <typename T, std::size_t Dim, std::size_t Rank>
void scale_step(T f, void const* in, void* out) noexcept {
  *static_cast<value_t<T, Dim, Rank>*>(out) = f * *static_cast<value_t<T, Dim, Rank> const*>(in);
}

template <typename T, std::size_t Dim>
scale_kernel<T> scale_kernel_for(std::size_t rank) {
  constexpr std::array<scale_kernel<T>, 3> table{&scale_step<T, Dim, 0>, &scale_step<T, Dim, 2>, &scale_step<T, Dim, 4>};
  return table[detail::rank_index(rank)];
}

/// A zero value of rank 0 to 8.
template <typename T, std::size_t Dim>
std::unique_ptr<value_holder<T>> make_value(std::size_t rank) {
  switch (rank) {
  case 0: return std::make_unique<derivatives::typed_value<T, T>>();
  case 2: return std::make_unique<derivatives::typed_value<T, value_t<T, Dim, 2>>>();
  case 4: return std::make_unique<derivatives::typed_value<T, value_t<T, Dim, 4>>>();
  case 6: return std::make_unique<derivatives::typed_value<T, value_t<T, Dim, 6>>>();
  case 8: return std::make_unique<derivatives::typed_value<T, value_t<T, Dim, 8>>>();
  default: throw std::logic_error("derivatives: no derivative has rank " + std::to_string(rank));
  }
}

/// d x / d x for x of rank 0, 2 or 4 into `out` (rank 2 rank): 1, the symmetric identity on tensor2, or its
/// counterpart on minor-symmetric tensor4; for an unsymmetric x (`full`) the identity d_ik d_jl (rank 8: d_im d_jn d_ko d_lp).
template <typename T, std::size_t Dim>
void identity(std::size_t rank, void* out, bool full = false) {
  auto const I{tmech::eye<T, Dim, 2>()};
  value_t<T, Dim, 4> const Is{full ? value_t<T, Dim, 4>{tmech::otimesu(I, I)} : value_t<T, Dim, 4>{T{0.5} * (tmech::otimesu(I, I) + tmech::otimesl(I, I))}};
  switch (rank) {
  case 0: *static_cast<T*>(out) = T{1}; return;
  case 2: *static_cast<value_t<T, Dim, 4>*>(out) = Is; return;
  case 4:
    *static_cast<value_t<T, Dim, 8>*>(out) = tmech::outer_product<tmech::sequence<1, 2, 5, 6>, tmech::sequence<3, 4, 7, 8>>(Is, Is);
    return;
  default: throw std::logic_error("derivatives: no node value has rank " + std::to_string(rank));
  }
}

} // namespace numsim::materials::derivatives

#endif // NUMSIM_MATERIALS_CORE_DERIVATIVE_KERNELS_H
