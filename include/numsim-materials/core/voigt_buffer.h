#ifndef NUMSIM_MATERIALS_CORE_VOIGT_BUFFER_H
#define NUMSIM_MATERIALS_CORE_VOIGT_BUFFER_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <tmech/tmech.h>

/// Flat buffers <-> tmech tensors, host-neutral: the canonical 6-slot (Abaqus order) and full layouts.
namespace numsim::materials {

/// The canonical buffer is always 6 slots in Abaqus/Standard order
/// {11, 22, 33, 12, 13, 23}.
inline constexpr std::size_t canonical_width = 6;

// ---------------------------------------------------------------------------
// (2) canonical 6-slot buffer <-> tmech, always Dim = 3
// ---------------------------------------------------------------------------
//
// The shear convention is asymmetric across these four functions, and that
// asymmetry is exactly the UMAT contract, not an oversight.
//
// One caveat on how it is achieved: the `false` in the rank-4 functions is
// INERT. tmech's abq_std honours _ShearStrain only on the rank-2 path; its
// rank-4 assign_tensor ignores the flag entirely. The no-scaling property of
// the tangent is therefore guaranteed by tmech's internals, not by the argument
// written here — and would silently acquire a factor 2 if tmech ever
// implemented it. The physical check in the tests (DDSDDE times the engineering
// strain vector must equal the tensor contraction) is what actually pins it.
//
// The conventions themselves:
//
//   strain  ShearStrain = true   STRAN/DSTRAN carry ENGINEERING shear (2*eps_12)
//   stress  ShearStrain = false  STRESS carries tensor components
//   tangent no scaling at all    DDSDDE maps engineering strain to stress, and
//                                the factor 2 is absorbed by the minor symmetry
//                                sum over kl = 12 and 21
//
// Applying a shear factor to the tangent as well would double-count it.

/// Strain buffer -> tensor. Halves the off-diagonals, undoing the engineering
/// convention the host uses for STRAN and DSTRAN.
template <typename T>
inline tmech::tensor<T, 3, 2> strain_from_buffer(const T buf6[canonical_width]) {
  const tmech::adaptor<const T, 3, 2, tmech::abq_std<3, true>> view(buf6);
  tmech::tensor<T, 3, 2> out;
  out = view;
  return out;
}

/// Tensor -> strain buffer. Doubles the off-diagonals back to engineering shear.
template <typename T>
inline void strain_to_buffer(const tmech::tensor<T, 3, 2>& eps,
                             T buf6[canonical_width]) {
  tmech::adaptor<T, 3, 2, tmech::abq_std<3, true>> view(buf6);
  view = eps;
}

/// Stress buffer -> tensor. No shear factor.
template <typename T>
inline tmech::tensor<T, 3, 2> stress_from_buffer(const T buf6[canonical_width]) {
  const tmech::adaptor<const T, 3, 2, tmech::abq_std<3, false>> view(buf6);
  tmech::tensor<T, 3, 2> out;
  out = view;
  return out;
}

/// Tensor -> stress buffer. No shear factor.
template <typename T>
inline void stress_to_buffer(const tmech::tensor<T, 3, 2>& sig,
                             T buf6[canonical_width]) {
  tmech::adaptor<T, 3, 2, tmech::abq_std<3, false>> view(buf6);
  view = sig;
}

/// Full buffer -> vector or rank-2 tensor, all components, row-major {11 12 13 21 ...}.
template <typename T, std::size_t Rank>
inline tmech::tensor<T, 3, Rank> full_from_buffer(const T* buf) {
  const tmech::adaptor<const T, 3, Rank, tmech::full<3>> view(buf);
  tmech::tensor<T, 3, Rank> out;
  out = view;
  return out;
}

/// Vector or rank-2 tensor -> full buffer (3 or 9 slots), no symmetry assumed.
template <typename T, std::size_t Rank>
inline void full_to_buffer(const tmech::tensor<T, 3, Rank>& t, T* buf) {
  tmech::adaptor<T, 3, Rank, tmech::full<3>> view(buf);
  view = t;
}

/// True if @p c is minor-symmetric to within a relative tolerance.
///
/// Mandel/Voigt storage keeps 36 of the 81 rank-4 components, so a tangent
/// without minor symmetry cannot round-trip. statev_map rejects such a tensor
/// outright; here the same hazard is only asserted, because tangent_to_buffer
/// sits in the innermost loop of every Gauss point and this check is 162
/// comparisons.
///
/// A production UMAT is built with -DNDEBUG, so in deployment a tangent lacking
/// minor symmetry is silently truncated to its 36 representable slots. The
/// consequence is bounded and worth stating precisely: STRESS is computed by
/// the material and is unaffected, so the converged answer is still correct;
/// only the Jacobian is degraded, costing global Newton convergence rate. That
/// asymmetry of treatment is deliberate — in statev_map a truncated state
/// variable would corrupt the solution itself, so there it throws.
template <typename T>
inline bool is_minor_symmetric(const tmech::tensor<T, 3, 4>& c,
                               T rel_tol = T{1e-12}) noexcept {
  T scale{0};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          scale = std::max(scale, std::abs(c(i, j, k, l)));
  const T tol = rel_tol * std::max(scale, T{1});

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          if (std::abs(c(i, j, k, l) - c(j, i, k, l)) > tol ||
              std::abs(c(i, j, k, l) - c(i, j, l, k)) > tol)
            return false;
  return true;
}

/// Rank-4 tangent -> canonical 6x6, ROW-major. Only minor-symmetric parts
/// survive: the 36 stored slots cannot represent more.
template <typename T>
inline void tangent_to_buffer(const tmech::tensor<T, 3, 4>& C,
                              T buf36[canonical_width * canonical_width]) {
  assert(is_minor_symmetric(C) &&
         "tangent_to_buffer: the tangent is not minor-symmetric, so packing it "
         "into 36 slots loses information");
  tmech::adaptor<T, 3, 4, tmech::abq_std<3, false>> view(buf36);
  view = C;
}

/// Canonical 6x6 (ROW-major) -> rank-4 tangent. The inverse of
/// tangent_to_buffer; the result is minor-symmetric by construction.
template <typename T>
inline tmech::tensor<T, 3, 4> tangent_from_buffer(
    const T buf36[canonical_width * canonical_width]) {
  const tmech::adaptor<const T, 3, 4, tmech::abq_std<3, false>> view(buf36);
  tmech::tensor<T, 3, 4> out;
  out = view;
  return out;
}

}  // namespace numsim::materials

#endif  // NUMSIM_MATERIALS_CORE_VOIGT_BUFFER_H
