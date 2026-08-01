#ifndef NUMSIM_MATERIALS_UMAT_TENSOR_CONVERSION_H
#define NUMSIM_MATERIALS_UMAT_TENSOR_CONVERSION_H

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tmech/tmech.h>
#include "numsim-materials/umat/errors.h"

/// Conversion layer between an Abaqus/Standard UMAT's raw `double*` arguments
/// and the `tmech` tensors the material graph works in.
///
/// Two independent concerns, deliberately kept separate:
///
///   host pointer  <--(1)-->  canonical 6-slot buffer  <--(2)-->  tmech tensor
///
/// (1) is pure index shuffling — widening the host's NTENS slots to 6 and back,
///     which is where the element-family differences live (see element_case).
/// (2) is the tensor packing, and is ALWAYS 3D: every element type evaluates the
///     same Dim=3 material, so no 2D adaptor is ever instantiated.
///
/// Keeping them apart means the ordering/shear conventions are stated exactly
/// once, in (2), and the 2D cases cannot quietly grow their own copy.
namespace numsim::materials::umat {

/// The Abaqus element family the host is calling for. Determines NTENS and the
/// slot mapping, nothing else.
enum class element_case { solid3d, plane_strain, axisymmetric, plane_stress };

/// The canonical buffer is always 6 slots in Abaqus/Standard order
/// {11, 22, 33, 12, 13, 23}.
inline constexpr std::size_t canonical_width = 6;

/// Canonical index of the 33 component — the out-of-plane slot every 2D case
/// turns on.
inline constexpr std::size_t canonical_33 = 2;

/// NTENS the host passes for each family.
inline constexpr std::size_t ntens(element_case c) noexcept {
  switch (c) {
    case element_case::solid3d:      return 6;
    case element_case::plane_strain:
    case element_case::axisymmetric: return 4;
    case element_case::plane_stress: return 3;
  }
  return 0;
}

/// Host slot k -> canonical slot index. Only the first ntens(c) entries are
/// meaningful; the tail is padding.
///
/// Note that plane_stress is NOT a prefix of the 3D map: its components are
/// {11, 22, 12}, so host slot 2 maps to canonical slot 3 (the 12 slot), not to
/// canonical slot 2 (the 33 slot). Assuming a prefix here is the single easiest
/// way to get plane stress silently wrong, because the result still looks like
/// a plausible 3-vector.
inline constexpr std::array<std::size_t, canonical_width> slot_map(
    element_case c) noexcept {
  switch (c) {
    case element_case::solid3d:      return {0, 1, 2, 3, 4, 5};
    case element_case::plane_strain:
    case element_case::axisymmetric: return {0, 1, 2, 3, 0, 0};
    case element_case::plane_stress: return {0, 1, 3, 0, 0, 0};
  }
  return {};
}

// ---------------------------------------------------------------------------
// (1) host pointer <-> canonical 6-slot buffer
// ---------------------------------------------------------------------------

/// Widen the host's NTENS-slot vector into the canonical 6-slot buffer. Slots
/// the host does not supply are zeroed.
///
/// `out_of_plane` fills the 33 slot for plane stress only, where the host does
/// not supply it and the caller has to iterate on it. Plane strain and
/// axisymmetric both DO supply the 33 component (as zero, and as the hoop
/// strain, respectively), so it is ignored for them.
template <typename T>
inline void widen_vector(const T* host, element_case c, T out6[canonical_width],
                         T out_of_plane = T{0}) noexcept {
  const auto map = slot_map(c);
  const auto n = ntens(c);
  for (std::size_t i = 0; i < canonical_width; ++i) out6[i] = T{0};
  for (std::size_t k = 0; k < n; ++k) out6[map[k]] = host[k];
  if (c == element_case::plane_stress) out6[canonical_33] = out_of_plane;
}

/// Narrow the canonical 6-slot buffer back to the host's NTENS slots.
template <typename T>
inline void narrow_vector(const T in6[canonical_width], element_case c,
                          T* host) noexcept {
  const auto map = slot_map(c);
  const auto n = ntens(c);
  for (std::size_t k = 0; k < n; ++k) host[k] = in6[map[k]];
}

/// Narrow the canonical 6x6 into the host's NTENS x NTENS DDSDDE.
///
/// Two storage conventions cross here, and both matter:
///
///  * `in36` is ROW-major, because that is how tmech's assign_tensor packs a
///    rank-4 tensor (ptr[i*W + j]).
///  * `host` is written COLUMN-major, because DDSDDE(NTENS,NTENS) is a Fortran
///    array. For a major-SYMMETRIC tangent the two agree and this is invisible;
///    for a non-associative (major-asymmetric) tangent it is a transpose, which
///    is why the tests exercise an asymmetric block specifically.
///
///    Narrower still: Abaqus uses only the SYMMETRIC part of DDSDDE unless the
///    unsymmetric solver is requested with *USER MATERIAL, UNSYMM. So the
///    transpose is observable only in that configuration — which is exactly the
///    configuration a non-associative model needs, and the one where getting it
///    wrong would be least likely to be noticed.
///
/// For plane stress the block is additionally statically condensed on the 33
/// row/column, which is what enforces sigma_33 = 0 in the tangent:
///
///     C_ps[a][b] = C[a][b] - C[a][33] * C[33][b] / C[33][33]
template <typename T>
inline void narrow_matrix(const T in36[canonical_width * canonical_width],
                          element_case c, T* host) {
  const auto map = slot_map(c);
  const auto n = ntens(c);
  const bool condense = (c == element_case::plane_stress);

  T c33{0};
  if (condense) {
    c33 = in36[canonical_33 * canonical_width + canonical_33];
    if (c33 == T{0})
      throw cutback_error(
          "narrow_matrix: plane-stress condensation needs a nonzero C_3333, "
          "got exactly zero — the 33 direction carries no stiffness, so "
          "sigma_33 = 0 cannot be enforced by condensation");
  }

  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b = 0; b < n; ++b) {
      const auto ia = map[a];
      const auto ib = map[b];
      T v = in36[ia * canonical_width + ib];
      if (condense)
        v -= in36[ia * canonical_width + canonical_33] *
             in36[canonical_33 * canonical_width + ib] / c33;
      host[a + b * n] = v;  // column-major
    }
  }
}

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

// ---------------------------------------------------------------------------
// (3) Co-rotational state update
// ---------------------------------------------------------------------------

/// Read Abaqus's DROT(3,3) — Fortran, hence COLUMN-major — as a tensor.
template <typename T>
inline tmech::tensor<T, 3, 2> rotation_from_buffer(const T* drot) {
  tmech::tensor<T, 3, 2> R;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) R(i, j) = drot[i + 3 * j];
  return R;
}

/// True if @p R is the identity to within @p tol, i.e. there is no rotation to
/// apply and the whole co-rotational update can be skipped.
template <typename T>
inline bool is_identity_rotation(const tmech::tensor<T, 3, 2>& R,
                                 T tol = T{1e-14}) noexcept {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (std::abs(R(i, j) - (i == j ? T{1} : T{0})) > tol) return false;
  return true;
}

/// Rotate a rank-2 state variable into the current configuration: a' = R a Rᵀ.
template <typename T>
inline tmech::tensor<T, 3, 2> rotate(const tmech::tensor<T, 3, 2>& a,
                                     const tmech::tensor<T, 3, 2>& R) {
  // operator* on two rank-2 tensors contracts the last index of the left with
  // the first of the right, i.e. the matrix product. (tmech::dot is for
  // first-order tensors only.)
  tmech::tensor<T, 3, 2> out;
  out = R * a * tmech::trans(R);
  return out;
}

/// Rotate a rank-4 state variable: c'_ijkl = R_ip R_jq R_kr R_ls c_pqrs.
///
/// Written as explicit loops rather than tmech expressions because the
/// four-index basis change has no single-call spelling; it is also cold, since
/// rank-4 state variables are rare.
template <typename T>
inline tmech::tensor<T, 3, 4> rotate(const tmech::tensor<T, 3, 4>& c,
                                     const tmech::tensor<T, 3, 2>& R) {
  tmech::tensor<T, 3, 4> out;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l) {
          T acc{0};
          for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q)
              for (int r = 0; r < 3; ++r)
                for (int sdx = 0; sdx < 3; ++sdx)
                  acc += R(i, p) * R(j, q) * R(k, r) * R(l, sdx) *
                         c(p, q, r, sdx);
          out(i, j, k, l) = acc;
        }
  return out;
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

}  // namespace numsim::materials::umat

#endif  // NUMSIM_MATERIALS_UMAT_TENSOR_CONVERSION_H
