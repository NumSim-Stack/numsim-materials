#ifndef CALCULIX_INTERFACE_H
#define CALCULIX_INTERFACE_H

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>

#include "numsim-materials/umat/umat_interface.h"

/// The CalculiX external-behaviour entry point (`call_external_umat_user`).
///
/// CalculiX has two external hooks, and they do NOT share a convention:
///
///  * `call_external_umat` is reached from umat_abaqus.f AFTER that wrapper has
///    already converted to the Abaqus UMAT convention (STRAN = strain at the
///    start with engineering shear, DSTRAN the increment, DDSDDE a full 6x6,
///    STATEV already sliced to the integration point). A material built with
///    NUMSIM_MATERIALS_DEFINE_UMAT already serves it — no adapter is needed.
///
///  * `call_external_umat_user` is the NATIVE hook (umat_user.f). It speaks
///    CalculiX's own convention, which this adapter translates to and from:
///
///      - kode, not nprops: the 4th argument is `kode = -100 - #constants`, so
///        the constant count is `-kode - 100` (umat_user.f:37-43).
///      - emec / emec0 are the Lagrange mechanical strain at the END and START
///        of the increment, component order {11,22,33,12,13,23}, TENSORIAL
///        (no engineering doubling — umat_abaqus.f:280-283 doubles the shear
///        itself before handing strain to an Abaqus UMAT, which is exactly the
///        conversion this adapter performs).
///      - stre is the second Piola-Kirchhoff stress, in on entry / out on exit,
///        plain components with no scaling.
///      - stiff(21) is the upper triangle of the symmetric 6x6 tangent, packed
///        column-major, carrying no engineering factors.
///      - xstateini / xstate are the FULL three-dimensional state arrays
///        `(nstate_, mi(1), #elements)`, NOT a slice for this point. The callee
///        must index them itself — which is why iel, iint and mi are in the
///        signature. See the STATE VARIABLES note below.
///      - TIME must be rebased onto the START of the increment; ccx passes the
///        step time at its END. See the TIME note below.
///
/// One emitted symbol maps to one registered model (the CalculiX deck selects a
/// specific `@LIB,FUNC` per material), so the model name is baked into the macro
/// rather than taken from `amat`. That also means one FUNC implies one constant
/// set: the registry's props-invariance check is what enforces it, and its
/// message is phrased for Abaqus ("use distinct *MATERIAL names"), which in ccx
/// means a distinct FUNC.
///
/// SCOPE — GEOMETRICALLY LINEAR (SMALL STRAIN) ONLY. `emec` is the Green-Lagrange
/// strain and `stre` is PK2, while the models below consume and produce these as
/// small-strain quantities. For a linear `C:E` law the two coincide (that pairing
/// IS St-Venant-Kirchhoff), so the elastic case is exactly right. It is NOT
/// merely a slower tangent for anything else: under NLGEOM ccx passes large
/// Green-Lagrange strains, and feeding those to a small- or logarithmic-strain
/// return map produces a wrong CONVERGED stress, silently. Do not use this hook
/// for an inelastic model under finite deformation without first choosing a
/// conjugate stress-strain pair and converting here.
///
/// INHERITED ERROR SEMANTICS. Fault handling belongs to umat_dispatch, not to
/// this file: a setup fault zeroes the outputs and terminates via the fatal
/// handler, and any other exception zeroes the outputs and requests a cutback
/// with PNEWDT = 0.25 — which is a valid ccx pnewdt (0 < pnewdt < 1), so the
/// Abaqus-side convention transfers unchanged. The zeroed 6x6 is packed into
/// stiff(21) like any other result.
///
/// NOT handled, and rejected rather than ignored: a local material orientation
/// (`iorien != 0`) and a nonzero initial stress (`beta`). Both would otherwise be
/// silently wrong — see the guards in calculix_dispatch. Deliberately IGNORED:
/// the deformation gradients and Jacobians (small-strain scope, above),
/// temperature (constants are read once when the graph is built), and `ielas`,
/// whose elastic-iteration request an inelastic model would need to honour.
namespace numsim::materials::umat {

/// The native hook is always full 3D: six components, and the symmetric 6x6
/// tangent packed as 21 upper-triangular entries.
inline constexpr std::size_t calculix_ntens = 6;
inline constexpr std::size_t calculix_nstiff =
    calculix_ntens * (calculix_ntens + 1) / 2;

/// Every argument the CalculiX shim forwards, named.
///
/// The Abaqus side funnels through dispatch_args for the same reason: a long
/// positional list of same-typed pointers lets a transposition compile cleanly,
/// and `time`/`ttime` or `emec`/`emec0` are exactly the pairs that would be
/// swapped. Scalars are held by value; CalculiX passes them by reference as
/// Fortran always does, and the macro dereferences them, so a null there would
/// be a CalculiX defect rather than a case to handle.
struct calculix_args {
  const char* amat{nullptr};
  int iel{0};             ///< element number, 1-based
  int iint{0};            ///< integration point number, 1-based
  int kode{0};            ///< -100 - #constants
  const double* mprops{nullptr};
  const double* emec{nullptr};   ///< tensorial strain at the END
  const double* emec0{nullptr};  ///< tensorial strain at the START
  const double* beta{nullptr};   ///< initial (residual) stress
  double dtime{0};
  double time{0};   ///< STEP time at the END of the increment
  double ttime{0};  ///< TOTAL time at the START of the step
  int icmd{0};      ///< 3 => stress only, leave stiff alone
  int ielas{0};
  int mi1{0};      ///< mi(1): max integration points per element
  int nstatv{0};   ///< nstate_: state variables PER integration point
  const double* statev_old{nullptr};  ///< xstateini, FULL array
  double* statev_new{nullptr};        ///< xstate, FULL array
  double* stress{nullptr};
  double* stiff{nullptr};  ///< 21 packed entries
  int iorien{0};
  double* pnewdt{nullptr};
};

/// Zero what CalculiX reads back, in CalculiX's own shapes.
inline void calculix_zero_outputs(double* stress, double* stiff) noexcept {
  if (stress)
    for (std::size_t i = 0; i < calculix_ntens; ++i) stress[i] = 0.0;
  if (stiff)
    for (std::size_t i = 0; i < calculix_nstiff; ++i) stiff[i] = 0.0;
}

/// Shared implementation behind the CalculiX external symbol.
///
/// Everything here is fixed-size stack work with no allocation, so it cannot
/// throw before reaching umat_dispatch, which owns the try/catch that keeps a
/// C++ exception from unwinding into CalculiX's Fortran. The guards below
/// report through the same fatal handler rather than throwing, for that reason.
template <typename Traits>
void calculix_dispatch(const calculix_args& a, const char* model_name) noexcept {
  using T = typename Traits::value_type;
  static_assert(std::is_same_v<T, double>,
                "the CalculiX external interface is double-precision (ccxreal)");

  // A local orientation is NOT applied around this hook. umat_abaqus.f rotates
  // strain in and stress/stiffness out around its call (lines 202-276, 305-330),
  // but umat_main.f does no such thing around the native or external umat_user
  // call: umat_user.f:86-104 requires results in the MATERIAL frame and tells
  // the user to rotate with transformatrix(). Ignoring iorien would therefore
  // return results in the wrong frame with no symptom, so refuse instead.
  if (a.iorien != 0) {
    calculix_zero_outputs(a.stress, a.stiff);
    report_fatal(model_name,
                 "numsim CalculiX: *ORIENTATION (iorien != 0) is not supported "
                 "by this external behaviour — the native umat_user hook makes "
                 "frame rotation the material's own responsibility, and it is "
                 "not implemented here");
    return;
  }

  // *INITIAL CONDITIONS,TYPE=STRESS (umat_user.f:52). Dropping it would make a
  // preloaded model wrong from the first step, again with no symptom.
  if (a.beta) {
    for (std::size_t i = 0; i < calculix_ntens; ++i) {
      if (a.beta[i] != 0.0) {
        calculix_zero_outputs(a.stress, a.stiff);
        report_fatal(model_name,
                     "numsim CalculiX: a nonzero initial stress (*INITIAL "
                     "CONDITIONS,TYPE=STRESS) is not supported by this external "
                     "behaviour");
        return;
      }
    }
  }

  if (!a.emec || !a.emec0) {
    calculix_zero_outputs(a.stress, a.stiff);
    report_fatal(model_name,
                 "numsim CalculiX: the mechanical strain arrays are null");
    return;
  }

  // Widened before negating so kode == INT_MIN cannot trap, and clamped because
  // a malformed kode must not become a negative count.
  const long long nconst_ll = -static_cast<long long>(a.kode) - 100;
  const int nconst = nconst_ll > 0 ? static_cast<int>(nconst_ll) : 0;
  const int nstatv = a.nstatv > 0 ? a.nstatv : 0;

  // STATE VARIABLES. xstateini/xstate are the FULL arrays
  // `real*8 xstate(nstate_, mi(1), *)` (umat_main.f:40), and umat_main.f passes
  // their BASE to this hook (line 233) — unlike umat_abaqus.f:295, which slices
  // `xstate(1,iint,iel)` before calling. So the offset of this point's block is
  // the Fortran column-major stride. Getting this wrong does not fail loudly: it
  // makes every integration point share element 1 / point 1's state.
  std::size_t point_offset = 0;
  if (nstatv > 0) {
    if (a.iel < 1 || a.iint < 1 || a.mi1 < 1 || a.iint > a.mi1) {
      calculix_zero_outputs(a.stress, a.stiff);
      report_fatal(model_name,
                   "numsim CalculiX: iel/iint/mi(1) are inconsistent, so the "
                   "state-variable block for this integration point cannot be "
                   "located");
      return;
    }
    point_offset = static_cast<std::size_t>(nstatv) *
                   (static_cast<std::size_t>(a.iint - 1) +
                    static_cast<std::size_t>(a.mi1) *
                        static_cast<std::size_t>(a.iel - 1));
  }

  // Native emec is TENSORIAL; umat_dispatch consumes ENGINEERING shear
  // (strain_from_buffer halves slots 3-5). Doubling the shear here makes the
  // round-trip land back on the tensorial strain the models expect — the same
  // conversion umat_abaqus.f:280-283 applies before an Abaqus UMAT. STRAN is the
  // strain at the START (emec0); DSTRAN the increment, so STRAN+DSTRAN == emec.
  double stran[calculix_ntens];
  double dstran[calculix_ntens];
  for (std::size_t i = 0; i < calculix_ntens; ++i) {
    const double shear = (i < 3) ? 1.0 : 2.0;
    stran[i] = shear * a.emec0[i];
    dstran[i] = shear * (a.emec[i] - a.emec0[i]);
  }

  // The evaluator reads and writes ONE state array in place; CalculiX splits it
  // into a read-only xstateini and a write-only xstate. Seed this point's block
  // in the output from the committed input so the read sees last increment's
  // state and the write lands where CalculiX expects the update.
  double* statev_point = nullptr;
  if (nstatv > 0 && a.statev_new) {
    statev_point = a.statev_new + point_offset;
    if (a.statev_old)
      std::copy_n(a.statev_old + point_offset, nstatv, statev_point);
  }

  // umat_dispatch fills a full 6x6; CalculiX wants only the packed upper
  // triangle. Note this buffer is COLUMN-major (material_point_evaluator.h:82,
  // narrow_matrix writes host[a + b*n]), which the packing below relies on.
  double ddsdde36[calculix_ntens * calculix_ntens] = {0.0};

  // TIME must be rebased onto the START of the increment. ccx hands over the
  // step time at the END of the increment plus the total time at the start of
  // the STEP, so mirror umat_abaqus.f:187-188 exactly:
  //     abqtime(1) = time - dtime
  //     abqtime(2) = ttime + time - dtime
  // Passing {time, ttime} instead is correct only on the first increment of the
  // first step, and silently wrong for every rate- or time-dependent model after
  // that.
  const double time2[2] = {a.time - a.dtime,
                           a.ttime + a.time - a.dtime};

  dispatch_args<double> d;
  d.stress = a.stress;
  d.statev = statev_point;
  d.ddsdde = ddsdde36;
  d.stran = stran;
  d.dstran = dstran;
  d.time = time2;
  d.dtime = a.dtime;
  d.pnewdt = a.pnewdt;
  d.props = a.mprops;
  d.nprops = nconst;
  d.cmname = model_name;
  d.cmname_len = std::char_traits<char>::length(model_name);
  d.ndi = 3;
  d.nshr = 3;
  d.ntens = static_cast<int>(calculix_ntens);
  d.nstatv = nstatv;

  umat_dispatch<Traits>(d);

  // Pack into stiff(21): column-major upper triangle, so the 0-based (i, j) with
  // i <= j lands at i + j*(j+1)/2.
  //
  // SYMMETRIZED, exactly as the reference umat_abaqus.f:335-355 does
  // (stiff(2) = (ddsdde(1,2)+ddsdde(2,1))/2, and so on). Two reasons: ddsdde36
  // is column-major, so reading it row-major would transpose every off-diagonal
  // for a major-ASYMMETRIC tangent (non-associative flow, damage); and stiff(21)
  // has no room for an antisymmetric part anyway, so averaging is what CalculiX
  // itself keeps. For a symmetric tangent this is the identity.
  //
  // icmd == 3 requests stress only and leaves stiff untouched, as umat_abaqus.f
  // does; on an error path umat_dispatch has zeroed ddsdde36 already.
  if (a.stiff && a.icmd != 3) {
    for (std::size_t j = 0; j < calculix_ntens; ++j)
      for (std::size_t i = 0; i <= j; ++i)
        a.stiff[i + j * (j + 1) / 2] =
            0.5 * (ddsdde36[i + j * calculix_ntens] +
                   ddsdde36[j + i * calculix_ntens]);
  }
}

}  // namespace numsim::materials::umat

/// Emit a CalculiX-callable external-behaviour symbol @p FUNC bound to the
/// registered model @p MODELNAME. Place this in exactly ONE translation unit of
/// the shared library CalculiX dlopen's for `*MATERIAL, NAME=@LIB,FUNC`.
///
/// The signature is the `calculixptr` prototype from CalculiX's
/// call_external_umat_user.c: every argument is passed by pointer (ccxint ==
/// int, ccxreal == double), and the final `int` is the hidden Fortran length of
/// the leading character string. That trailing width is `int` because the C
/// typedef in call_external_umat_user.c fixes it as `int` — unlike the Abaqus
/// entry, where the Fortran compiler chooses and
/// NUMSIM_MATERIALS_FORTRAN_STRLEN exists to track it. Do not make it
/// configurable here; the C prototype is the contract.
///
/// Arguments the small-strain path does not use are named in comments so their
/// absence reads as deliberate, not forgotten.
#define NUMSIM_MATERIALS_DEFINE_CALCULIX_BEHAVIOUR(TRAITS, FUNC, MODELNAME)     \
  extern "C" void FUNC(                                                         \
      const char* AMAT, const int* IEL, const int* IINT,                        \
      const int* KODE, const double* MPROPS, const double* EMEC,                \
      const double* EMEC0, const double* BETA, const double* /*XOKL*/,          \
      const double* /*VOJ*/, const double* /*XKL*/, const double* /*VJ*/,       \
      const int* /*ITHERMAL*/, const double* /*T1L*/, const double* DTIME,      \
      const double* TIME, const double* TTIME, const int* ICMD,                 \
      const int* IELAS, const int* MI, const int* NSTATV,                       \
      const double* STATEV0, double* STATEV1, double* STRESS, double* STIFF,    \
      const int* IORIEN, const double* /*PGAUSS*/, const double* /*ORAB*/,      \
      double* PNEWDT, const int* /*IPKON*/, const int /*SIZE*/) {               \
    ::numsim::materials::umat::calculix_args a;                                 \
    a.amat = AMAT;          a.iel = *IEL;         a.iint = *IINT;               \
    a.kode = *KODE;         a.mprops = MPROPS;                                  \
    a.emec = EMEC;          a.emec0 = EMEC0;      a.beta = BETA;                \
    a.dtime = *DTIME;       a.time = *TIME;       a.ttime = *TTIME;             \
    a.icmd = *ICMD;         a.ielas = *IELAS;     a.mi1 = *MI;                  \
    a.nstatv = *NSTATV;                                                         \
    a.statev_old = STATEV0; a.statev_new = STATEV1;                             \
    a.stress = STRESS;      a.stiff = STIFF;                                    \
    a.iorien = *IORIEN;     a.pnewdt = PNEWDT;                                  \
    ::numsim::materials::umat::calculix_dispatch<TRAITS>(a, MODELNAME);         \
  }

#endif  // CALCULIX_INTERFACE_H
