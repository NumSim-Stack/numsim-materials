#ifndef CALCULIX_INTERFACE_H
#define CALCULIX_INTERFACE_H

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>

#include "numsim-materials/umat/umat_interface.h"

/// The CalculiX external-behaviour entry point (`call_external_umat_user`).
///
/// CalculiX has two external hooks. `call_external_umat` is reached from
/// umat_abaqus.f AFTER it has converted to the Abaqus convention, so
/// NUMSIM_MATERIALS_DEFINE_UMAT already serves it. `call_external_umat_user` is
/// the NATIVE hook, and this adapter translates it:
///
///   kode          #constants = -kode - 100                 (umat_user.f:37-43)
///   emec/emec0    TENSORIAL {11,22,33,12,13,23}, end/start; doubled to the
///                 engineering shear umat_dispatch expects  (umat_abaqus.f:280)
///   stre          PK2, in/out, unscaled
///   stiff(21)     symmetrized upper triangle               (umat_abaqus.f:335)
///   xstate*       FULL (nstate_, mi(1), #elem) arrays, NOT a per-point slice,
///                 so the callee indexes by iel/iint        (umat_main.f:40,233)
///   TIME          rebased onto the increment START         (umat_abaqus.f:187)
///
/// One emitted symbol serves one registered model: the deck picks a `@LIB,FUNC`
/// per material, so one FUNC implies one constant set.
///
/// SCOPE: geometrically linear. `emec` is Green-Lagrange and `stre` is PK2,
/// consumed as small-strain quantities. Exact for a linear `C:E` law (that IS
/// St-Venant-Kirchhoff); under NLGEOM an inelastic model would get a wrong
/// CONVERGED stress, not merely a slow tangent.
///
/// Errors belong to umat_dispatch: a setup fault zeroes outputs and terminates,
/// anything else zeroes them and asks for a cutback with PNEWDT = 0.25 (a valid
/// ccx pnewdt). REFUSED rather than ignored: `iorien != 0`, a nonzero `beta`,
/// and `ielas != 0`. IGNORED: deformation gradients and temperature.
///
/// ## Building and naming the .so
///
/// ccx resolves the plugin itself, and both halves of the name are its choice,
/// not ours. `external.c` prepends "lib" and appends ".so" to the LIB part of
/// `*MATERIAL, NAME=@LIB,FUNC`, then `dlsym`s FUNC **verbatim** -- no Fortran
/// mangling, no trailing underscore. So for
///
///     *MATERIAL, NAME=@numsimmat,my_model
///
/// the shared object must be `libnumsimmat.so` on ccx's library path, and this
/// macro must be instantiated with FUNC spelled exactly `my_model`:
///
///     NUMSIM_MATERIALS_DEFINE_CALCULIX_BEHAVIOUR(my_traits, my_model, "MYMODEL")
///
/// That is why FUNC carries no underscore here while the Abaqus entry point in
/// umat_interface.h is `umat_` -- that one is called from Fortran directly, this
/// one through dlsym. Getting it backwards fails at run time with ccx's
/// "unable to load function" and produces nothing at build time.
///
/// Two further constraints, both ccx's:
///
///  * The symbol must be EXPORTED. `extern "C"` alone is enough only while the
///    build leaves default visibility; a target compiled `-fvisibility=hidden`
///    hides it and ccx reports "unable to load function" with no other symptom.
///    Add `-fvisibility=default` for this translation unit if that changes.
///  * Keep LIB short. `external.c` builds the file name into a fixed `char
///    b[80]` with an unchecked `memcpy`, so a long name overwrites its stack.
///
namespace numsim::materials::umat {

/// The native hook is always full 3D.
inline constexpr std::size_t calculix_ntens = 6;
inline constexpr std::size_t calculix_nstiff =
    calculix_ntens * (calculix_ntens + 1) / 2;

/// Every argument the shim forwards, named — a positional list of same-typed
/// pointers lets `time`/`ttime` or `emec`/`emec0` transpose silently. Scalars
/// are by value; CalculiX always passes them by reference, so the macro derefs.
struct calculix_args {
  const char* amat{nullptr};
  int iel{0};   ///< 1-based
  int iint{0};  ///< 1-based
  int kode{0};  ///< -100 - #constants
  const double* mprops{nullptr};
  const double* emec{nullptr};   ///< tensorial strain, END of increment
  const double* emec0{nullptr};  ///< tensorial strain, START of increment
  const double* beta{nullptr};   ///< initial (residual) stress
  double dtime{0};
  double time{0};   ///< STEP time at the END of the increment
  double ttime{0};  ///< TOTAL time at the START of the step
  int icmd{0};      ///< 3 => stress only
  int ielas{0};
  int mi1{0};     ///< mi(1): max integration points per element
  int nstatv{0};  ///< nstate_: state variables PER integration point
  const double* statev_old{nullptr};  ///< xstateini, FULL array
  double* statev_new{nullptr};        ///< xstate, FULL array
  double* stress{nullptr};
  double* stiff{nullptr};  ///< 21 packed entries
  int iorien{0};
  double* pnewdt{nullptr};
};

inline void calculix_zero_outputs(double* stress, double* stiff) noexcept {
  if (stress)
    for (std::size_t i = 0; i < calculix_ntens; ++i) stress[i] = 0.0;
  if (stiff)
    for (std::size_t i = 0; i < calculix_nstiff; ++i) stiff[i] = 0.0;
}

/// Shared implementation behind the CalculiX external symbol.
///
/// Fixed-size stack work only, so nothing here can throw before umat_dispatch,
/// which owns the try/catch keeping exceptions out of Fortran. The guards report
/// through the fatal handler rather than throwing, for the same reason.
template <typename Traits>
void calculix_dispatch(const calculix_args& a, const char* model_name) noexcept {
  using T = typename Traits::value_type;
  static_assert(std::is_same_v<T, double>,
                "the CalculiX external interface is double-precision (ccxreal)");

  // umat_main.f does not rotate around this hook, unlike umat_abaqus.f; the
  // material owes results in the material frame (umat_user.f:86-104). Ignoring
  // iorien would return the wrong frame with no symptom.
  // ielas = 1 is an ELASTIC iteration: ccx is asking for a response with no
  // irreversible deformation. arpack.c, arpackbu.c and arpackcs.c set it, i.e.
  // every eigenvalue and buckling analysis. A material that ignores it returns
  // a tangent with plastic flow folded in, and the extracted eigenvalues are
  // quietly wrong. Nothing in this library can suppress irreversible effects on
  // request, so refuse rather than answer the wrong question.
  if (a.ielas != 0) {
    calculix_zero_outputs(a.stress, a.stiff);
    report_fatal(model_name,
                 "numsim CalculiX: an elastic iteration (ielas != 0) was "
                 "requested -- *BUCKLE and *FREQUENCY need a response with no "
                 "irreversible deformation, which this material cannot "
                 "produce. Use a linear elastic material for those steps.");
    return;
  }

  if (a.iorien != 0) {
    calculix_zero_outputs(a.stress, a.stiff);
    report_fatal(model_name,
                 "numsim CalculiX: *ORIENTATION (iorien != 0) is not supported "
                 "by this external behaviour — the native umat_user hook makes "
                 "frame rotation the material's own responsibility, and it is "
                 "not implemented here");
    return;
  }

  // *INITIAL CONDITIONS,TYPE=STRESS: dropping it makes a preloaded model wrong
  // from the first step.
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

  // Widened before negating so kode == INT_MIN cannot trap; clamped so a
  // malformed kode cannot become a negative count.
  const long long nconst_ll = -static_cast<long long>(a.kode) - 100;
  const int nconst = nconst_ll > 0 ? static_cast<int>(nconst_ll) : 0;
  const int nstatv = a.nstatv > 0 ? a.nstatv : 0;

  // Fortran column-major stride into the full state array. Getting this wrong is
  // silent: every point would share element 1 / point 1's state.
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

  // Tensorial -> engineering, so strain_from_buffer's halving lands back on the
  // tensorial strain. STRAN is the start, so STRAN + DSTRAN == emec.
  double stran[calculix_ntens];
  double dstran[calculix_ntens];
  for (std::size_t i = 0; i < calculix_ntens; ++i) {
    const double shear = (i < 3) ? 1.0 : 2.0;
    stran[i] = shear * a.emec0[i];
    dstran[i] = shear * (a.emec[i] - a.emec0[i]);
  }

  // The evaluator updates one array in place; CalculiX splits read and write, so
  // seed this point's block from the committed state.
  double* statev_point = nullptr;
  if (nstatv > 0 && a.statev_new) {
    statev_point = a.statev_new + point_offset;
    if (a.statev_old)
      std::copy_n(a.statev_old + point_offset, nstatv, statev_point);
  }

  // COLUMN-major (material_point_evaluator.h:82), which the packing below needs.
  double ddsdde36[calculix_ntens * calculix_ntens] = {0.0};

  // ccx gives step time at the increment END and total time at the STEP start.
  // Passing {time, ttime} through is right only on the very first increment.
  const double time2[2] = {a.time - a.dtime, a.ttime + a.time - a.dtime};

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

  // Column-major upper triangle. SYMMETRIZED as umat_abaqus.f:335-355 does: a
  // row-major read would transpose every off-diagonal of an asymmetric tangent,
  // and stiff(21) has no room for an antisymmetric part anyway. icmd == 3 leaves
  // stiff alone; on an error path ddsdde36 is already zeroed.
  if (a.stiff && a.icmd != 3) {
    for (std::size_t j = 0; j < calculix_ntens; ++j)
      for (std::size_t i = 0; i <= j; ++i)
        a.stiff[i + j * (j + 1) / 2] =
            0.5 * (ddsdde36[i + j * calculix_ntens] +
                   ddsdde36[j + i * calculix_ntens]);
  }
}

}  // namespace numsim::materials::umat

/// Emit a CalculiX-callable symbol @p FUNC bound to model @p MODELNAME. Place in
/// exactly ONE translation unit of the library ccx loads for `@LIB,FUNC`.
///
/// Signature is the `calculixptr` prototype (call_external_umat_user.c): all
/// arguments by pointer (ccxint == int, ccxreal == double), with the hidden
/// Fortran string length last. That length is `int` because the C typedef fixes
/// it — unlike the Abaqus entry, where the Fortran compiler chooses. Do not make
/// it configurable.
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
