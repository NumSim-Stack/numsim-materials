#ifndef NUMSIM_MATERIALS_UMAT_UMAT_INTERFACE_H
#define NUMSIM_MATERIALS_UMAT_UMAT_INTERFACE_H

#include <cstddef>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "numsim-materials/umat/errors.h"
#include "numsim-materials/umat/material_point_evaluator.h"
#include "numsim-materials/umat/plane_stress_evaluator.h"

/// The Abaqus/Standard UMAT entry point.
///
/// Two things happen here that the layers below deliberately avoid:
///
///  * the Fortran ABI (argument order, pointer-to-scalar, the hidden trailing
///    length for CMNAME's character*80), and
///  * per-thread context management.
///
/// Abaqus calls UMAT concurrently from multiple threads. A material_context is
/// not thread-safe and is neither copyable nor movable, so each thread builds
/// its own on first use. That this is *sufficient* — rather than needing one
/// context per integration point — is a consequence of the evaluator being
/// stateless: all point state lives in the host's STATEV array.
namespace numsim::materials::umat {

/// Element family from Abaqus's NDI/NSHR.
///
/// Plane strain and axisymmetric are indistinguishable here — both are
/// NDI=3, NSHR=1 — and deliberately need no distinction: both supply all four
/// strain components, so they take the identical code path (plane strain simply
/// happens to pass eps_33 = 0).
inline element_case case_from_element(int ndi, int nshr) {
  if (ndi == 3 && nshr == 3) return element_case::solid3d;
  if (ndi == 3 && nshr == 1) return element_case::plane_strain;
  if (ndi == 2 && nshr == 1) return element_case::plane_stress;
  throw fatal_error(
      "numsim UMAT: unsupported element type with NDI=" + std::to_string(ndi) +
      ", NSHR=" + std::to_string(nshr));
}

/// Component count for an element family, under a name that is not shadowed by
/// umat_dispatch's `int ntens` parameter.
inline std::size_t ntens_for(element_case ec) noexcept { return ntens(ec); }

/// Trim a Fortran character*N: blank-padded, not NUL-terminated.
inline std::string trim_fortran_name(const char* s, std::size_t len) {
  while (len > 0 && (s[len - 1] == ' ' || s[len - 1] == '\0')) --len;
  return std::string(s, len);
}

/// Registry of UMAT-callable models, keyed by CMNAME.
///
/// Registration happens once (typically from a static initialiser, since Abaqus
/// offers no init hook); contexts are then built lazily per thread.
template <typename Traits>
class umat_registry {
public:
  using context_type = material_context<Traits>;
  using evaluator_type = material_point_evaluator<Traits>;
  using ps_evaluator_type = plane_stress_evaluator<Traits>;
  using config = typename evaluator_type::config;
  using call = typename evaluator_type::call;
  /// Populates a context with materials AND calls finalize().
  using builder = std::function<void(context_type&)>;

  static umat_registry& instance() {
    static umat_registry reg;
    return reg;
  }

  void register_model(std::string cmname, builder build, config cfg,
                      typename ps_evaluator_type::options ps_opts = {}) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_models.insert_or_assign(
        std::move(cmname),
        model{std::move(build), std::move(cfg), ps_opts});
  }

  [[nodiscard]] bool contains(const std::string& cmname) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_models.contains(cmname);
  }

  /// STATEV width for a model, so a user can size *DEPVAR. Plane stress needs
  /// one slot more than the other families.
  [[nodiscard]] std::size_t nstatv(const std::string& cmname,
                                   element_case ec = element_case::solid3d) {
    auto& ts = thread_state_for(cmname);
    return ec == element_case::plane_stress ? ts.ps->nstatv()
                                            : ts.solid->nstatv();
  }

  [[nodiscard]] std::vector<std::string> describe_statev(
      const std::string& cmname, element_case ec = element_case::solid3d) {
    auto& ts = thread_state_for(cmname);
    return ec == element_case::plane_stress ? ts.ps->describe_statev()
                                            : ts.solid->describe_statev();
  }

  /// Evaluate one material point, dispatching on the element family.
  void evaluate(const std::string& cmname, const call& c) {
    auto& ts = thread_state_for(cmname);
    if (c.ec == element_case::plane_stress)
      ts.ps->evaluate(c);
    else
      ts.solid->evaluate(c);
  }

  /// Drop this thread's cached contexts. Only needed if models are
  /// re-registered after use, which outside tests they are not.
  void reset_thread_cache() { thread_cache().clear(); }

private:
  struct model {
    builder build;
    config cfg;
    typename ps_evaluator_type::options ps_opts;
  };

  struct thread_state {
    std::unique_ptr<context_type> ctx;
    std::unique_ptr<evaluator_type> solid;
    std::unique_ptr<ps_evaluator_type> ps;
  };

  static std::unordered_map<std::string, thread_state>& thread_cache() {
    static thread_local std::unordered_map<std::string, thread_state> cache;
    return cache;
  }

  thread_state& thread_state_for(const std::string& cmname) {
    auto& cache = thread_cache();
    if (auto it = cache.find(cmname); it != cache.end()) return it->second;

    model m;
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      auto it = m_models.find(cmname);
      if (it == m_models.end())
        throw fatal_error(
            "numsim UMAT: no model registered for material name '" + cmname +
            "' — check *MATERIAL, NAME= against the registered models");
      m = it->second;  // copy under the lock; building runs unlocked
    }

    thread_state ts;
    ts.ctx = std::make_unique<context_type>();
    m.build(*ts.ctx);
    if (!ts.ctx->is_finalized())
      throw fatal_error(
          "numsim UMAT: the builder for '" + cmname +
          "' returned without calling finalize() on the context");

    // Both evaluators share the thread's context. They hold no mutable state
    // beyond diagnostics, and only one is used per call, so this is safe — and
    // it lets one material serve solid and plane-stress elements in one job.
    ts.solid = std::make_unique<evaluator_type>(*ts.ctx, m.cfg);
    ts.ps = std::make_unique<ps_evaluator_type>(*ts.ctx, m.cfg, m.ps_opts);

    return cache.emplace(cmname, std::move(ts)).first->second;
  }

  mutable std::mutex m_mutex;
  std::unordered_map<std::string, model> m_models;
};

/// Zero the host's outputs. Used only on the fatal path, so that a handler
/// which (against contract) returns cannot hand Abaqus the uninitialised
/// contents of DDSDDE as if they were a real tangent.
template <typename T>
inline void zero_outputs(T* stress, T* ddsdde, std::size_t n) noexcept {
  if (stress)
    for (std::size_t i = 0; i < n; ++i) stress[i] = T{0};
  if (ddsdde)
    for (std::size_t i = 0; i < n * n; ++i) ddsdde[i] = T{0};
}

/// Shared implementation behind the extern "C" entry point.
///
/// A C++ exception must never unwind into Fortran, so everything is caught
/// here and routed by the classification in errors.h:
///
///  * fatal_error — a setup fault. Zero the outputs and terminate the analysis
///    through XIT. Returning instead would let Abaqus consume an untouched
///    DDSDDE as a valid tangent, and requesting a cutback instead would halve
///    the increment forever against a fault no increment size can fix.
///  * anything else — treated as a failure at this increment size, which is the
///    right default for exceptions escaping the constitutive models themselves.
///    Ask for a smaller increment via PNEWDT and let the host retry.
/// Every argument the shim forwards, grouped so the dispatch signature does not
/// grow past readability.
template <typename T>
struct dispatch_args {
  T* stress{nullptr};
  T* statev{nullptr};
  T* ddsdde{nullptr};
  T* sse{nullptr};
  T* spd{nullptr};
  T* scd{nullptr};
  const T* stran{nullptr};
  const T* dstran{nullptr};
  const T* time{nullptr};
  const T* drot{nullptr};
  T dtime{0};
  T* pnewdt{nullptr};
  const char* cmname{nullptr};
  std::size_t cmname_len{0};
  int ndi{0};
  int nshr{0};
  int ntens{0};
  int nstatv{0};
};

template <typename Traits>
void umat_dispatch(const dispatch_args<typename Traits::value_type>& a) noexcept {
  using T = typename Traits::value_type;
  const std::string name = trim_fortran_name(a.cmname, a.cmname_len);
  try {
    const auto ec = case_from_element(a.ndi, a.nshr);
    const auto n = ntens_for(ec);

    if (static_cast<std::size_t>(a.ntens) != n)
      throw fatal_error("numsim UMAT: NTENS=" + std::to_string(a.ntens) +
                        " is inconsistent with NDI/NSHR for material '" + name +
                        "'");

    typename umat_registry<Traits>::call c;
    c.stran = {a.stran, n};
    c.dstran = {a.dstran, n};
    c.stress = {a.stress, n};
    c.ddsdde = {a.ddsdde, n * n};
    c.statev = {a.statev, static_cast<std::size_t>(a.nstatv)};
    // TIME(2) is the total time at the START of the increment.
    c.time = a.time[1];
    c.dtime = a.dtime;
    c.ec = ec;
    // DROT is always 3x3, whatever the element family.
    if (a.drot) c.drot = {a.drot, 9};
    c.sse = a.sse;
    c.spd = a.spd;
    c.scd = a.scd;

    umat_registry<Traits>::instance().evaluate(name, c);
  } catch (const fatal_error& e) {
    zero_outputs(a.stress, a.ddsdde, static_cast<std::size_t>(a.ntens));
    invoke_fatal("Unrecoverable fault in material '" + name +
                 "' — terminating the analysis.\n  " + e.what());
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "numsim UMAT: material '%s' failed (%s) — requesting a "
                 "smaller increment\n",
                 name.c_str(), e.what());
    if (a.pnewdt) *a.pnewdt = T{0.25};
  } catch (...) {
    std::fprintf(stderr,
                 "numsim UMAT: material '%s' failed with an unknown error — "
                 "requesting a smaller increment\n",
                 name.c_str());
    if (a.pnewdt) *a.pnewdt = T{0.25};
  }
}

}  // namespace numsim::materials::umat

/// Type of the hidden trailing length Fortran appends for a character*N
/// argument.
///
/// Not detectable at compile time, and the two candidates differ in width, so
/// getting it wrong corrupts CMNAME. Modern Intel Fortran and gfortran on
/// x86-64 pass std::size_t; older toolchains passed int. Override before
/// including this header if your Abaqus build differs:
///
///   #define NUMSIM_MATERIALS_FORTRAN_STRLEN int
#ifndef NUMSIM_MATERIALS_FORTRAN_STRLEN
#define NUMSIM_MATERIALS_FORTRAN_STRLEN std::size_t
#endif

/// Emit the Fortran-callable `umat_` symbol. Place this in exactly ONE
/// translation unit of the shared library Abaqus loads.
///
/// It is a macro rather than a library-provided symbol because numsim-materials
/// is header-only: an inline definition is not guaranteed to be emitted, and a
/// symbol Fortran resolves by name must actually exist in the object file.
///
/// Arguments that are accepted but not used: RPL, DDSDDT, DRPLDE and DRPLDT are
/// for coupled temperature-displacement analysis; PREDEF/DPRED are field
/// variables; DFGRD0/DFGRD1 are the deformation gradients, unused by a
/// small-strain material. DROT and the energies ARE forwarded.
#define NUMSIM_MATERIALS_DEFINE_UMAT(TRAITS)                                   \
  extern "C" void umat_(                                                       \
      double* STRESS, double* STATEV, double* DDSDDE, double* SSE,             \
      double* SPD, double* SCD, double* /*RPL*/, double* /*DDSDDT*/,           \
      double* /*DRPLDE*/, double* /*DRPLDT*/, const double* STRAN,             \
      const double* DSTRAN, const double* TIME, const double* DTIME,           \
      const double* /*TEMP*/, const double* /*DTEMP*/,                         \
      const double* /*PREDEF*/, const double* /*DPRED*/, const char* CMNAME,   \
      const int* NDI, const int* NSHR, const int* NTENS, const int* NSTATV,    \
      const double* /*PROPS*/, const int* /*NPROPS*/,                          \
      const double* /*COORDS*/, const double* DROT, double* PNEWDT,            \
      const double* /*CELENT*/, const double* /*DFGRD0*/,                      \
      const double* /*DFGRD1*/, const int* /*NOEL*/, const int* /*NPT*/,       \
      const int* /*LAYER*/, const int* /*KSPT*/, const int* /*JSTEP*/,         \
      const int* /*KINC*/, NUMSIM_MATERIALS_FORTRAN_STRLEN CMNAME_LEN) {       \
    ::numsim::materials::umat::dispatch_args<double> a;                        \
    a.stress = STRESS;  a.statev = STATEV;  a.ddsdde = DDSDDE;                 \
    a.sse = SSE;        a.spd = SPD;        a.scd = SCD;                       \
    a.stran = STRAN;    a.dstran = DSTRAN;  a.time = TIME;                     \
    a.drot = DROT;      a.dtime = *DTIME;   a.pnewdt = PNEWDT;                 \
    a.cmname = CMNAME;                                                         \
    a.cmname_len = static_cast<std::size_t>(CMNAME_LEN);                       \
    a.ndi = *NDI;  a.nshr = *NSHR;  a.ntens = *NTENS;  a.nstatv = *NSTATV;     \
    ::numsim::materials::umat::umat_dispatch<TRAITS>(a);                       \
  }

#endif  // NUMSIM_MATERIALS_UMAT_UMAT_INTERFACE_H
