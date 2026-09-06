#ifndef UMAT_INTERFACE_H
#define UMAT_INTERFACE_H

#include <algorithm>
#include <cctype>
#include <string_view>
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
/// strain components, so they take the identical code path. Ordinary plane
/// strain happens to pass eps_33 = 0, axisymmetric passes the hoop strain, and
/// generalized plane strain (CPEG) reports the same NDI/NSHR with a nonzero
/// eps_33. Whatever slot 2 holds is simply consumed, so all three are correct
/// without being told apart.
///
/// Element families NOT covered fall through to the fatal error, which is the
/// intended outcome rather than a gap: beams (NDI=1, NSHR=1), trusses (1, 0),
/// axisymmetric shells (2, 0) and cohesive traction-separation (1, 2) all use
/// component orderings this layer does not implement, and a prefix-based guess
/// would silently misinterpret them.
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
///
/// Returns a view into the caller's buffer rather than a string: this runs once
/// per UMAT call — per integration point, per global iteration — and CMNAME is
/// character*80, well past the small-string buffer, so a std::string here would
/// be a heap allocation in the hottest path the library has.
inline std::string_view trim_fortran_name(const char* s, std::size_t len) {
  while (len > 0 && (s[len - 1] == ' ' || s[len - 1] == '\0')) --len;
  return {s, len};
}

/// Fold a material name to the form used as the registry key.
///
/// Abaqus input is case-insensitive and CMNAME arrives upper-cased whatever the
/// deck says, so a model registered as "j2steel" against *MATERIAL, NAME=j2steel
/// would never be found. Both registration and lookup normalise, so the caller's
/// choice of case is irrelevant on either side.
inline std::string normalise_material_name(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return name;
}

/// Trim and fold CMNAME into @p buf, returning a NUL-terminated view of it.
/// The NUL lets the same buffer feed both the registry lookup and the C
/// formatting in the error paths, with no allocation on either.
template <std::size_t N>
inline std::string_view normalise_cmname(const char* s, std::size_t len,
                                         char (&buf)[N]) {
  const auto trimmed = trim_fortran_name(s, len);
  const auto n = std::min(trimmed.size(), N - 1);
  for (std::size_t i = 0; i < n; ++i)
    buf[i] = static_cast<char>(
        std::toupper(static_cast<unsigned char>(trimmed[i])));
  buf[n] = '\0';
  return {buf, n};
}

/// Heterogeneous hash so the registry can be looked up by string_view without
/// materialising a std::string.
struct transparent_string_hash {
  using is_transparent = void;
  std::size_t operator()(std::string_view s) const noexcept {
    return std::hash<std::string_view>{}(s);
  }
};

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
  ///
  /// Receives the deck's *USER MATERIAL constants. They are read once, here,
  /// and copied into each material's parameter handler — material_interface
  /// stores it BY VALUE — so the span need not outlive the call, and per-call
  /// evaluation never touches parameters again.
  ///
  /// The mapping from slot to parameter is positional and is the builder's
  /// contract with the deck; document it next to each builder, and validate the
  /// count with require_props() before indexing.
  using builder =
      std::function<void(context_type&, std::span<const double> props)>;

  /// CMNAME is character*80 in the Abaqus interface.
  static constexpr std::size_t max_cmname = 80;

  static umat_registry& instance() {
    static umat_registry reg;
    return reg;
  }

  void register_model(std::string cmname, builder build, config cfg,
                      typename ps_evaluator_type::options ps_opts = {}) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_models.insert_or_assign(
        normalise_material_name(std::move(cmname)),
        model{std::move(build), std::move(cfg), ps_opts});
  }

  [[nodiscard]] bool contains(std::string_view cmname) const {
    char buf[max_cmname + 1];
    const auto key = normalise_cmname(cmname.data(), cmname.size(), buf);
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_models.contains(key);
  }

  /// STATEV width for a model, so a user can size *DEPVAR. Plane stress needs
  /// one slot more than the other families.
  [[nodiscard]] std::size_t nstatv(std::string_view cmname,
                                   std::span<const double> props = {},
                                   element_case ec = element_case::solid3d) {
    auto& ts = thread_state_for(cmname, props);
    return ec == element_case::plane_stress ? ts.ps->nstatv()
                                            : ts.solid->nstatv();
  }

  [[nodiscard]] std::vector<std::string> describe_statev(
      std::string_view cmname, std::span<const double> props = {},
      element_case ec = element_case::solid3d) {
    auto& ts = thread_state_for(cmname, props);
    return ec == element_case::plane_stress ? ts.ps->describe_statev()
                                            : ts.solid->describe_statev();
  }

  /// Evaluate one material point, dispatching on the element family.
  void evaluate(std::string_view cmname, std::span<const double> props,
                const call& c) {
    auto& ts = thread_state_for(cmname, props);
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

  /// Declaration order matters on USE, not on teardown. The evaluators hold raw
  /// pointers into the context's property storage, so an evaluator outliving its
  /// context is a use-after-free — but no destructor here dereferences the
  /// context, so the reverse order would in fact be clean today. Keeping the
  /// context first anyway means a future destructor body, or a reordering, does
  /// not quietly turn a latent hazard into a live one. Compare the load-bearing
  /// `DO NOT REORDER` ordering in material_context itself.
  struct thread_state {
    std::unique_ptr<context_type> ctx;
    std::unique_ptr<evaluator_type> solid;
    std::unique_ptr<ps_evaluator_type> ps;
    /// How many constants the context was built from. The graph is built once
    /// and reused, so a later call arriving with a different count would mean
    /// the cached parameters no longer describe this material.
    std::size_t nprops{0};
  };

  static std::unordered_map<std::string, thread_state, transparent_string_hash,
                            std::equal_to<>>&
  thread_cache() {
    static thread_local std::unordered_map<std::string, thread_state,
                                           transparent_string_hash,
                                           std::equal_to<>>
        cache;
    return cache;
  }

  thread_state& thread_state_for(std::string_view cmname,
                                 std::span<const double> props) {
    char buf[max_cmname + 1];
    const auto key = normalise_cmname(cmname.data(), cmname.size(), buf);
    auto& cache = thread_cache();
    if (auto it = cache.find(key); it != cache.end()) {
      // PROPS cannot vary for a given material name — two *MATERIAL blocks must
      // have distinct names — so a changed count means the deck contradicts the
      // cached graph. Checking the size is one comparison; checking the values
      // is not worth it per integration point.
      if (it->second.nprops != props.size())
        throw fatal_error(
            "numsim UMAT: material '" + std::string(key) +
            "' was built from " + std::to_string(it->second.nprops) +
            " constants but this call supplies " +
            std::to_string(props.size()) +
            " — PROPS must be constant for a given material name");
      return it->second;
    }

    model m;
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      auto it = m_models.find(key);
      if (it == m_models.end())
        throw fatal_error(
            "numsim UMAT: no model registered for material name '" +
            std::string(key) +
            "' — check *MATERIAL, NAME= against the registered models");
      m = it->second;  // copy under the lock; building runs unlocked
    }

    // Everything from here to the end of construction is SETUP, so every
    // failure is fatal regardless of which layer raised it.
    //
    // Classifying at each throw site does not work here: the builder runs
    // arbitrary graph construction, and the core library reports a mistyped
    // source name or an unknown material type as a plain std::runtime_error.
    // Those would otherwise reach the generic handler and request a cutback —
    // against a fault no increment size can fix, rebuilt and re-failed on every
    // call from every thread. Translating at the boundary is the only place
    // that covers code this layer does not own.
    try {
      thread_state ts;
      ts.ctx = std::make_unique<context_type>();
      m.build(*ts.ctx, props);
      ts.nprops = props.size();
      if (!ts.ctx->is_finalized())
        throw fatal_error(
            "the builder returned without calling finalize() on the context");

      // Both evaluators share the thread's context. They hold no mutable state
      // beyond diagnostics, and only one is used per call, so this is safe —
      // and it lets one material serve solid and plane-stress elements in one
      // job.
      ts.solid = std::make_unique<evaluator_type>(*ts.ctx, m.cfg);
      ts.ps = std::make_unique<ps_evaluator_type>(*ts.ctx, m.cfg, m.ps_opts);

      return cache.emplace(std::string(key), std::move(ts)).first->second;
    } catch (const fatal_error& e) {
      throw fatal_error("numsim UMAT: building the material graph for '" +
                        std::string(key) + "' failed: " + e.what());
    } catch (const std::exception& e) {
      throw fatal_error("numsim UMAT: building the material graph for '" +
                        std::string(key) + "' failed: " + e.what());
    }
  }

  mutable std::mutex m_mutex;
  std::unordered_map<std::string, model, transparent_string_hash,
                     std::equal_to<>>
      m_models;
};

/// Zero the host's outputs.
///
/// Used on EVERY error path, not just the fatal one. On a cutback it matters
/// more than it looks: PNEWDT is the minimum over all calls for the iteration,
/// so Abaqus finishes the element loop and assembles BEFORE acting on the
/// request — with whatever DDSDDE contains. Leaving it untouched means
/// assembling uninitialised memory, which can trap on a signalling NaN long
/// before the cutback is honoured.
///
/// A zero tangent is a poor stiffness, and that is the accepted trade: the
/// increment is being discarded either way, and predictably soft beats
/// arbitrarily wrong. STRESS is zeroed for the same reason.
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
/// Check that the deck supplied at least @p needed constants.
///
/// A *USER MATERIAL, CONSTANTS= count that disagrees with what the model reads
/// is a setup fault, not something a smaller increment fixes — and indexing past
/// the end would otherwise yield a plausible-looking value for a parameter that
/// was never given.
inline void require_props(std::span<const double> props, std::size_t needed,
                          const char* model) {
  if (props.size() < needed)
    throw fatal_error(std::string("numsim UMAT: model '") + model +
                      "' needs " + std::to_string(needed) +
                      " material constants but the deck supplied " +
                      std::to_string(props.size()) +
                      " — check *USER MATERIAL, CONSTANTS=");
}

/// Report an unrecoverable fault, then hand control to the fatal handler.
///
/// The message is assembled inside a nested try. An exception thrown from
/// within a catch handler is NOT caught by the sibling catch(...) of the same
/// try-block, so a bad_alloc while concatenating would escape a noexcept
/// function and terminate BEFORE the handler runs — losing precisely the
/// diagnostic the fatal path exists to produce. The fallback is a literal.
inline void report_fatal(const char* material, const char* what) noexcept {
  try {
    const std::string msg = std::string("Unrecoverable fault in material '") +
                            material + "' — terminating the analysis.\n  " +
                            what;
    invoke_fatal(msg.c_str());
  } catch (...) {
    invoke_fatal(
        "numsim UMAT: unrecoverable fault; the message could not be formatted "
        "— terminating the analysis");
  }
}

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
  const T* props{nullptr};
  T dtime{0};
  T* pnewdt{nullptr};
  const char* cmname{nullptr};
  std::size_t cmname_len{0};
  int ndi{0};
  int nshr{0};
  int ntens{0};
  int nstatv{0};
  int nprops{0};
};

template <typename Traits>
void umat_dispatch(const dispatch_args<typename Traits::value_type>& a) noexcept {
  using T = typename Traits::value_type;
  // A fixed buffer, so the success path allocates nothing at all: this runs per
  // integration point per global iteration. It is also NUL-terminated, so the
  // same storage feeds both the registry lookup and the C formatting below —
  // and, being stack storage, it cannot throw in a noexcept function.
  char namebuf[umat_registry<Traits>::max_cmname + 1] = {'\0'};
  std::string_view name;
  try {
    if (!a.cmname)
      throw fatal_error("numsim UMAT: CMNAME is null");
    if (!a.time)
      throw fatal_error("numsim UMAT: TIME is null");
    name = normalise_cmname(a.cmname, a.cmname_len, namebuf);
    const auto ec = case_from_element(a.ndi, a.nshr);
    const auto n = ntens_for(ec);

    if (static_cast<std::size_t>(a.ntens) != n)
      throw fatal_error("numsim UMAT: NTENS=" + std::to_string(a.ntens) +
                        " is inconsistent with NDI/NSHR for material '" +
                        std::string(name) + "'");

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

    // The deck's material constants. Only read when this thread builds the
    // graph for this material — never on the per-call path.
    const std::span<const double> props =
        (a.props && a.nprops > 0)
            ? std::span<const double>(a.props,
                                      static_cast<std::size_t>(a.nprops))
            : std::span<const double>{};

    umat_registry<Traits>::instance().evaluate(name, props, c);
  } catch (const fatal_error& e) {
    zero_outputs(a.stress, a.ddsdde, static_cast<std::size_t>(a.ntens));
    report_fatal(namebuf, e.what());
  } catch (const std::exception& e) {
    zero_outputs(a.stress, a.ddsdde, static_cast<std::size_t>(a.ntens));
    std::fprintf(stderr,
                 "numsim UMAT: material '%s' failed (%s) — requesting a "
                 "smaller increment\n",
                 namebuf, e.what());
    if (a.pnewdt) *a.pnewdt = T{0.25};
  } catch (...) {
    zero_outputs(a.stress, a.ddsdde, static_cast<std::size_t>(a.ntens));
    std::fprintf(stderr,
                 "numsim UMAT: material '%s' failed with an unknown error — "
                 "requesting a smaller increment\n",
                 namebuf);
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
/// Forwarded: STRESS, STATEV, DDSDDE, SSE, SPD, SCD, STRAN, DSTRAN, TIME,
/// DTIME, CMNAME, NDI, NSHR, NTENS, NSTATV, PROPS, NPROPS, DROT, PNEWDT.
///
/// PROPS reaches the registered builder, so *USER MATERIAL, CONSTANTS= drives
/// the model's parameters. It is read ONCE, when this thread first builds the
/// graph for this material name — constants cannot vary per increment, and a
/// changed NPROPS for the same name is reported as a setup fault.
///
/// Accepted and IGNORED — the full list, because an omission here reads as a
/// capability that is not there:
///
///   TEMP, DTEMP    Temperature-dependent properties are unavailable: constants
///                  are read once, when the graph is built. Thermal EXPANSION
///                  still behaves correctly, because with *EXPANSION in the same
///                  material definition Abaqus passes STRAN and DSTRAN already
///                  reduced to mechanical strain.
///   RPL, DDSDDT,   Required in fully coupled thermal-stress, coupled
///   DRPLDE, DRPLDT thermal-electrical-structural, and adiabatic analysis or
///                  with *INELASTIC HEAT FRACTION. Ignoring them there reports
///                  zero heat generation and degrades the coupled Jacobian; it
///                  is safe in every other procedure.
///   PREDEF, DPRED  Field variables.
///   DFGRD0/DFGRD1  Safe for this small-strain formulation. Under NLGEOM=YES
///                  Abaqus documents the stress measure as Cauchy and STRAN as
///                  an approximation to logarithmic strain, already rotated by
///                  DROT — which this layer handles. The residual caveat is
///                  that DDSDDE should then be d(dCauchy)/d(d log-strain); a
///                  small-strain tangent costs convergence rate, not converged
///                  accuracy.
///   COORDS         Position-dependent properties.
///   CELENT         Mesh-regularised softening/damage needs it.
///   NOEL, NPT,     Identify the point; only needed for diagnostics or
///   LAYER, KSPT    point-dependent behaviour.
///   JSTEP, KINC    JSTEP(4) flags a linear-perturbation step. Ignoring it means
///                  a perturbation step gets the full nonlinear update and a
///                  STATEV write; Abaqus restores the base state afterwards, so
///                  this is not corruption, but the returned Jacobian is the
///                  elastic-plastic tangent rather than the base-state elastic
///                  one.
///
/// Deck-side requirement not visible from this code: a UMAT used for beams or
/// shells that compute transverse shear energy must have the transverse shear
/// stiffness given in the beam or shell section definition.
#define NUMSIM_MATERIALS_DEFINE_UMAT(TRAITS)                                   \
  extern "C" void umat_(                                                       \
      double* STRESS, double* STATEV, double* DDSDDE, double* SSE,             \
      double* SPD, double* SCD, double* /*RPL*/, double* /*DDSDDT*/,           \
      double* /*DRPLDE*/, double* /*DRPLDT*/, const double* STRAN,             \
      const double* DSTRAN, const double* TIME, const double* DTIME,           \
      const double* /*TEMP*/, const double* /*DTEMP*/,                         \
      const double* /*PREDEF*/, const double* /*DPRED*/, const char* CMNAME,   \
      const int* NDI, const int* NSHR, const int* NTENS, const int* NSTATV,    \
      const double* PROPS, const int* NPROPS,                                   \
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
    a.props = PROPS;    a.nprops = *NPROPS;                                    \
    a.ndi = *NDI;  a.nshr = *NSHR;  a.ntens = *NTENS;  a.nstatv = *NSTATV;     \
    ::numsim::materials::umat::umat_dispatch<TRAITS>(a);                       \
  }

#endif  // UMAT_INTERFACE_H
