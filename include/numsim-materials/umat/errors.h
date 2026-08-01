#ifndef NUMSIM_MATERIALS_UMAT_ERRORS_H
#define NUMSIM_MATERIALS_UMAT_ERRORS_H

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

/// Failure classification for the UMAT layer.
///
/// A UMAT has exactly two useful responses to a failure, and they are opposites:
///
///   * ask the host for a smaller increment and retry  (cutback_error)
///   * stop the analysis, because retrying cannot help (fatal_error)
///
/// Choosing wrongly is worse than not choosing at all. A fatal fault treated as
/// recoverable makes Abaqus halve the increment until it dies on the minimum
/// time step, burying the real cause. A recoverable failure treated as fatal
/// throws away an analysis that would have converged.
///
/// The default for an UNCLASSIFIED exception is cutback, because exceptions
/// escaping the constitutive models themselves (a return-mapping Newton that
/// did not converge, say) are genuinely retry-worthy. Everything that a smaller
/// increment cannot fix — a missing material, an undersized STATEV, a property
/// that does not exist — must therefore say so explicitly by throwing
/// fatal_error.
namespace numsim::materials::umat {

/// A setup fault. No timestep cutback can fix it.
class fatal_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// A numerical failure at this increment size. Retry smaller.
class cutback_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

}  // namespace numsim::materials::umat

// Abaqus's analysis-termination routine. Declared weak so this library also
// links standalone (unit tests, host-free drivers), where the symbol is absent
// and the pointer compares null.
#if defined(__GNUC__) || defined(__clang__)
extern "C" __attribute__((weak)) void xit_(void);
#define NUMSIM_MATERIALS_HAVE_WEAK_XIT 1
#endif

namespace numsim::materials::umat {

/// Called when an unrecoverable fault is detected.
///
/// In production this MUST NOT return — the analysis is over. It is a hook only
/// so tests can observe the fault without terminating the test runner; a
/// handler that returns leaves the host holding zeroed outputs.
using fatal_handler = void (*)(const char* message);

/// Terminate the analysis through Abaqus, falling back to abort() when the host
/// is not present.
inline void abaqus_xit_handler(const char* message) {
  std::fprintf(stderr, "%s\n", message);
  std::fflush(stderr);
#ifdef NUMSIM_MATERIALS_HAVE_WEAK_XIT
  if (xit_) xit_();  // terminates the analysis; does not return
#endif
  // Reached only without a host. Aborting is deliberate: returning would hand
  // the caller a silently wrong material response, which is the failure mode
  // this whole classification exists to prevent.
  std::abort();
}

/// Atomic because the fatal path is reachable from every worker thread. The
/// contract is still "set once before any call" — the atomicity costs nothing
/// here and removes a footgun rather than enabling a use case.
inline std::atomic<fatal_handler>& fatal_handler_slot() {
  static std::atomic<fatal_handler> handler{&abaqus_xit_handler};
  return handler;
}

/// Replace the fatal handler. Intended for tests; set once before any call.
inline void set_fatal_handler(fatal_handler handler) {
  fatal_handler_slot().store(handler ? handler : &abaqus_xit_handler,
                             std::memory_order_relaxed);
}

inline void invoke_fatal(const char* message) noexcept {
  fatal_handler_slot().load(std::memory_order_relaxed)(message);
}

}  // namespace numsim::materials::umat

#endif  // NUMSIM_MATERIALS_UMAT_ERRORS_H
