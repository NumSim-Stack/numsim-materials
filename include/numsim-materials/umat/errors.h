#ifndef UMAT_ERRORS_H
#define UMAT_ERRORS_H

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "numsim-materials/core/errors.h"

/// The UMAT side of the failure classification in core/errors.h: the names under
/// umat::, and the fatal path that ends an Abaqus analysis.
namespace numsim::materials::umat {

using materials::cutback_error;
using materials::fatal_error;

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

#endif  // UMAT_ERRORS_H
