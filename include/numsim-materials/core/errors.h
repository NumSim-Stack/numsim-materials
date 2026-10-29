#ifndef NUMSIM_MATERIALS_CORE_ERRORS_H
#define NUMSIM_MATERIALS_CORE_ERRORS_H

#include <stdexcept>

/// Failure classification for every host (UMAT, the report loop, the FFT solver).
///
/// A host has exactly two useful responses to a failure, and they are opposites:
///
///   * retry with a smaller increment  (cutback_error)
///   * stop, because retrying cannot help (fatal_error)
///
/// Choosing wrongly is worse than not choosing at all. A fatal fault treated as
/// recoverable makes the host halve the increment until it dies on the minimum
/// step (Abaqus: the minimum time increment), burying the real cause. A
/// recoverable failure treated as fatal throws away an analysis that would have
/// converged.
///
/// Exceptions escaping the constitutive models themselves (a return-mapping
/// Newton that did not converge, say) are retry-worthy, so a host treats an
/// unclassified exception as a cutback. Everything that a smaller increment
/// cannot fix -- a missing material, an undersized history buffer, a property
/// that does not exist -- must therefore say so explicitly by throwing
/// fatal_error.
namespace numsim::materials {

/// A setup fault. No cutback can fix it.
class fatal_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// A numerical failure at this increment size. Retry smaller.
class cutback_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_ERRORS_H
