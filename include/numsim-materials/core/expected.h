#ifndef NUMSIM_MATERIALS_CORE_EXPECTED_H
#define NUMSIM_MATERIALS_CORE_EXPECTED_H

// numsim::materials::expected / unexpected are std::expected / std::unexpected.
// std::expected is part of the NumSim toolchain baseline: GCC >= 13, or
// Clang >= 19 (libstdc++ hides <expected> from Clang 18, whose __cpp_concepts
// is still 201907). Single include point, so a different implementation
// would be a local change.

#include <version>

#if !defined(__cpp_lib_expected) || __cpp_lib_expected < 202202L
#error "numsim-materials needs std::expected (C++23): use GCC >= 13 or Clang >= 19"
#endif

#include <expected>

namespace numsim::materials {
using std::expected;
using std::unexpected;
}  // namespace numsim::materials

#endif  // NUMSIM_MATERIALS_CORE_EXPECTED_H
