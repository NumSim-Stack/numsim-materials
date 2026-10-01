#ifndef NUMSIM_MATERIALS_CORE_EXPECTED_H
#define NUMSIM_MATERIALS_CORE_EXPECTED_H

// numsim::materials::expected / unexpected: std::expected when the standard
// library of the *consuming* compiler provides it, otherwise tl::expected.
// libstdc++ 13 hides std::expected from Clang 18 (it reports
// __cpp_concepts < 202002), and the NumSim stack's baseline is GCC 13 /
// Clang 18. Detected here rather than at configure time, because a
// header-only library may be built with one compiler and used with another.

#include <version>

#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
#include <expected>
namespace numsim::materials {
using std::expected;
using std::unexpected;
}  // namespace numsim::materials
#else
#include <tl/expected.hpp>
namespace numsim::materials {
template <typename T, typename E>
using expected = tl::expected<T, E>;
using tl::unexpected;
}  // namespace numsim::materials
#endif

#endif  // NUMSIM_MATERIALS_CORE_EXPECTED_H
