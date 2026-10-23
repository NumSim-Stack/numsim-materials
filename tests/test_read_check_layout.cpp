#undef NDEBUG
#include <cstddef>
#include <gtest/gtest.h>
#include "numsim-materials/core/material_context.h"

// The debug checks change no class layout, and CMake gives every translation unit one setting, NDEBUG or not:
// release and debug translation units can share one context (#180).

std::size_t release_context_size();
int release_debug_checks();

namespace {

TEST(ReadCheckLayout, ReleaseAndDebugContextsHaveOneLayout) {
  EXPECT_EQ(sizeof(numsim::materials::material_context<numsim::materials::material_policy_default>), release_context_size());
}

TEST(ReadCheckLayout, NDEBUGLeavesTheSettingAlone) { EXPECT_EQ(NUMSIM_MATERIALS_DEBUG_CHECKS, release_debug_checks()); }

} // namespace
