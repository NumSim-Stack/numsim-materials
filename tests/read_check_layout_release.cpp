#ifndef NDEBUG
#define NDEBUG
#endif
#include <cstddef>
#include "numsim-materials/core/material_context.h"

// Compiled as a release translation unit, linked with debug ones (test_read_check_layout.cpp).

std::size_t release_context_size();
int release_debug_checks();
int release_debug_checks() { return NUMSIM_MATERIALS_DEBUG_CHECKS; }
std::size_t release_context_size() { return sizeof(numsim::materials::material_context<numsim::materials::material_policy_default>); }
