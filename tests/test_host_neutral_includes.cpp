// Only the host-neutral headers: nothing here may pull in umat/ (#212).
#include <gtest/gtest.h>
#include "numsim-materials/core/errors.h"
#include "numsim-materials/driver/external_state_source.h"

#ifdef UMAT_ERRORS_H
#error "core/errors.h and driver/external_state_source.h must not include umat/errors.h"
#endif
#ifndef NUMSIM_MATERIALS_CORE_ERRORS_H
#error "core/errors.h guard must be NUMSIM_MATERIALS_CORE_ERRORS_H"
#endif
#ifndef NUMSIM_MATERIALS_DRIVER_EXTERNAL_STATE_SOURCE_H
#error "driver/external_state_source.h guard must be NUMSIM_MATERIALS_DRIVER_EXTERNAL_STATE_SOURCE_H"
#endif

TEST(HostNeutralIncludes, HeadersCompileWithoutUmat) {
  EXPECT_THROW(throw numsim::materials::fatal_error("x"), std::runtime_error);
}
