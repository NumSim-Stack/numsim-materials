// The forwarding header keeps its old transitive include of tensor_conversion.h (#212).
#include <gtest/gtest.h>
#include "numsim-materials/umat/external_state_source.h"

#ifndef TENSOR_CONVERSION_H
#error "umat/external_state_source.h must still include umat/tensor_conversion.h"
#endif

TEST(UmatForwarder, OldIncludeStillBuilds) {
  using source = numsim::materials::external_strain_source<numsim::materials::material_policy_default>;
  EXPECT_TRUE(std::is_class_v<source>);
}
