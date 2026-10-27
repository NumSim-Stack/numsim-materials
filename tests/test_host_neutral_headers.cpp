// Host-neutral headers build without umat/ and keep the umat names working (#212).
#include <type_traits>
#include <gtest/gtest.h>
#include "numsim-materials/core/errors.h"
#include "numsim-materials/core/traits.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/umat/errors.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using strain_source = nm::external_strain_source<policy>;

static_assert(std::is_same_v<nm::umat::fatal_error, nm::fatal_error>);
static_assert(std::is_same_v<nm::umat::cutback_error, nm::cutback_error>);
// raw host buffers are converted by the host adapter, not by the source
template <typename Source>
concept binds_raw_buffers = requires(Source &s, T const *p) { s.bind(p, p); };
static_assert(!binds_raw_buffers<strain_source>);

TEST(HostNeutralHeaders, ErrorsAreRuntimeErrors) {
  EXPECT_THROW(throw nm::cutback_error("x"), std::runtime_error);
  EXPECT_THROW(throw nm::fatal_error("x"), nm::umat::fatal_error);
}

} // namespace
