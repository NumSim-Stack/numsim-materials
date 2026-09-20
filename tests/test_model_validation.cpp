/// Tests for the checks that reject a malformed model instead of running it.
///
/// Each of these used to be silent: a duplicate material name aliased one
/// material's properties onto another (and corrupted the heap when their types
/// differed), an out-of-range "indices" subscripted a tensor past its end, an
/// update_property() typo looked like a property that never changes, and
/// commit()/revert() on an unfinalized context did nothing at all.
#include <gtest/gtest.h>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/scalar_stepper.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/default_materials.h"

namespace {

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param_type = policy::ParameterHandler;

/// Aliased because the comma in the template argument list would otherwise be
/// read as a macro argument separator by EXPECT_THROW.
using stepper2 = numsim::materials::tensor_component_stepper<2, policy>;

/// A stepper named @p name producing "strain" at component [0,0].
void add_stepper(ctx_type& ctx, param_type& p, const std::string& name) {
  p.clear();
  p.insert<std::string>("name", name);
  p.insert<T>("increment", T{0.1});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<stepper2>(p);
}

// --- Duplicate material names (#56) ---

TEST(ModelValidation, ASecondMaterialWithALiveNameIsRejected) {
  ctx_type ctx;
  param_type p;
  add_stepper(ctx, p, "stepper");

  p.clear();
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.5});
  p.insert<std::vector<std::size_t>>("indices", {1, 1});

  EXPECT_THROW(
      ctx.create<stepper2>(p),
      std::invalid_argument);
}

/// The dangerous case: the two materials publish properties of DIFFERENT
/// types under the same name. Lookups resolve through a static_cast, so
/// before the check this wired a tensor consumer to a double and wrote past
/// the end of it -- a heap-buffer-overflow under ASan, silent without it.
TEST(ModelValidation, ADuplicateNameIsRejectedEvenWhenTheTypesDiffer) {
  ctx_type ctx;
  param_type p;
  add_stepper(ctx, p, "collide");

  p.clear();
  p.insert<std::string>("name", "collide");
  p.insert<T>("increment", T{0.1});

  EXPECT_THROW(ctx.create<numsim::materials::scalar_stepper<policy>>(p),
               std::invalid_argument);
}

TEST(ModelValidation, TheDuplicateNameErrorNamesTheMaterial) {
  ctx_type ctx;
  param_type p;
  add_stepper(ctx, p, "the_clashing_name");

  p.clear();
  p.insert<std::string>("name", "the_clashing_name");
  p.insert<T>("increment", T{0.1});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});

  try {
    ctx.create<stepper2>(p);
    FAIL() << "a duplicate material name was accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("the_clashing_name"),
              std::string::npos)
        << "the message does not name the offending material: " << e.what();
  }
}

/// The rejection must happen before the second material is constructed, or it
/// has already overwritten the first one's entries in the property registry.
/// The surviving material must be the one that was there first.
TEST(ModelValidation, TheFirstMaterialSurvivesARejectedDuplicate) {
  ctx_type ctx;
  param_type p;
  add_stepper(ctx, p, "stepper");

  p.clear();
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{99.0});  // would be obvious in the result
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  EXPECT_THROW(
      ctx.create<stepper2>(p),
      std::invalid_argument);

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", T{166.67});
  p.insert<T>("G", T{76.92});
  ctx.create<numsim::materials::linear_elasticity<policy>>(p);

  ctx.finalize();
  ctx.update();

  // One increment of 0.1 from the FIRST stepper, not 99.0 from the rejected one.
  const auto& strain = ctx.get<tmech::tensor<T, 3, 2>>("stepper", "strain");
  EXPECT_DOUBLE_EQ(strain(0, 0), 0.1);
}

// --- tensor_component_stepper "indices" (#57) ---

TEST(ModelValidation, AnOutOfRangeIndexIsRejected) {
  ctx_type ctx;
  param_type p;
  p.clear();
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.1});
  p.insert<std::vector<std::size_t>>("indices", {9, 9});

  EXPECT_THROW(
      ctx.create<stepper2>(p),
      std::invalid_argument);
}

/// Dim is the last valid index plus one -- the off-by-one a range check is
/// most likely to get wrong.
TEST(ModelValidation, TheIndexRangeCheckIsInclusiveOfTheLastComponent) {
  ctx_type ctx;
  param_type p;
  p.clear();
  p.insert<std::string>("name", "edge");
  p.insert<T>("increment", T{0.1});
  p.insert<std::vector<std::size_t>>("indices", {2, 2});  // valid for 3D
  EXPECT_NO_THROW(
      ctx.create<stepper2>(p));

  ctx_type ctx2;
  p.clear();
  p.insert<std::string>("name", "past_edge");
  p.insert<T>("increment", T{0.1});
  p.insert<std::vector<std::size_t>>("indices", {3, 0});  // one past
  EXPECT_THROW(
      ctx2.create<stepper2>(p),
      std::invalid_argument);
}

/// Too few entries read m_indices[1] past the end of the vector itself, which
/// is a different overflow from the out-of-range one above.
TEST(ModelValidation, AnIndexListOfTheWrongLengthIsRejected) {
  ctx_type ctx;
  param_type p;
  p.clear();
  p.insert<std::string>("name", "short");
  p.insert<T>("increment", T{0.1});
  p.insert<std::vector<std::size_t>>("indices", {0});  // rank 2 needs 2

  EXPECT_THROW(
      ctx.create<stepper2>(p),
      std::invalid_argument);

  ctx_type ctx2;
  p.clear();
  p.insert<std::string>("name", "long");
  p.insert<T>("increment", T{0.1});
  p.insert<std::vector<std::size_t>>("indices", {0, 0, 0});
  EXPECT_THROW(
      ctx2.create<stepper2>(p),
      std::invalid_argument);
}

TEST(ModelValidation, TheIndexErrorReportsTheOffendingValueAndTheRange) {
  ctx_type ctx;
  param_type p;
  p.clear();
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.1});
  p.insert<std::vector<std::size_t>>("indices", {0, 7});

  try {
    ctx.create<stepper2>(p);
    FAIL() << "an out-of-range index was accepted";
  } catch (const std::invalid_argument& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find('7'), std::string::npos)
        << "the message does not report the offending value: " << msg;
    EXPECT_NE(msg.find("0..2"), std::string::npos)
        << "the message does not report the valid range: " << msg;
  }
}

// --- update_property with a name that is not in the graph (#69) ---

TEST(ModelValidation, UpdatePropertyWithAnUnknownNameThrows) {
  ctx_type ctx;
  param_type p;
  add_stepper(ctx, p, "stepper");
  ctx.finalize();

  EXPECT_THROW(ctx.update_property("stepper", "starin"), std::runtime_error);
  EXPECT_THROW(ctx.update_property("steppr", "strain"), std::runtime_error);
  EXPECT_NO_THROW(ctx.update_property("stepper", "strain"));
}

TEST(ModelValidation, TheUnknownPropertyErrorNamesBothHalves) {
  ctx_type ctx;
  param_type p;
  add_stepper(ctx, p, "stepper");
  ctx.finalize();

  try {
    ctx.update_property("stepper", "starin");
    FAIL() << "a property that is not in the graph was accepted";
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("stepper"), std::string::npos) << msg;
    EXPECT_NE(msg.find("starin"), std::string::npos) << msg;
  }
}

// --- commit()/revert() before finalize() (#60) ---

TEST(ModelValidation, CommitAndRevertRequireAFinalizedContext) {
  ctx_type ctx;
  param_type p;
  add_stepper(ctx, p, "stepper");

  // The engine's history list is built by finalize(), so before it these
  // walked an empty list and reported success.
  EXPECT_THROW(ctx.commit(), std::logic_error);
  EXPECT_THROW(ctx.revert(), std::logic_error);

  ctx.finalize();
  EXPECT_NO_THROW(ctx.commit());
  EXPECT_NO_THROW(ctx.revert());
}

} // namespace
