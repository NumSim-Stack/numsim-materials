/// Errors about a deck must name what is wrong and where.
///
/// Three failures used to name nothing useful: a wrong parameter type threw a
/// bare std::bad_any_cast; a malformed weighted_sum term reported an error
/// about vector_newton's "zero_blocks", because both parameters are the same
/// C++ type and readers are found by type; and a missing or misspelt "type"
/// leaked nlohmann's own exception or echoed the name back with no candidates.
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/default_materials.h"
#include "numsim-materials/io/json_material_factory.h"
#include "numsim-materials/materials/linear_elasticity.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using json = nlohmann::json;

struct Registration {
  Registration() { nm::register_default_materials<policy>(); }
};
const Registration registration_{};

/// The message produced by building @p entry, or "" if it built.
std::string error_from(const json& entry) {
  ctx_type ctx;
  try {
    nm::create_from_json(ctx, entry);
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

// --- A wrong parameter type names the parameter and the material (#65) ---

TEST(ParameterDiagnostics, AWrongParameterTypeNamesTheParameterAndMaterial) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<int>("K", 166);  // the material reads a double
  p.insert<T>("G", T{76.92});

  try {
    ctx.create<nm::linear_elasticity<policy>>(p);
    FAIL() << "an int was accepted where a double is read";
  } catch (const std::invalid_argument& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("'K'"), std::string::npos)
        << "does not name the parameter: " << msg;
    EXPECT_NE(msg.find("elastic"), std::string::npos)
        << "does not name the material: " << msg;
  }
}

/// get_parameter is overloaded on how the key arrives, and materials use BOTH:
/// a string literal binds to the rvalue overload, a stored std::string to the
/// const-reference one. A test that reached only one left the other's guard
/// free to be deleted with the suite still green -- which is what happened.
TEST(ParameterDiagnostics, BothGetParameterOverloadsReportTheSameWay) {
  param_type p;
  p.insert<std::string>("name", "probe");
  p.insert<int>("K", 166);

  struct probe : nm::material_interface<policy> {
    using nm::material_interface<policy>::material_interface;
    // "K" as a temporary -> std::string&&
    void read_rvalue() { (void)this->get_parameter<T>("K"); }
    // a named string -> const std::string&
    void read_lvalue() {
      const std::string key = "K";
      (void)this->get_parameter<T>(key);
    }
  };

  policy::PropertyHandler props;
  probe m{p, props};

  EXPECT_THROW(m.read_rvalue(), std::invalid_argument);
  EXPECT_THROW(m.read_lvalue(), std::invalid_argument);
}

/// It must be an exception type a caller would think to catch, not
/// std::bad_any_cast, which is neither invalid_argument nor runtime_error.
TEST(ParameterDiagnostics, AWrongParameterTypeIsNotABareBadAnyCast) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<int>("K", 166);
  p.insert<T>("G", T{76.92});

  EXPECT_THROW(ctx.create<nm::linear_elasticity<policy>>(p),
               std::invalid_argument);
  try {
    ctx_type ctx2;
    ctx2.create<nm::linear_elasticity<policy>>(p);
  } catch (const std::bad_any_cast&) {
    FAIL() << "still surfacing std::bad_any_cast";
  } catch (const std::invalid_argument&) {
    SUCCEED();
  }
}

// --- Same C++ type, different parameters (#66) ---

/// block_ref is std::pair<std::string, std::string>, and weighted_sum's
/// terms_type is a vector of exactly that, so both resolve to one reader.
/// A malformed term used to be reported as a "zero_blocks" error.
TEST(ParameterDiagnostics, AMalformedWeightedSumTermDoesNotMentionZeroBlocks) {
  const json entry = {{"type", "weighted_sum"},
                      {"name", "mix"},
                      // three entries where a pair is required
                      {"terms", {{"w1", "mat1", "extra"}}}};

  const auto msg = error_from(entry);
  ASSERT_FALSE(msg.empty()) << "a malformed term was accepted";
  EXPECT_EQ(msg.find("zero_blocks"), std::string::npos)
      << "a weighted_sum error still names vector_newton's parameter: " << msg;
  EXPECT_NE(msg.find("terms"), std::string::npos)
      << "the error does not name the parameter at fault: " << msg;
}

/// ...and the parameter that DOES own that wording keeps it.
TEST(ParameterDiagnostics, AMalformedZeroBlocksEntryStillSaysRowColumn) {
  const json entry = {{"type", "vector_newton"},
                      {"name", "solver"},
                      {"function", "model"},
                      {"unknowns",
                       {{{"name", "dgamma"}, {"kind", "scalar"}},
                        {{"name", "backstress"}, {"kind", "sym_tensor"}}}},
                      {"zero_blocks", {{"dgamma"}}}};

  const auto msg = error_from(entry);
  ASSERT_FALSE(msg.empty()) << "a malformed zero_blocks entry was accepted";
  EXPECT_NE(msg.find("zero_blocks"), std::string::npos) << msg;
  EXPECT_NE(msg.find("row"), std::string::npos)
      << "zero_blocks lost its own wording: " << msg;
}

// --- "type" errors (#67) ---

TEST(ParameterDiagnostics, AMissingTypeKeyDoesNotLeakTheJsonLibrary) {
  const json entry = {{"name", "nameless"}, {"K", 1.0}};

  const auto msg = error_from(entry);
  ASSERT_FALSE(msg.empty()) << "an entry with no \"type\" was accepted";
  EXPECT_NE(msg.find("type"), std::string::npos) << msg;
  EXPECT_NE(msg.find("nameless"), std::string::npos)
      << "the error does not say which entry: " << msg;
  // nlohmann's own message. Leaking it names a library the user never used.
  EXPECT_EQ(msg.find("key 'type' not found"), std::string::npos)
      << "still surfacing the nlohmann exception: " << msg;
}

TEST(ParameterDiagnostics, AMisspeltTypeSuggestsTheRealOne) {
  const json entry = {{"type", "j2_plasticty"},  // transposed letters
                      {"name", "j2"}};

  const auto msg = error_from(entry);
  ASSERT_FALSE(msg.empty()) << "an unknown type was accepted";
  EXPECT_NE(msg.find("j2_plasticity"), std::string::npos)
      << "no suggestion offered for a one-transposition typo: " << msg;
}

/// A name close to nothing registered gets the list instead of a bad guess.
TEST(ParameterDiagnostics, ATypeCloseToNothingGetsTheFullList) {
  const json entry = {{"type", "zzzzzzzzzzzzzzzz"}, {"name", "x"}};

  const auto msg = error_from(entry);
  ASSERT_FALSE(msg.empty());
  EXPECT_NE(msg.find("Registered types are:"), std::string::npos) << msg;
  EXPECT_NE(msg.find("linear_elasticity"), std::string::npos)
      << "the list does not actually list them: " << msg;
}

TEST(ParameterDiagnostics, TheUnknownTypeErrorNamesTheEntry) {
  const json entry = {{"type", "no_such_material_at_all"},
                      {"name", "the_bad_entry"}};

  const auto msg = error_from(entry);
  EXPECT_NE(msg.find("the_bad_entry"), std::string::npos)
      << "cannot tell which of twenty entries failed: " << msg;
}

/// The happy path must stay happy: these checks run before every creation.
TEST(ParameterDiagnostics, AWellFormedEntryStillBuilds) {
  const json entry = {{"type", "linear_elasticity"},
                      {"name", "elastic"},
                      {"strain_producer_name", "stepper"},
                      {"K", 166.67},
                      {"G", 76.92}};

  EXPECT_EQ(error_from(entry), "");
}

} // namespace
