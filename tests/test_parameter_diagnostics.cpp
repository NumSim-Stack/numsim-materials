/// Errors about a deck name what is wrong and where: the parameter, its type, the material and the entry.
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

// a literal and a stored key both go through the one get_parameter
TEST(ParameterDiagnostics, LiteralAndStoredKeysReportTheSameWay) {
  param_type p;
  p.insert<std::string>("name", "probe");
  p.insert<int>("K", 166);

  struct probe : nm::material_interface<policy> {
    using nm::material_interface<policy>::material_interface;
    void read_rvalue() { (void)this->get_parameter<T>("K"); }
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

TEST(ParameterDiagnostics, AWrongParameterTypeNamesTheTypeRead) {
  param_type p;
  p.insert<std::string>("name", "probe");
  p.insert<int>("K", 166);
  struct probe : nm::material_interface<policy> {
    using nm::material_interface<policy>::material_interface;
    void read() { (void)this->get_parameter<T>("K"); }
  };
  policy::PropertyHandler props;
  probe m{p, props};
  try {
    m.read();
    FAIL() << "an int was accepted where a double is read";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string{e.what()}.find("a real number"), std::string::npos) << e.what();
  }
}

// the handler's "Key K not found" named neither the material nor what was being read
TEST(ParameterDiagnostics, AMissingParameterNamesTheParameterAndMaterial) {
  param_type p;
  p.insert<std::string>("name", "probe");
  struct probe : nm::material_interface<policy> {
    using nm::material_interface<policy>::material_interface;
    void read() { (void)this->get_parameter<T>("absent"); }
  };
  policy::PropertyHandler props;
  probe m{p, props};
  try {
    m.read();
    FAIL() << "a missing parameter was read";
  } catch (const std::invalid_argument& e) {
    const std::string msg{e.what()};
    EXPECT_NE(msg.find("'absent'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("probe"), std::string::npos) << msg;
    EXPECT_NE(msg.find("missing"), std::string::npos) << msg;
  }
}

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

// weighted_sum's terms and vector_newton's zero_blocks are one C++ type and used to share one message
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
  EXPECT_NE(msg.find("\"mix\""), std::string::npos) << "a reader failure does not name the entry: " << msg;
}

TEST(ParameterDiagnostics, AMissingRequiredParameterNamesTheEntry) {
  const json entry = {{"type", "linear_elasticity"}, {"name", "el"}, {"strain_producer_name", "s"}, {"K", 1.0}}; // no G
  const auto msg = error_from(entry);
  ASSERT_FALSE(msg.empty()) << "a material without a required parameter was created";
  EXPECT_NE(msg.find("required"), std::string::npos) << msg; // the names of the parameters go to the log, not what()
  EXPECT_NE(msg.find("\"el\""), std::string::npos) << "the material's own check does not name the entry: " << msg;
}

TEST(ParameterDiagnostics, ANonStringNameIsReportedAsSuch) {
  const json entry = {{"type", "linear_elasticity"}, {"name", 3}, {"strain_producer_name", "s"}, {"K", 1.0}, {"G", 1.0}};
  const auto msg = error_from(entry);
  ASSERT_FALSE(msg.empty()) << "a numeric name was accepted";
  EXPECT_NE(msg.find("'name'"), std::string::npos) << msg;
  EXPECT_NE(msg.find("not a string"), std::string::npos) << msg;
}

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
  EXPECT_EQ(msg.find("key 'type' not found"), std::string::npos) // nlohmann's own message
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
