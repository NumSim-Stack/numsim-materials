/// A registered expansion turns one deck entry into plain entries before anything is created.
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/default_materials.h"
#include "numsim-materials/io/json_material_factory.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using json = nlohmann::json;

/// "<name>_a" = 1 and the head "<name>" = 2; "bad_constants" leaves out a value, "headless" the head.
struct Registration {
  Registration() {
    nm::register_default_materials<policy>();
    auto &expansions{nm::json_expansion_registry<json>::instance()};
    expansions.add("two_constants", [](json const &e) {
      auto const name{e.at("name").get<std::string>()};
      return std::vector<json>{{{"type", "constant_scalar"}, {"name", name + "_a"}, {"value", 1.0}},
                               {{"type", "constant_scalar"}, {"name", name}, {"value", 2.0}}};
    });
    expansions.add("bad_constants", [](json const &e) {
      auto const name{e.at("name").get<std::string>()};
      return std::vector<json>{{{"type", "constant_scalar"}, {"name", name + "_a"}},
                               {{"type", "constant_scalar"}, {"name", name}, {"value", 2.0}}};
    });
    expansions.add("headless", [](json const &) {
      return std::vector<json>{{{"type", "constant_scalar"}, {"name", "other"}, {"value", 1.0}}};
    });
  }
};
const Registration registration_{};

std::string error_from(json const &entry) {
  ctx_type ctx;
  try {
    nm::create_from_json(ctx, entry);
    return "";
  } catch (std::exception const &e) {
    return e.what();
  }
}

TEST(JsonExpansions, CreatesTheEntriesAndReturnsTheHead) {
  ctx_type ctx;
  auto &head{nm::create_from_json(ctx, json{{"type", "two_constants"}, {"name", "k"}})};
  EXPECT_EQ(head.name(), "k");
  ctx.finalize();
  EXPECT_EQ(ctx.get<T>("k_a", "value"), 1.0);
  EXPECT_EQ(ctx.get<T>("k", "value"), 2.0);
}

TEST(JsonExpansions, TheHeadCarriesTheNameOfTheEntry) {
  auto const msg{error_from({{"type", "headless"}, {"name", "k"}})};
  EXPECT_NE(msg.find("headless"), std::string::npos) << msg;
  EXPECT_NE(msg.find("\"k\""), std::string::npos) << msg;
}

TEST(JsonExpansions, AnErrorInAGeneratedEntryNamesTheExpansion) {
  auto const msg{error_from({{"type", "bad_constants"}, {"name", "k"}})};
  EXPECT_NE(msg.find("k_a"), std::string::npos) << msg;
  EXPECT_NE(msg.find("bad_constants \"k\""), std::string::npos) << msg;
}

TEST(JsonExpansions, AnUnknownTypeIsComparedWithTheExpansionsToo) {
  auto const msg{error_from({{"type", "two_constant"}, {"name", "k"}})};
  EXPECT_NE(msg.find("Did you mean 'two_constants'?"), std::string::npos) << msg;
}

TEST(JsonExpansions, SubstituteReplacesThePlaceholderInEveryStringValue) {
  json const deck = {{"name", "f_{q}"}, {"{q}", 1},
                     {"list", {"a_{q}", 2.5, json{{"deep", "{q}_{q}"}}}}};
  json const out = nm::substitute(deck, "{q}", "7");
  EXPECT_EQ(out.at("name"), "f_7");
  EXPECT_TRUE(out.contains("{q}")); // keys are not values
  EXPECT_EQ(out.at("list")[0], "a_7");
  EXPECT_EQ(out.at("list")[1], 2.5);
  EXPECT_EQ(out.at("list")[2].at("deep"), "7_7");
}

} // namespace
