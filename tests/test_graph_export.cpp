// The engine keeps its typed edges; graph(), to_dot()/to_json() and the named cycle (#213, #188).
#include <algorithm>
#include <format>
#include <ranges>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "numsim-materials/core/graph_description.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "test_fixture_materials.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

/// "value", read from "<from>::value" over a global or local edge (no "from": a source).
template <typename Traits>
class node final : public nm::material_base<node<Traits>, Traits> {
public:
  using base = nm::material_base<node<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit node(Args &&...args) : base(std::forward<Args>(args)...), m_value(base::template add_output<T>("value")) {
    auto const &from{base::template get_parameter<std::string>("from")};
    if (from.empty()) {
      base::add_callback(&node::compute, m_value).reads();
      return;
    }
    auto const kind{base::template get_parameter<std::string>("edge") == "local" ? nm::EdgeKind::Local : nm::EdgeKind::Global};
    m_in = &base::template add_input<T>(from, "value", kind);
    base::add_callback(&node::compute, m_value).reads(*m_in);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<std::string>(para, "from", std::string{});
    nm::optional<std::string>(para, "edge", std::string{"global"});
    return para;
  }

  void compute() { m_value = m_in ? m_in->get() + T{1} : T{1}; }

private:
  T &m_value;
  nm::input_property<T, nm::property_traits> const *m_in{nullptr};
};

void add(ctx_type &ctx, std::string const &name, std::string const &from = {}, std::string const &edge = "global") {
  param_type p;
  p.insert<std::string>("name", name);
  p.insert<std::string>("from", from);
  p.insert<std::string>("edge", edge);
  ctx.create<node<policy>>(p);
}

/// Outputs named by parameters, so a test can give them awkward names; "self_order" makes it run after itself.
template <typename Traits>
class odd final : public nm::material_base<odd<Traits>, Traits> {
public:
  using base = nm::material_base<odd<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit odd(Args &&...args)
      : base(std::forward<Args>(args)...), m_value(base::template add_output<T>(base::template get_parameter<std::string>("prop"))) {
    base::add_callback(&odd::compute, m_value).reads();
    if (base::template get_parameter<bool>("self_order")) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations" // the only way to order a callback behind itself
      base::runs_after(m_value, m_value);
#pragma GCC diagnostic pop
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<std::string>(para, "prop", std::string{"value"});
    nm::optional<bool>(para, "self_order", false);
    return para;
  }

  void compute() { m_value = T{1}; }

private:
  T &m_value;
};

/// "first" and "second" from one callback: "second" is routed to its writer.
template <typename Traits>
class pair final : public nm::material_base<pair<Traits>, Traits> {
public:
  using base = nm::material_base<pair<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit pair(Args &&...args)
      : base(std::forward<Args>(args)...), m_first(base::template add_output<T>("first")), m_second(base::template add_output<T>("second")) {
    base::add_callback(&pair::compute, m_first, m_second).reads();
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() { m_first = m_second = T{1}; }

private:
  T &m_first;
  T &m_second;
};

/// Adds to its last value: never constant.
template <typename Traits>
class ticker final : public nm::material_base<ticker<Traits>, Traits> {
public:
  using base = nm::material_base<ticker<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit ticker(Args &&...args) : base(std::forward<Args>(args)...), m_value(base::template add_output<T>("value")) {
    base::add_callback(&ticker::compute, m_value).reads().stateful();
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() { m_value += T{1}; }

private:
  T &m_value;
};

template <template <typename> class Material>
void create(ctx_type &ctx, std::string const &name) {
  param_type p;
  p.insert<std::string>("name", name);
  ctx.create<Material<policy>>(p);
}

void add_odd(ctx_type &ctx, std::string const &name, std::string const &prop) {
  param_type p;
  p.insert<std::string>("name", name);
  p.insert<std::string>("prop", prop);
  ctx.create<odd<policy>>(p);
}

/// True when every quoted string closes, escapes only \\ \" \n and holds no raw newline.
bool dot_strings_are_valid(std::string const &dot) {
  bool in_string{false};
  for (std::size_t i{0}; i < dot.size(); ++i) {
    char const c{dot[i]};
    if (!in_string) {
      in_string = c == '"';
    } else if (c == '\\') {
      if (++i == dot.size() || std::string{"\\\"n"}.find(dot[i]) == std::string::npos) return false;
    } else if (c == '"') {
      in_string = false;
    } else if (c == '\n') {
      return false;
    }
  }
  return !in_string;
}

std::string key(nm::property_base const *p) { return p->traits().id.owner + "::" + p->traits().id.name; }

std::vector<std::string> edge_list(ctx_type const &ctx) {
  auto out{ctx.graph().edges | std::views::transform([](nm::graph_edge const &e) {
             return std::format("{} -> {} [{}]", key(e.from), key(e.to), nm::to_string(e.kind));
           }) |
           std::ranges::to<std::vector>()};
  std::ranges::sort(out);
  return out;
}

TEST(GraphExport, EdgesKeepTheirKinds) {
  ctx_type ctx;
  add(ctx, "a");
  add(ctx, "c", "b", "local");
  add(ctx, "b", "a");
  ctx.finalize();
  EXPECT_EQ(edge_list(ctx), (std::vector<std::string>{"a::value -> b::value [declared read]", "b::value -> c::value [local]"}));
}

TEST(GraphExport, LocalEdgesDoNotOrder) {
  ctx_type ctx;
  add(ctx, "a");
  add(ctx, "c", "b", "local");
  add(ctx, "b", "a");
  ctx.finalize();
  auto const order{ctx.property_execution_order() | std::views::transform(key) | std::ranges::to<std::vector>()};
  EXPECT_EQ(order, (std::vector<std::string>{"a::value", "c::value", "b::value"}));
}

TEST(GraphExport, NodesCarryTheirFlags) {
  ctx_type ctx;
  add(ctx, "a");
  param_type p;
  p.insert<std::string>("name", "counter");
  ctx.create<nm::test::int_counter<policy>>(p);
  ctx.finalize();
  auto const &nodes{ctx.graph().nodes};
  auto const find{[&](std::string const &k) {
    return *std::ranges::find_if(nodes, [&](nm::graph_node const &n) { return key(n.property) == k; });
  }};
  EXPECT_EQ(nodes.size(), 2u);
  EXPECT_TRUE(find("counter::count").history);
  EXPECT_TRUE(find("counter::count").callback);
  EXPECT_FALSE(find("a::value").history);
  EXPECT_TRUE(find("a::value").constant);
}

TEST(GraphExport, CycleIsNamedWithItsEdgeKinds) {
  ctx_type ctx;
  add(ctx, "a", "b");
  add(ctx, "b", "a");
  try {
    ctx.finalize();
    FAIL() << "a cycle must throw";
  } catch (std::runtime_error const &e) {
    std::string const what{e.what()};
    EXPECT_NE(what.find("a::value -[declared read]-> b::value"), std::string::npos) << what;
    EXPECT_NE(what.find("b::value -[declared read]-> a::value"), std::string::npos) << what;
  }
}

TEST(GraphExport, CycleThroughAThirdNodeListsAllThree) {
  ctx_type ctx;
  add(ctx, "a", "c");
  add(ctx, "b", "a");
  add(ctx, "c", "b");
  add(ctx, "d", "a"); // downstream of the cycle, not part of it
  try {
    ctx.finalize();
    FAIL() << "a cycle must throw";
  } catch (std::runtime_error const &e) {
    std::string const what{e.what()};
    for (auto const *k : {"a::value", "b::value", "c::value"})
      EXPECT_NE(what.find(k), std::string::npos) << what;
    EXPECT_EQ(what.find("d::value"), std::string::npos) << what;
  }
}

TEST(GraphExport, DotHasOneNodePerPropertyAndOneEdgePerGraphEdge) {
  ctx_type ctx;
  add(ctx, "a");
  add(ctx, "b", "a");
  add(ctx, "c", "b", "local");
  ctx.finalize();
  auto const dot{nm::to_dot(ctx)};
  auto const count{[&](std::string const &s) {
    std::size_t n{0};
    for (auto pos{dot.find(s)}; pos != std::string::npos; pos = dot.find(s, pos + 1)) ++n;
    return n;
  }};
  EXPECT_EQ(dot.rfind("digraph", 0), 0u);
  EXPECT_EQ(count("[label=\"value\""), 3u);
  EXPECT_EQ(count(" -> "), 2u);
  EXPECT_NE(dot.find("\"b::value\" -> \"c::value\" [style=dashed"), std::string::npos) << dot;
}

TEST(GraphExport, JsonListsNodesAndEdges) {
  ctx_type ctx;
  add(ctx, "a");
  add(ctx, "b", "a");
  ctx.finalize();
  auto const j = nlohmann::json::parse(nm::to_json(ctx));
  ASSERT_EQ(j.at("nodes").size(), 2u);
  ASSERT_EQ(j.at("edges").size(), 1u);
  EXPECT_EQ(j.at("edges")[0].at("from"), "a::value");
  EXPECT_EQ(j.at("edges")[0].at("to"), "b::value");
  EXPECT_EQ(j.at("edges")[0].at("kind"), "declared read");
}

TEST(GraphExport, SelfLoopCycleIsNamed) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "a");
  p.insert<std::string>("prop", "value");
  p.insert<bool>("self_order", true);
  ctx.create<odd<policy>>(p);
  try {
    ctx.finalize();
    FAIL() << "a self-loop must throw";
  } catch (std::runtime_error const &e) {
    std::string const what{e.what()};
    EXPECT_NE(what.find("Circular dependency"), std::string::npos) << what;
    EXPECT_NE(what.find("a::value -[callback order]-> a::value"), std::string::npos) << what;
  }
}

TEST(GraphExport, UpstreamSurvivesASecondFinalize) {
  ctx_type ctx;
  add(ctx, "a");
  add(ctx, "b", "a");
  ctx.finalize();
  auto const before{nm::to_json(ctx)};
  EXPECT_NO_THROW(ctx.update_property("b", "value"));
  ctx.finalize(); // already finalized: a no-op that must leave the graph and its upstream cache intact
  EXPECT_NO_THROW(ctx.update_property("b", "value"));
  EXPECT_EQ(nm::to_json(ctx), before);
  EXPECT_EQ(ctx.get<T>("b", "value"), T{2});
}

TEST(GraphExport, StatefulNodeIsFlagged) {
  ctx_type ctx;
  create<ticker>(ctx, "t");
  add(ctx, "a");
  ctx.finalize();
  for (auto const &n : ctx.graph().nodes) {
    EXPECT_EQ(n.stateful, key(n.property) == "t::value") << key(n.property);
    EXPECT_EQ(n.constant, key(n.property) == "a::value") << key(n.property);
  }
}

TEST(GraphExport, RoutedNodeHasNoCallbackOfItsOwn) {
  ctx_type ctx;
  create<pair>(ctx, "p");
  ctx.finalize();
  auto const &nodes{ctx.graph().nodes};
  ASSERT_EQ(nodes.size(), 2u);
  for (auto const &n : nodes) {
    bool const second{key(n.property) == "p::second"};
    EXPECT_EQ(n.routed, second) << key(n.property);
    EXPECT_EQ(n.callback, !second) << key(n.property);
  }
}

TEST(GraphExport, DotEscapesQuotesBackslashesAndNewlines) {
  ctx_type ctx;
  add_odd(ctx, "m\"x\\y\nz", "p\"q\\r\ns");
  ctx.finalize();
  auto const dot{nm::to_dot(ctx)};
  EXPECT_TRUE(dot_strings_are_valid(dot)) << dot;
  EXPECT_NE(dot.find("label=\"m\\\"x\\\\y\\nz\""), std::string::npos) << dot;
  EXPECT_NE(dot.find("label=\"p\\\"q\\\\r\\ns\""), std::string::npos) << dot;
}

TEST(GraphExport, JsonSurvivesInvalidUtf8) {
  ctx_type ctx;
  add_odd(ctx, "m", "bad\xff" "name");
  ctx.finalize();
  std::string json;
  ASSERT_NO_THROW(json = nm::to_json(ctx));
  auto const j = nlohmann::json::parse(json);
  EXPECT_NE(j.at("nodes")[0].at("key").get<std::string>().find("\xEF\xBF\xBD" "name"), std::string::npos) << json;
}

} // namespace
