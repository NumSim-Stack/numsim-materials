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

} // namespace
