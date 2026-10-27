#ifndef NUMSIM_MATERIALS_CORE_GRAPH_DESCRIPTION_H
#define NUMSIM_MATERIALS_CORE_GRAPH_DESCRIPTION_H

#include <algorithm>
#include <format>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "numsim-materials/core/material_context.h"

namespace numsim::materials {

struct produced_property {
  std::string name;
  bool history{false};
};

struct consumed_property {
  std::string source; ///< "material::property"
  bool local{false};  ///< EdgeKind::Local
};

struct material_description {
  std::string name;
  std::vector<produced_property> produced; ///< sorted by name
  std::vector<consumed_property> inputs;
};

/// Materials in creation order with what they produce and consume.
template <typename Traits>
std::vector<material_description> describe_graph(material_context<Traits> const &ctx) {
  std::vector<material_description> out;
  for (auto const *mat : ctx.materials()) {
    material_description d{std::string{mat->name()}, {}, {}};
    for (auto const *p : mat->get_property_registry().produced_properties())
      d.produced.push_back({p->traits().id.name, p->is_history()});
    std::ranges::sort(d.produced, {}, &produced_property::name); // the registry orders by address
    for (auto const &input : mat->typed_inputs())
      d.inputs.push_back({input->source_owner() + "::" + input->source_name(), input->edge_kind() == EdgeKind::Local});
    out.push_back(std::move(d));
  }
  return out;
}

namespace detail {

inline std::string graph_key(property_base const *p) { return p->traits().id.owner + "::" + p->traits().id.name; }

inline std::string dot_style(graph_node const &n) {
  std::string style;
  if (n.constant) style += ", style=filled, fillcolor=lightgrey";
  if (n.history) style += ", peripheries=2";
  if (n.stateful) style += ", color=orange";
  if (!n.callback && !n.routed) style += ", fontcolor=gray40";
  return style;
}

} // namespace detail

/// Graphviz: one cluster per material; local edges dashed, constants shaded, history doubled.
template <typename Traits>
std::string to_dot(material_context<Traits> const &ctx) {
  auto const g{ctx.graph()};
  std::string out{"digraph materials {\n  rankdir=LR;\n  node [shape=box];\n"};
  for (auto const *mat : ctx.materials()) {
    out += std::format("  subgraph \"cluster_{0}\" {{\n    label=\"{0}\";\n", mat->name());
    for (auto const &n : g.nodes)
      if (n.property->traits().id.owner == mat->name())
        out += std::format("    \"{}\" [label=\"{}\"{}];\n", detail::graph_key(n.property), n.property->traits().id.name,
                           detail::dot_style(n));
    out += "  }\n";
  }
  for (auto const &e : g.edges)
    out += std::format("  \"{}\" -> \"{}\" [{}label=\"{}\"];\n", detail::graph_key(e.from), detail::graph_key(e.to),
                       e.kind == edge_kind::local ? "style=dashed, " : "", to_string(e.kind));
  return out + "}\n";
}

/// {"nodes": [{"key", "history", "constant", "stateful", "routed", "callback"}], "edges": [{"from", "to", "kind"}]}
template <typename Traits>
std::string to_json(material_context<Traits> const &ctx) {
  auto const g{ctx.graph()};
  nlohmann::json nodes = nlohmann::json::array();
  for (auto const &n : g.nodes)
    nodes.push_back({{"key", detail::graph_key(n.property)},
                     {"history", n.history},
                     {"constant", n.constant},
                     {"stateful", n.stateful},
                     {"routed", n.routed},
                     {"callback", n.callback}});
  nlohmann::json edges = nlohmann::json::array();
  for (auto const &e : g.edges)
    edges.push_back({{"from", detail::graph_key(e.from)}, {"to", detail::graph_key(e.to)}, {"kind", std::string{to_string(e.kind)}}});
  nlohmann::json const doc = {{"nodes", nodes}, {"edges", edges}};
  return doc.dump(2);
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_GRAPH_DESCRIPTION_H
