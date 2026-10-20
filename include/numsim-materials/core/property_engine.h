#ifndef PROPERTY_ENGINE_H
#define PROPERTY_ENGINE_H

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numsim-core/print.h>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "numsim-materials/core/property_traits.h"
#include "numsim-materials/core/property.h"
#include "numsim-materials/core/history_property.h"
#include "numsim-materials/core/material_interface.h"

namespace numsim::materials {

/// Manages the property dependency graph, topological sort, and execution.
template<typename Traits>
class property_engine {
public:
  using property_handler = typename Traits::PropertyHandler;
  using material_interface_type = material_interface<Traits>;

  property_engine() = default;

  /// Build the execution order from the property registry and material interfaces.
  void build(property_handler& properties,
             const std::vector<material_interface_type*>& interfaces) {
    compute_execution_order(properties, interfaces);
    collect_history_properties(properties);
    find_constants(properties, interfaces);
  }

  /// Run all property update callbacks in topological order; a constant one only until it has run.
  void update() {
    refresh_constants();
    for (auto* prop : m_property_execution_order)
      run(prop);
  }

  /// Update a single property and all its upstream dependencies.
  /// Properties in @p exclude are skipped during evaluation.
  void update_property(const std::string& material, const std::string& property,
                       const std::unordered_set<const property_base*>& exclude = {}) {
    auto* target = find_in_graph(material, property);
    if (!target) return;
    update_property(target, exclude);
  }

  /// Update a single property (by pointer) and all its upstream dependencies.
  void update_property(const property_base* target,
                       const std::unordered_set<const property_base*>& exclude = {}) {
    refresh_constants();
    auto subgraph = collect_upstream(target);
    for (auto* prop : m_property_execution_order) {
      if (subgraph.contains(prop) && !exclude.contains(prop))
        run(prop);
    }
  }

  /// A property computed once: declared constant (material_base::constant_output) and reading only constants.
  bool is_constant(const property_base* prop) const { return m_constant.contains(prop); }

  /// Bumped by every write outside the graph (set_parameter, get_mutable): constants are recomputed.
  std::size_t* revision() noexcept { return m_revision.get(); }

  void commit() {
    for (auto* hp : m_history_properties)
      hp->commit();
  }

  void revert() {
    for (auto* hp : m_history_properties)
      hp->revert();
  }

  property_base* find_property(property_handler& properties,
                               const std::string& material,
                               const std::string& property) {
    auto it = properties.data().find(material);
    if (it == properties.data().end()) return nullptr;
    auto jt = it->second.find(property);
    return jt != it->second.end() ? jt->second.get() : nullptr;
  }

  /// Read-only access to a property value.
  /// Handles both regular properties and history properties (returns new_value).
  template<typename T>
  const T& get(property_handler& properties,
               const std::string& material, const std::string& prop_name) {
    auto* p = find_property(properties, material, prop_name);
    if (!p)
      throw std::runtime_error("property_engine::get(): property '" +
                               material + "::" + prop_name + "' not found");
    if (auto* typed = dynamic_cast<const numsim_core::property<T, property_traits>*>(p))
      return typed->get();
    if (auto* hist = dynamic_cast<const numsim_core::history_property<T, property_traits>*>(p))
      return hist->new_value();
    throw std::runtime_error("property_engine::get(): type mismatch for '" +
                             material + "::" + prop_name + "'");
  }

  /// Mutable access to a property value — bypasses the property graph.
  /// Handles both regular properties and history properties (returns new_value).
  template<typename T>
  T& get_mutable(property_handler& properties,
                 const std::string& material, const std::string& prop_name) {
    ++*m_revision; // the caller may write anything, a constant included
    auto* p = find_property(properties, material, prop_name);
    if (!p)
      throw std::runtime_error("property_engine::get_mutable(): property '" +
                               material + "::" + prop_name + "' not found");
    if (auto* typed = dynamic_cast<numsim_core::property<T, property_traits>*>(p))
      return typed->get();
    if (auto* hist = dynamic_cast<numsim_core::history_property<T, property_traits>*>(p))
      return hist->new_value();
    throw std::runtime_error("property_engine::get_mutable(): type mismatch for '" +
                             material + "::" + prop_name + "'");
  }

  const std::vector<property_base*>& property_execution_order() const noexcept {
    return m_property_execution_order;
  }

  const std::vector<material_interface_type*>& execution_order() const noexcept {
    return m_execution_order;
  }

  void dump() const {
    numsim_core::println(stderr, "=== Property execution order ({} properties) ===",
                 m_property_execution_order.size());
    for (auto* prop : m_property_execution_order)
      numsim_core::println(stderr, "  {}::{}{}{}", prop->traits().id.owner, prop->traits().id.name,
                   prop->traits().update ? "" : " (no callback)", is_constant(prop) ? " (constant)" : "");
  }

private:
  static std::string prop_key(const property_id& id) {
    return id.owner + "::" + id.name;
  }

  /// Lookup a property in the pointer-based graph (no string allocation).
  property_base* find_in_graph(const std::string& material, const std::string& property) const {
    auto key = material + "::" + property;
    auto it = m_key_to_prop.find(key);
    return it != m_key_to_prop.end() ? it->second : nullptr;
  }

  void compute_execution_order(property_handler& properties,
                                const std::vector<material_interface_type*>& interfaces) {
    // --- Build phase uses strings (runs once at finalize) ---
    std::unordered_map<std::string, property_base*> key_to_prop;
    std::unordered_map<std::string, int> indegree;
    std::unordered_map<std::string, std::vector<std::string>> str_edges;
    std::unordered_map<std::string, std::vector<std::string>> str_reverse_edges;

    for (auto& [owner, props] : properties.data()) {
      for (auto& [name, prop] : props) {
        auto key = owner + "::" + name;
        key_to_prop[key] = prop.get();
        indegree.try_emplace(key, 0);
        str_edges.try_emplace(key);
        str_reverse_edges.try_emplace(key);
      }
    }

    auto add_edge = [&](const std::string& from, const std::string& to) {
      auto& adj = str_edges[from];
      if (std::find(adj.begin(), adj.end(), to) == adj.end()) {
        adj.push_back(to);
        str_reverse_edges[to].push_back(from);
        indegree[to]++;
      }
    };

    for (auto& [owner, props] : properties.data()) {
      for (auto& [name, prop] : props) {
        auto this_key = owner + "::" + name;
        for (auto& dep : prop->traits().input_dependencies) {
          if (dep.kind == EdgeKind::Global) {
            auto dep_key = dep.id.owner + "::" + dep.id.name;
            if (key_to_prop.contains(dep_key))
              add_edge(dep_key, this_key);
          }
        }
      }
    }

    std::unordered_map<std::string, material_interface_type*> by_name;
    for (auto* mat : interfaces)
      by_name[std::string(mat->name())] = mat;

    for (auto* mat : interfaces) {
      for (auto& input : mat->typed_inputs()) {
        if (input->edge_kind() == EdgeKind::Global) {
          auto src_key = input->source_owner() + "::" + input->source_name();
          const auto& reg = mat->get_property_registry();
          for (auto* prop : reg.produced_properties()) {
            auto dst_key = prop_key(prop->traits().id);
            if (key_to_prop.contains(src_key) && src_key != dst_key)
              add_edge(src_key, dst_key);
          }
        }
      }
    }

    // --- Topological sort ---
    std::queue<std::string> ready;
    for (auto& [key, deg] : indegree)
      if (deg == 0) ready.push(key);

    m_property_execution_order.clear();
    std::unordered_set<std::string> visited_materials;
    m_execution_order.clear();

    while (!ready.empty()) {
      auto key = ready.front();
      ready.pop();
      if (auto it = key_to_prop.find(key); it != key_to_prop.end())
        m_property_execution_order.push_back(it->second);

      auto dot = key.find("::");
      auto mat_name = key.substr(0, dot);
      if (by_name.contains(mat_name) && !visited_materials.contains(mat_name)) {
        visited_materials.insert(mat_name);
        m_execution_order.push_back(by_name[mat_name]);
      }

      for (auto& next : str_edges[key])
        if (--indegree[next] == 0)
          ready.push(next);
    }

    if (m_property_execution_order.size() != key_to_prop.size())
      throw std::runtime_error("Circular dependency in property graph "
                               "(considering Global edges only)");

    // --- Convert to pointer-based structures for runtime ---
    m_key_to_prop = std::move(key_to_prop);
    m_reverse_edges.clear();
    for (auto& [key, upstreams] : str_reverse_edges) {
      auto* prop = m_key_to_prop[key];
      auto& ptr_upstreams = m_reverse_edges[prop];
      for (auto& up_key : upstreams)
        ptr_upstreams.push_back(m_key_to_prop[up_key]);
    }
  }

  /// Constant: declared, not history, every read a global edge from a constant (the topological order has the
  /// reads first). A callback runs once only when the outputs of its material without a callback of their own
  /// (written by some callback of it) are constant as well.
  void find_constants(property_handler& properties, const std::vector<material_interface_type*>& interfaces) {
    std::unordered_map<const property_base*, const std::vector<numsim_core::input_wire_base const*>*> declared;
    for (auto* mat : interfaces)
      for (auto& d : mat->constant_outputs()) {
        auto* p = find_property(properties, std::string(mat->name()), d.output);
        if (!p)
          throw std::logic_error("constant_output(): '" + std::string(mat->name()) + "::" + d.output +
                                 "' is not an output of the material");
        declared[p] = &d.reads;
      }
    m_constant.clear();
    for (auto* prop : m_property_execution_order) {
      auto it = declared.find(prop);
      if (it == declared.end() || prop->is_history())
        continue;
      bool const reads_constants{std::ranges::all_of(*it->second, [&](auto const* read) {
        if (read->edge_kind() != EdgeKind::Global)
          return false;
        auto* source = find_in_graph(read->source_owner(), read->source_name());
        return source && m_constant.contains(source);
      })};
      if (reads_constants)
        m_constant.insert(prop);
    }
    m_once.clear();
    for (auto& [owner, props] : properties.data()) {
      bool const quiet_constant{std::ranges::all_of(props, [&](auto const& entry) {
        return entry.second->traits().update || m_constant.contains(entry.second.get());
      })};
      for (auto& [name, prop] : props)
        if (quiet_constant && prop->traits().update && m_constant.contains(prop.get()))
          m_once.insert(prop.get());
    }
    m_done.clear();
  }

  void refresh_constants() {
    if (*m_revision != m_seen_revision) {
      m_done.clear();
      m_seen_revision = *m_revision;
    }
  }

  void run(property_base* prop) {
    auto& update = prop->traits().update;
    if (!update)
      return;
    bool const once{m_once.contains(prop)};
    if (once && m_done.contains(prop))
      return;
    update();
    if (once)
      m_done.insert(prop);
  }

  void collect_history_properties(property_handler& properties) {
    m_history_properties.clear();
    for (auto& [owner, props] : properties.data())
      for (auto& [name, prop] : props)
        if (prop->is_history())
          m_history_properties.push_back(prop.get());
  }

  /// Collect all upstream dependencies of a property (pointer-based, no string allocation).
  std::unordered_set<property_base*> collect_upstream(const property_base* target) const {
    std::unordered_set<property_base*> result;
    std::queue<const property_base*> queue;
    std::unordered_set<const property_base*> visited;

    queue.push(target);
    visited.insert(target);

    while (!queue.empty()) {
      auto* current = queue.front();
      queue.pop();
      result.insert(const_cast<property_base*>(current));
      if (auto rit = m_reverse_edges.find(const_cast<property_base*>(current));
          rit != m_reverse_edges.end())
        for (auto* upstream : rit->second)
          if (!visited.contains(upstream)) {
            visited.insert(upstream);
            queue.push(upstream);
          }
    }
    return result;
  }

  std::vector<material_interface_type*> m_execution_order;
  std::vector<property_base*> m_property_execution_order;
  std::unordered_map<std::string, property_base*> m_key_to_prop;  // for string-based find
  std::unordered_map<property_base*, std::vector<property_base*>> m_reverse_edges;  // pointer-based
  std::vector<property_base*> m_history_properties;
  std::unordered_set<const property_base*> m_constant;
  std::unordered_set<const property_base*> m_once; ///< callbacks run once
  std::unordered_set<const property_base*> m_done; ///< ... and already run since the last revision
  std::unique_ptr<std::size_t> m_revision{std::make_unique<std::size_t>(0)}; ///< stable address for the materials
  std::size_t m_seen_revision{0};
};

} // namespace numsim::materials

#endif // PROPERTY_ENGINE_H
