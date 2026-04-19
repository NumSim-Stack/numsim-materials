#ifndef NUMSIM_MATERIALS_PROPERTY_ENGINE_H
#define NUMSIM_MATERIALS_PROPERTY_ENGINE_H

#include <algorithm>
#include <print>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "numsim-materials/core/property_traits.h"
#include "numsim-materials/core/property.h"
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
  }

  /// Run all property update callbacks in topological order.
  void update() {
    for (auto* prop : m_property_execution_order) {
      if (prop->traits().update)
        prop->traits().update();
    }
  }

  /// Update a single property and all its upstream dependencies.
  /// Properties in @p exclude are skipped during evaluation.
  void update_property(const std::string& material, const std::string& property,
                       const std::unordered_set<const property_base*>& exclude = {}) {
    auto key = material + "::" + property;
    auto subgraph = collect_upstream(key);
    for (auto* prop : m_property_execution_order) {
      if (subgraph.contains(prop) && prop->traits().update
          && !exclude.contains(prop))
        prop->traits().update();
    }
  }

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

  template<typename T>
  const T& get(property_handler& properties,
               const std::string& material, const std::string& prop_name) {
    auto* p = find_property(properties, material, prop_name);
    if (!p)
      throw std::runtime_error("property_engine::get(): property '" +
                               material + "::" + prop_name + "' not found");
    auto* typed = dynamic_cast<const numsim_core::property<T, property_traits>*>(p);
    if (!typed)
      throw std::runtime_error("property_engine::get(): type mismatch for '" +
                               material + "::" + prop_name + "'");
    return typed->get();
  }

  template<typename T>
  T& get_mutable(property_handler& properties,
                 const std::string& material, const std::string& prop_name) {
    auto* p = find_property(properties, material, prop_name);
    if (!p)
      throw std::runtime_error("property_engine::get_mutable(): property '" +
                               material + "::" + prop_name + "' not found");
    auto* typed = dynamic_cast<numsim_core::property<T, property_traits>*>(p);
    if (!typed)
      throw std::runtime_error("property_engine::get_mutable(): type mismatch for '" +
                               material + "::" + prop_name + "'");
    return typed->get();
  }

  const std::vector<property_base*>& property_execution_order() const noexcept {
    return m_property_execution_order;
  }

  const std::vector<material_interface_type*>& execution_order() const noexcept {
    return m_execution_order;
  }

  void dump() const {
    std::println("=== Property execution order ({} properties) ===",
                 m_property_execution_order.size());
    for (auto* prop : m_property_execution_order)
      std::println("  {}::{}{}", prop->traits().id.owner, prop->traits().id.name,
                   prop->traits().update ? "" : " (no callback)");
  }

private:
  static std::string prop_key(const property_id& id) {
    return id.owner + "::" + id.name;
  }

  void compute_execution_order(property_handler& properties,
                                const std::vector<material_interface_type*>& interfaces) {
    std::unordered_map<std::string, property_base*> key_to_prop;
    std::unordered_map<std::string, int> indegree;
    std::unordered_map<std::string, std::vector<std::string>> edges;
    std::unordered_map<std::string, std::vector<std::string>> reverse_edges;

    for (auto& [owner, props] : properties.data()) {
      for (auto& [name, prop] : props) {
        auto key = owner + "::" + name;
        key_to_prop[key] = prop.get();
        indegree.try_emplace(key, 0);
        edges.try_emplace(key);
        reverse_edges.try_emplace(key);
      }
    }

    auto add_edge = [&](const std::string& from, const std::string& to) {
      auto& adj = edges[from];
      if (std::find(adj.begin(), adj.end(), to) == adj.end()) {
        adj.push_back(to);
        reverse_edges[to].push_back(from);
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

      for (auto& next : edges[key])
        if (--indegree[next] == 0)
          ready.push(next);
    }

    if (m_property_execution_order.size() != key_to_prop.size())
      throw std::runtime_error("Circular dependency in property graph "
                               "(considering Global edges only)");

    m_reverse_edges = std::move(reverse_edges);
    m_key_to_prop = std::move(key_to_prop);
  }

  void collect_history_properties(property_handler& properties) {
    m_history_properties.clear();
    for (auto& [owner, props] : properties.data())
      for (auto& [name, prop] : props)
        if (prop->is_history())
          m_history_properties.push_back(prop.get());
  }

  std::unordered_set<property_base*> collect_upstream(const std::string& key) const {
    std::unordered_set<property_base*> result;
    std::queue<std::string> queue;
    std::unordered_set<std::string> visited;

    queue.push(key);
    visited.insert(key);

    while (!queue.empty()) {
      auto current = queue.front();
      queue.pop();
      if (auto it = m_key_to_prop.find(current); it != m_key_to_prop.end())
        result.insert(it->second);
      if (auto rit = m_reverse_edges.find(current); rit != m_reverse_edges.end())
        for (auto& upstream : rit->second)
          if (!visited.contains(upstream)) {
            visited.insert(upstream);
            queue.push(upstream);
          }
    }
    return result;
  }

  std::vector<material_interface_type*> m_execution_order;
  std::vector<property_base*> m_property_execution_order;
  std::unordered_map<std::string, property_base*> m_key_to_prop;
  std::unordered_map<std::string, std::vector<std::string>> m_reverse_edges;
  std::vector<property_base*> m_history_properties;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_PROPERTY_ENGINE_H
