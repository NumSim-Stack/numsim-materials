#ifndef PROPERTY_ENGINE_H
#define PROPERTY_ENGINE_H

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <print>
#include <queue>
#include <ranges>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "numsim-materials/core/property_traits.h"
#include "numsim-materials/core/property.h"
#include "numsim-materials/core/history_property.h"
#include "numsim-materials/core/material_interface.h"

namespace numsim::materials {

/// How an edge entered the graph; local edges are kept but not sorted.
enum class edge_kind { dependency, input, declared_read, drives, material_ref, callback_order, written_by, local };

constexpr std::string_view to_string(edge_kind kind) noexcept {
  switch (kind) {
  case edge_kind::dependency: return "dependency";
  case edge_kind::input: return "input";
  case edge_kind::declared_read: return "declared read";
  case edge_kind::drives: return "drives";
  case edge_kind::material_ref: return "material_ref";
  case edge_kind::callback_order: return "callback order";
  case edge_kind::written_by: return "written_by";
  case edge_kind::local: return "local";
  }
  return "?";
}

struct graph_edge {
  const property_base* from;
  const property_base* to;
  edge_kind kind;
};

struct graph_node {
  const property_base* property;
  bool history{false};
  bool constant{false};
  bool stateful{false};
  bool routed{false};   ///< filled by its writer's callback
  bool callback{false}; ///< has a callback of its own
};

/// The finalized graph: nodes in execution order; edges are the sorted global ones, then the sorted local ones.
struct property_graph {
  std::vector<graph_node> nodes;
  std::vector<graph_edge> edges;
};

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
    std::ranges::for_each(interfaces, &material_interface_type::resolve_all_inputs);
    std::ranges::for_each(interfaces, &material_interface_type::check_reads);
    auto const writers{collect_writers(interfaces)};
    compute_execution_order(properties, interfaces, writers);
    m_stateful.clear();
    for (auto const* mat : interfaces)
      m_stateful.insert(mat->stateful_outputs().begin(), mat->stateful_outputs().end());
    collect_history_properties(properties);
    find_constants(interfaces, writers);
    if constexpr (NUMSIM_MATERIALS_DEBUG_CHECKS)
      track_reads(interfaces);
    route_to_writers(interfaces);
  }

  /// Run the callbacks still to run, in topological order: a constant one only until it has run.
  void update() {
    guarded("update", [this] {
      refresh_constants();
      if (m_verify)
        std::ranges::for_each(m_retired, [this](property_base* prop) { verify(prop); });
      for (auto* prop : m_active)
        prop->traits().update();
      retire([](property_base*) { return true; });
    });
  }

  /// Update a single property and all its upstream dependencies.
  /// Properties in @p exclude are skipped during evaluation.
  void update_property(const std::string& material, const std::string& property,
                       const std::unordered_set<const property_base*>& exclude = {}) {
    auto* target = find_in_graph(material, property);
    // Returning quietly here made a typo indistinguishable from a property that
    // legitimately never changes: the driver asks for an update, gets no error,
    // and reads a stale value for the rest of the analysis.
    if (!target)
      throw std::runtime_error(
          "update_property(): property '" + material + "::" + property +
          "' is not in the graph. Check the material and property names, and "
          "that the material publishes this property.");
    update_property(target, exclude);
  }

  /// Update a single property (by pointer) and all its upstream dependencies.
  /// From a callback (numerical_diff_checker), a reevaluation: every upstream callback runs, constants included,
  /// and the running update keeps its state.
  void update_property(const property_base* target,
                       const std::unordered_set<const property_base*>& exclude = {}) {
    auto const& subgraph{upstream_of(target)};
    auto const runs{[&](property_base* prop) { return subgraph.contains(prop) && !exclude.contains(prop); }};
    if (m_running) {
      for (auto* prop : m_callbacks | std::views::filter(runs))
        prop->traits().update();
      return;
    }
    guarded("update_property", [&] {
      refresh_constants();
      if (m_verify)
        for (auto* prop : m_retired | std::views::filter(runs))
          verify(prop);
      for (auto* prop : m_active | std::views::filter(runs))
        prop->traits().update();
      retire(runs);
    });
  }

  /// A property computed once: declared constant (material_base::constant_output) and reading only constants.
  bool is_constant(const property_base* prop) const { return m_constant.contains(prop); }

  /// Bumped by every write outside the graph (set_parameter, get_mutable): constants are recomputed.
  std::size_t* revision() noexcept { return m_revision.get(); }

  /// Rerun the constant callbacks on every update and throw when an output changes (on with NUMSIM_MATERIALS_DEBUG_CHECKS).
  void verify_constants(bool on) noexcept { m_verify = on; }

  /// Throw when a callback reads an input it did not declare (compiled in with NUMSIM_MATERIALS_DEBUG_CHECKS).
  void check_reads(bool on) noexcept { *m_check_reads = on; }

  /// Off: every callback runs on every update, the constant ones too.
  void fold_constants(bool on) noexcept {
    m_fold = on;
    reactivate();
  }

  void commit() {
    if (m_failed)
      throw std::logic_error("commit(): the last update threw, the context is partly updated: revert() first");
    for (auto* hp : m_history_properties)
      hp->commit();
  }

  /// Restores the history and clears a failed update.
  void revert() {
    m_failed = false;
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

  /// Nodes with their flags and every edge with its kind (after build()).
  property_graph graph() const {
    property_graph g{{}, m_edges};
    for (auto const* prop : m_property_execution_order)
      g.nodes.push_back({prop, prop->is_history(), is_constant(prop), m_stateful.contains(prop), m_routed.contains(prop),
                         has_callback(prop) && !m_routed.contains(prop)});
    return g;
  }

  void dump() const {
    std::println(stderr, "=== Property execution order ({} properties) ===",
                 m_property_execution_order.size());
    for (auto* prop : m_property_execution_order)
      std::println(stderr, "  {}::{}{}{}", prop->traits().id.owner, prop->traits().id.name,
                   m_routed.contains(prop) ? " (by its writer)" : prop->traits().update ? "" : " (no callback)", is_constant(prop) ? " (constant)" : "");
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

  using writers_map = std::unordered_map<const property_base*, std::vector<const property_base*>>;
  using writers_by_material = std::unordered_map<const material_interface_type*, writers_map>;

  void compute_execution_order(property_handler& properties,
                                const std::vector<material_interface_type*>& interfaces,
                                const writers_by_material& writers) {
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

    // edges keep the first kind seen; local edges are recorded but take no part in the sort
    std::map<std::pair<std::string, std::string>, edge_kind> str_kinds;
    std::map<std::pair<std::string, std::string>, edge_kind> str_local;
    auto add_edge = [&](const std::string& from, const std::string& to, edge_kind kind) {
      auto& adj = str_edges[from];
      if (!std::ranges::contains(adj, to)) {
        adj.push_back(to);
        str_reverse_edges[to].push_back(from);
        str_kinds.try_emplace({from, to}, kind);
        indegree[to]++;
      }
    };

    for (auto& [owner, props] : properties.data()) {
      for (auto& [name, prop] : props) {
        auto this_key = owner + "::" + name;
        for (auto& dep : prop->traits().input_dependencies) {
          auto dep_key = dep.id.owner + "::" + dep.id.name;
          if (!key_to_prop.contains(dep_key))
            continue;
          if (dep.kind == EdgeKind::Global)
            add_edge(dep_key, this_key, edge_kind::dependency);
          else
            str_local.try_emplace({dep_key, this_key}, edge_kind::local);
        }
      }
    }

    std::unordered_map<std::string, material_interface_type*> by_name;
    for (auto* mat : interfaces)
      by_name[std::string(mat->name())] = mat;

    // global inputs: to the callback declaring the read, to every other output of the material without a declaration
    auto const add_input_edge{[&](input_wire_base const* input, const property_base* to, edge_kind kind) {
      auto const src_key{input->source_owner() + "::" + input->source_name()};
      auto const dst_key{prop_key(to->traits().id)};
      if (!key_to_prop.contains(src_key) || src_key == dst_key)
        return;
      if (input->edge_kind() == EdgeKind::Global)
        add_edge(src_key, dst_key, kind);
      else if (kind == edge_kind::declared_read || kind == edge_kind::input)
        str_local.try_emplace({src_key, dst_key}, edge_kind::local);
    }};
    for (auto* mat : interfaces) {
      std::unordered_set<const property_base*> declared;
      for (auto const& d : mat->declared_reads()) {
        declared.insert(d.outputs.begin(), d.outputs.end());
        for (auto const* input : d.inputs)
          add_input_edge(input, d.callback, edge_kind::declared_read);
      }
      for (auto* prop : mat->get_property_registry().produced_properties())
        if (!declared.contains(prop))
          for (auto const& input : mat->typed_inputs())
            add_input_edge(input.get(), prop, edge_kind::input);
    }

    // dependencies outside the inputs: what a driven callback reads globally, and the material behind a
    // material_ref; reads of the material's own outputs order nothing there (it fills them itself)
    auto const add_outside_edge{[&](const material_interface_type& mat, input_wire_base const* input, const property_base* to,
                                     edge_kind kind) {
      if (input->source_owner() != mat.name())
        add_input_edge(input, to, kind);
    }};
    for (auto* mat : interfaces)
      for (auto const& d : mat->declared_reads())
        for (auto const* driven : d.drives)
          if (auto const it{by_name.find(driven->source_owner())}; it != by_name.end())
            for (auto const* read : reads_of(*it->second, driven->source_name()))
              add_outside_edge(*mat, read, d.callback, edge_kind::drives);
    for (auto* mat : interfaces)
      for (auto const& target_name : mat->material_ref_targets())
        if (auto const it{by_name.find(target_name)}; it != by_name.end()) {
          auto const& target{*it->second};
          for (auto const* callback : callbacks_of(*mat)) {
            for (auto const& input : target.typed_inputs())
              add_outside_edge(*mat, input.get(), callback, edge_kind::material_ref); // what the target reads, the referencing callbacks read
            auto const reads{reads_of(*mat, callback->traits().id.name)};
            for (auto const* out : bare_outputs(target)) // filled by the referencing callbacks, unless one reads it
              if (std::ranges::none_of(reads, [&](auto const* read) {
                    return read->source_owner() == target.name() && read->source_name() == out->traits().id.name;
                  }))
                add_edge(prop_key(callback->traits().id), prop_key(out->traits().id), edge_kind::material_ref);
          }
        }

    // within a material: declared callback order, and an output without a callback after its writers
    for (auto* mat : interfaces) {
      for (auto const& [earlier, later] : mat->callback_order())
        add_edge(prop_key(earlier->traits().id), prop_key(later->traits().id), edge_kind::callback_order);
      for (auto const& [output, by] : writers.at(mat))
        for (auto const* writer : by)
          add_edge(prop_key(writer->traits().id), prop_key(output->traits().id), edge_kind::written_by);
    }

    // --- Topological sort, ties broken by creation order, then key: the order does not depend on hashing ---
    std::unordered_map<std::string, std::size_t> created;
    for (std::size_t i{0}; i < interfaces.size(); ++i)
      created.emplace(interfaces[i]->name(), i);
    auto const rank{[&](std::string const& key) {
      auto const it{created.find(key.substr(0, key.find("::")))};
      return std::pair{it != created.end() ? it->second : interfaces.size(), key};
    }};
    std::set<std::pair<std::size_t, std::string>> ready;
    for (auto& [key, deg] : indegree)
      if (deg == 0) ready.insert(rank(key));

    m_property_execution_order.clear();
    std::unordered_set<std::string> visited_materials;
    m_execution_order.clear();

    while (!ready.empty()) {
      auto key = std::move(ready.extract(ready.begin()).value().second);
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
          ready.insert(rank(next));
    }

    if (m_property_execution_order.size() != key_to_prop.size()) {
      std::set<std::string> left;
      for (auto const& [key, deg] : indegree)
        if (deg > 0) left.insert(key);
      throw std::runtime_error("Circular dependency in property graph: " + describe_cycle(find_cycle(left, str_edges), str_kinds));
    }

    // --- Convert to pointer-based structures for runtime ---
    m_key_to_prop = std::move(key_to_prop);
    m_reverse_edges.clear();
    for (auto& [key, upstreams] : str_reverse_edges) {
      auto* prop = m_key_to_prop[key];
      auto& ptr_upstreams = m_reverse_edges[prop];
      for (auto& up_key : upstreams)
        ptr_upstreams.push_back(m_key_to_prop[up_key]);
    }
    m_upstream.clear();
    m_edges.clear();
    for (auto const* kinds : {&str_kinds, &str_local})
      for (auto const& [ends, kind] : *kinds)
        m_edges.push_back({m_key_to_prop.at(ends.first), m_key_to_prop.at(ends.second), kind});
  }

  /// One cycle among the nodes the sort could not place (each lies on a cycle or behind one), starting at its smallest key.
  static std::vector<std::string> find_cycle(std::set<std::string> const& left,
                                             std::unordered_map<std::string, std::vector<std::string>> const& edges) {
    std::unordered_map<std::string, int> state; // 1: on the path, 2: done
    std::vector<std::string> path;
    std::vector<std::string> cycle;
    auto const visit{[&](auto const& self, std::string const& key) -> bool {
      state[key] = 1;
      path.push_back(key);
      for (auto const& next : edges.at(key)) {
        if (!left.contains(next) || state[next] == 2)
          continue;
        if (state[next] == 1) {
          cycle.assign(std::ranges::find(path, next), path.end());
          return true;
        }
        if (self(self, next))
          return true;
      }
      state[key] = 2;
      path.pop_back();
      return false;
    }};
    for (auto const& key : left)
      if (state[key] == 0 && visit(visit, key))
        break;
    std::ranges::rotate(cycle, std::ranges::min_element(cycle));
    return cycle;
  }

  static std::string describe_cycle(std::vector<std::string> const& cycle,
                                    std::map<std::pair<std::string, std::string>, edge_kind> const& kinds) {
    std::string out{cycle.empty() ? std::string{} : cycle.front()};
    for (std::size_t i{0}; i < cycle.size(); ++i) {
      auto const& to{cycle[(i + 1) % cycle.size()]};
      out += " -[" + std::string{to_string(kinds.at({cycle[i], to}))} + "]-> " + to;
    }
    if (out.contains(to_string(edge_kind::callback_order)))
      out += " (callback order: runs_after() or a read of the material's own output)";
    return out;
  }

  /// The inputs behind the callback that fills material::property: its declared reads, else every input.
  static std::vector<const input_wire_base*> reads_of(const material_interface_type& mat, const std::string& property) {
    for (auto const& d : mat.declared_reads())
      if (std::ranges::any_of(d.outputs, [&](auto const* out) { return out->traits().id.name == property; }))
        return d.inputs;
    return mat.all_inputs();
  }

  /// The outputs with no callback of their own and no declared writer.
  static std::vector<const property_base*> bare_outputs(const material_interface_type& mat) {
    auto const written{mat.written_outputs() | std::views::keys};
    return mat.get_property_registry().produced_properties() | std::views::filter([&](const property_base* prop) {
             return !has_callback(prop) && !std::ranges::contains(written, prop);
           }) |
           std::ranges::to<std::vector<const property_base*>>();
  }

  static std::vector<const property_base*> callbacks_of(const material_interface_type& mat) {
    return mat.get_property_registry().produced_properties() | std::views::filter(has_callback) |
           std::ranges::to<std::vector<const property_base*>>();
  }

  static bool has_callback(const property_base* prop) { return static_cast<bool>(prop->traits().update); }

  /// Per material, its outputs without a callback -> the callbacks that may write them: the declared writer
  /// (written_by), else every callback of the material.
  static writers_by_material collect_writers(const std::vector<material_interface_type*>& interfaces) {
    writers_by_material result;
    for (auto const* mat : interfaces) {
      auto& writers{result[mat]};
      auto const& props{mat->get_property_registry().produced_properties()};
      auto const callbacks{callbacks_of(*mat)};
      for (auto const* prop : props | std::views::filter(std::not_fn(has_callback)))
        writers[prop] = callbacks;
      for (auto const& [output, by] : mat->written_outputs())
        writers[output] = {by};
    }
    return result;
  }

  /// Constant: declared, not history, every read a global edge from a constant or a constant output of its material. A constant callback runs once
  /// when every output it may write is constant.
  void find_constants(const std::vector<material_interface_type*>& interfaces, const writers_by_material& writers) {
    std::unordered_map<const property_base*, std::pair<const material_interface_type*, const declaration*>> declared;
    for (auto const* mat : interfaces)
      for (auto const& d : mat->constant_outputs())
        declared[d.output] = {mat, &d};
    auto const reads_constant{[&](const input_wire_base* read) {
      if (read->edge_kind() != EdgeKind::Global)
        return false;
      auto* source = find_in_graph(read->source_owner(), read->source_name());
      return source && m_constant.contains(source);
    }};
    auto const is_constant{[&](const property_base* prop) {
      auto const it{declared.find(prop)};
      if (it == declared.end() || prop->is_history())
        return false;
      auto const& [mat, d]{it->second};
      auto const own_constant{[&](const property_base* read) { return m_constant.contains(read); }};
      // a target is constant when everything it holds is: no history, no state, no bare or undecided output
      auto const targets_constant{std::ranges::all_of(mat->material_ref_targets(), [&](std::string const& name) {
        auto const target{std::ranges::find(interfaces, name, &material_interface_type::name)};
        return target == interfaces.end() ||
               (std::ranges::all_of((*target)->all_inputs(), reads_constant) &&
                std::ranges::all_of((*target)->get_property_registry().produced_properties(),
                                    [&](const property_base* out) { return m_constant.contains(out); }));
      })};
      return targets_constant && std::ranges::all_of(d->own_reads, own_constant) &&
             (d->reads ? std::ranges::all_of(*d->reads, reads_constant)
                       : std::ranges::all_of(mat->typed_inputs(), reads_constant, [](auto const& input) { return input.get(); }));
    }};
    // in topological order, so every read is decided before its consumer
    m_constant.clear();
    std::ranges::copy_if(m_property_execution_order, std::inserter(m_constant, m_constant.end()), is_constant);

    m_once.clear();
    m_checks.clear();
    for (auto const* mat : interfaces) {
      auto const& mat_writers{writers.at(mat)};
      auto const writes_constants{[&](const property_base* callback) {
        return std::ranges::all_of(mat_writers, [&](auto const& w) {
          return m_constant.contains(w.first) || !std::ranges::contains(w.second, callback);
        });
      }};
      auto const outputs{mat->constant_outputs() | std::views::filter([&](auto const& d) { return m_constant.contains(d.output); }) |
                         std::views::transform([](auto const& d) { return &d; }) | std::ranges::to<std::vector>()};
      for (auto const* prop : mat->get_property_registry().produced_properties())
        if (prop->traits().update && m_constant.contains(prop) && writes_constants(prop)) {
          m_once.insert(prop);
          m_checks[prop] = {mat, outputs};
        }
    }
    m_callbacks = m_property_execution_order | std::views::filter(has_callback) | std::ranges::to<std::vector>();
    reactivate();
  }

  /// Runs `run` as the update in progress; an exception from it leaves the context failed until revert().
  template <typename Run>
  void guarded(char const* what, Run&& run) {
    if (m_running)
      throw std::logic_error(std::string{what} + "(): called from a callback; a callback reevaluates with update_property()");
    if (m_failed)
      throw std::logic_error(std::string{what} + "(): the last update threw, the context is partly updated: revert() first");
    m_running = true;
    try {
      run();
    } catch (...) {
      m_running = false;
      m_failed = true;
      throw;
    }
    m_running = false;
  }

  /// Every callback runs again, the constant ones once more.
  void reactivate() {
    m_active = m_callbacks;
    m_retired.clear();
  }

  void refresh_constants() {
    if (*m_revision != m_seen_revision) {
      reactivate();
      m_seen_revision = *m_revision;
    }
  }

  /// Moves the once-callbacks that `ran` out of the loop; only after reactivate() is there any left.
  template <typename Ran>
  void retire(Ran const& ran) {
    if (!m_fold || m_retired.size() == m_once.size())
      return;
    auto const done{std::ranges::stable_partition(m_active, [&](property_base* prop) {
      return !(m_once.contains(prop) && ran(prop));
    })};
    m_retired.insert(m_retired.end(), done.begin(), done.end());
    m_active.erase(done.begin(), done.end());
  }

  /// Reruns a constant callback and throws when one of its material's constant outputs changes.
  void verify(property_base* prop) {
    auto const& [mat, outputs]{m_checks.at(prop)};
    auto const checks{outputs | std::views::transform([](auto const* d) { return d->take(); }) | std::ranges::to<std::vector>()};
    prop->traits().update();
    for (auto const& [check, d] : std::views::zip(checks, outputs))
      if (auto const verdict{check()}; verdict != detail::verdict::same)
        throw std::logic_error("constant_output(): '" + mat->name() + "::" + d->output->traits().id.name + "' " +
                               (verdict == detail::verdict::not_finite
                                    ? "is not finite"
                                    : "changed although constant: it reads more than its parameters and declared inputs"));
  }

  /// Wraps every callback in a scope of the inputs it declared (none: untracked), so a nested callback a solver
  /// reruns is checked against its own declaration; throws after a callback that read an undeclared input.
  void track_reads(const std::vector<material_interface_type*>& interfaces) {
    std::unordered_map<const property_base*, std::vector<const input_wire_base*>> declared;
    for (auto const* mat : interfaces)
      for (auto const& d : mat->declared_reads()) {
        auto& allowed{declared[d.callback] = d.inputs};
        for (auto const& target : mat->material_ref_targets())
          if (auto const it{std::ranges::find(interfaces, target, &material_interface_type::name)}; it != interfaces.end())
            std::ranges::copy((*it)->all_inputs(), std::back_inserter(allowed));
      }
    for (auto* prop : m_callbacks) {
      if (!m_tracked.insert(prop).second)
        continue;
      auto const it{declared.find(prop)};
      auto reads{it != declared.end() ? std::optional{std::move(it->second)} : std::nullopt};
      auto& update{prop->traits().update};
      update = [inner = std::move(update), reads = std::move(reads), on = m_check_reads.get(), key = prop_key(prop->traits().id)] {
        detail::read_scope scope{*on && reads ? &*reads : nullptr};
        struct restore {
          detail::read_scope* outer;
          ~restore() { detail::current_reads = outer; }
        } const guard{std::exchange(detail::current_reads, &scope)};
        inner();
        if (scope.undeclared)
          throw std::logic_error("reads(): '" + key + "' reads '" + scope.undeclared->source_owner() + "::" +
                                 scope.undeclared->source_name() + "', an input it did not declare");
      };
    }
  }
  std::unordered_set<const property_base*> m_tracked; ///< callbacks wrapped by track_reads()

  void collect_history_properties(property_handler& properties) {
    m_history_properties = properties.data() | std::views::values | std::views::join | std::views::values |
                           std::views::transform([](auto const& prop) { return prop.get(); }) |
                           std::views::filter([](auto const* prop) { return prop->is_history(); }) | std::ranges::to<std::vector>();
  }

  /// An output filled by another output's callback (an add_callback group, written_by) reruns that callback when
  /// asked for directly (update_source()); the update loop runs only the writer.
  void route_to_writers(const std::vector<material_interface_type*>& interfaces) {
    for (auto const* mat : interfaces)
      for (auto const& [output, by] : mat->written_outputs()) {
        auto* routed{find_in_graph(mat->name(), output->traits().id.name)};
        auto* writer{find_in_graph(mat->name(), by->traits().id.name)};
        m_routed.insert(routed);
        routed->traits().update = [writer] { writer->traits().update(); };
      }
  }
  std::unordered_set<const property_base*> m_routed; ///< outputs whose update reruns their writer

  /// The upstream set of a property, computed once per finalize().
  const std::unordered_set<property_base*>& upstream_of(const property_base* target) const {
    if (auto const it{m_upstream.find(target)}; it != m_upstream.end())
      return it->second;
    return m_upstream.emplace(target, collect_upstream(target)).first->second;
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
  std::vector<graph_edge> m_edges;                                               ///< global edges sorted by key, then local ones sorted by key
  mutable std::unordered_map<const property_base*, std::unordered_set<property_base*>> m_upstream; ///< upstream_of() cache
  std::unordered_set<const property_base*> m_stateful;                           ///< outputs declared stateful()
  std::vector<property_base*> m_history_properties;
  using declaration = typename material_interface_type::constant_declaration;
  std::unordered_set<const property_base*> m_constant;
  std::unordered_set<const property_base*> m_once; ///< callbacks run once
  std::vector<property_base*> m_callbacks;          ///< every callback, in topological order
  std::vector<property_base*> m_active;             ///< ... still to run on every update
  std::vector<property_base*> m_retired;            ///< ... constant and done: verified in debug builds
  std::unique_ptr<std::size_t> m_revision{std::make_unique<std::size_t>(0)}; ///< stable address for the materials
  std::size_t m_seen_revision{0};
  /// once-callback -> its material and that material's constant outputs, for verify()
  std::unordered_map<const property_base*, std::pair<const material_interface_type*, std::vector<const declaration*>>> m_checks;
  bool m_verify{NUMSIM_MATERIALS_DEBUG_CHECKS != 0};
  bool m_fold{true};
  bool m_running{false}; ///< an update is in progress
  bool m_failed{false};  ///< the last update threw
  std::unique_ptr<bool> m_check_reads{std::make_unique<bool>(true)}; ///< stable address for the wrapped callbacks
};

} // namespace numsim::materials

#endif // PROPERTY_ENGINE_H
