#ifndef PROPERTY_ENGINE_H
#define PROPERTY_ENGINE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <functional>
#include <iterator>
#include <limits>
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
#include <Eigen/Dense>
#include "numsim-materials/core/derivatives.h"
#include "numsim-materials/core/errors.h"
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
    form_groups(interfaces);
    find_constants(interfaces, writers);
    if constexpr (NUMSIM_MATERIALS_DEBUG_CHECKS)
      track_reads(interfaces);
    route_to_writers(interfaces);
    index_partials(interfaces);
    plan_systems();
  }

  /// Run the callbacks still to run, in topological order: a constant one only until it has run. A residual system
  /// (solvers/system.h, material_base::converges) is solved between what it reads and the rest.
  void update() {
    guarded("update", [this] {
      refresh_constants();
      if (m_verify)
        std::ranges::for_each(m_retired, [this](property_base* prop) { verify(prop); });
      for (auto const& s : m_stages)
        s.system ? solve(*s.system, s.callbacks) : run(s.callbacks);
      retire([](property_base*) { return true; });
      m_evaluated = true;
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

  /// d target / d seed as R (rank(target) + rank(seed), a scalar for two scalars), filled by differentiate().
  /// Resolved once: a path from the seed to the target through a callback without the partial it needs throws.
  /// On an evaluated context the callbacks on the path run once more and fill the result at once.
  template <typename R>
    requires derivatives::derivative_value<R, typename Traits::value_type, Traits::Dim>
  const R& derivative(const property_base* target, const property_base* seed) {
    using slot_type = derivatives::derivative_slot<value_type, Traits::Dim, R>;
    auto const what{[&] { return "derivative(): d '" + prop_key(target->traits().id) + "' / d '" + prop_key(seed->traits().id) + "'"; }};
    if (m_running)
      throw std::logic_error(what() + ": called from a callback");
    if (auto const it{std::ranges::find_if(m_plans, [&](auto const& p) { return p.targets.front() == target && p.seed == seed; })};
        it != m_plans.end()) {
      if (auto* slot{dynamic_cast<slot_type*>(it->slot.get())})
        return slot->value;
      throw std::logic_error(what() + " is requested as another type already");
    }
    auto const target_rank{derivatives::property_rank<value_type, Traits::Dim>(*target)};
    auto const seed_rank{derivatives::property_rank<value_type, Traits::Dim>(*seed)};
    if (!target_rank || !seed_rank)
      throw std::logic_error(what() + ": both must be scalars or symmetric tensors of rank 2 or 4");
    if (*target_rank + *seed_rank != derivatives::rank_of<R>())
      throw std::logic_error(std::format("{} has rank {}, the requested type rank {}", what(), *target_rank + *seed_rank,
                                         derivatives::rank_of<R>()));
    auto plan{plan_derivative(target, seed, *seed_rank, what())};
    auto slot{std::make_unique<slot_type>(*target_rank)};
    std::vector<value_type> const zero(derivatives::width<Traits::Dim>(*target_rank) * derivatives::width<Traits::Dim>(*seed_rank));
    slot->unpack(zero.data());
    auto& value{slot->value};
    plan.slot = std::move(slot);
    for (auto const& step : plan.steps)
      step.material->request_partials(step.callback, true);
    m_plans.push_back(std::move(plan));
    reactivate(); // a constant callback fills its partials once more
    if (m_evaluated)
      fill_now(m_plans.back());
    return value;
  }

  /// Whether `seed` is upstream of `target` (or the target itself): a derivative of one by the other can be non-zero.
  /// Without a path derivative() stays zero.
  bool has_path(const property_base* target, const property_base* seed) const {
    return upstream_of(target).contains(const_cast<property_base*>(seed));
  }

  /// Accumulates the requested derivatives from the partials the last update filled: D[out] += P * D[in], through a
  /// residual system by the implicit function theorem.
  void differentiate() {
    if (m_plans.empty())
      return;
    for (auto& [block, packed] : m_blocks)
      block->pack(packed.data());
    for (auto& plan : m_plans)
      finish(plan);
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
      m_evaluated = false;
      throw;
    }
    m_running = false;
  }

  /// Every callback runs again, the constant ones once more.
  void reactivate() {
    m_active = m_callbacks;
    m_retired.clear();
    distribute();
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
    distribute();
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

  using value_type = typename Traits::value_type;
  using matrix = Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>;
  using column_vector = Eigen::Matrix<value_type, Eigen::Dynamic, 1>;
  using tensor2 = tmech::tensor<value_type, Traits::Dim, 2>;
  using system_declaration = typename material_interface_type::system_declaration;
  struct system_stage;

  /// Callbacks run in sequence (no system) or as a residual system's members until it converges.
  struct stage {
    system_stage* system{nullptr};
    std::vector<property_base*> callbacks; ///< the active ones, in topological order
  };

  static void run(std::vector<property_base*> const& callbacks) {
    for (auto* prop : callbacks)
      prop->traits().update();
  }

  /// Throws when tolerance or max_iter (max_passes) of a system is unusable (NaN included).
  static void check_settings(system_stage const& s) {
    auto const converges{s.declared->own_residual != nullptr};
    if (!(*s.declared->tolerance >= 0) || *s.declared->max_iter < (converges ? 1 : 0))
      throw std::invalid_argument(describe(s) + " needs a tolerance >= 0 and " + (converges ? "max_passes >= 1" : "max_iter >= 0"));
  }

  /// Sets the unknowns, reruns the members and gathers the residuals until |R|_inf <= tolerance (a non-finite residual never
  /// converges); max_iter counts the updates of the unknowns, converges() counts passes.
  void solve(system_stage& s, std::vector<property_base*> const& callbacks) {
    auto const& d{*s.declared};
    auto const limit{d.own_residual ? *d.max_iter - 1 : *d.max_iter};
    check_settings(s);
    start(s);
    for (int it{0};; ++it) {
      scatter(s, s.x);
      run(callbacks);
      gather(s, s.R);
      auto const norm{s.R.hasNaN() ? std::numeric_limits<value_type>::quiet_NaN() : s.R.template lpNorm<Eigen::Infinity>()};
      if (std::isfinite(norm) && norm <= *d.tolerance) {
        if (d.report)
          d.report->iterations = it;
        return;
      }
      if (it >= limit || !s.R.allFinite())
        throw cutback_error(d.own_residual ? std::format("'{}' did not reach {:g} in {} pass{} (last {:g})", s.name, *d.tolerance, it + 1,
                                                         it == 0 ? "" : "es", norm)
                                           : std::format("system '{}' did not reach {:g} in {} iteration{} (last |R| {:g})", s.name,
                                                         *d.tolerance, it, it == 1 ? "" : "s", norm));
      step(s, callbacks, it);
    }
  }

  /// One update of the unknowns from the residual at x.
  void step(system_stage& s, std::vector<property_base*> const& callbacks, int it) {
    switch (s.declared->method) {
    case system_method::picard:
      if (s.width > 0)
        s.x -= s.R;
      return;
    case system_method::newton: jacobian(s); break;
    case system_method::newton_fd: jacobian_by_differences(s, callbacks); break;
    }
    s.lu.compute(s.J);
    column_vector const dx{s.lu.solve(s.R)};
    if (!dx.allFinite())
      throw cutback_error(std::format("system '{}': the Jacobian is singular at iteration {}", s.name, it));
    s.x -= dx;
  }

  /// J = dR/dx from the partials the members filled at x.
  void jacobian(system_stage& s) {
    auto& plan{*s.jacobian};
    for (auto const b : plan.blocks)
      m_blocks[b].first->pack(m_blocks[b].second.data());
    accumulate(plan);
    s.J.setZero(s.height, s.width);
    for (std::size_t i{0}; i < s.residuals.size(); ++i)
      if (auto const node{plan.target_nodes[i]})
        s.J.middleRows(s.rows[i], plan.D[*node].rows()) = plan.D[*node];
  }

  /// J by forward differences, one Mandel component of x at a time.
  void jacobian_by_differences(system_stage& s, std::vector<property_base*> const& callbacks) {
    s.J.resize(s.height, s.width);
    column_vector x{s.x}, R(s.height);
    for (Eigen::Index j{0}; j < s.width; ++j) {
      auto const h{std::sqrt(std::numeric_limits<value_type>::epsilon()) * std::max(value_type{1}, std::abs(s.x[j]))};
      x[j] = s.x[j] + h;
      scatter(s, x);
      run(callbacks);
      gather(s, R);
      s.J.col(j) = (R - s.R) / (x[j] - s.x[j]);
      x[j] = s.x[j];
    }
  }

  /// x from the seeds, a history unknown from its value at the end of the last step.
  void start(system_stage& s) {
    s.x.setZero(s.width);
    for (std::size_t i{0}; i < s.unknowns.size(); ++i)
      if (s.unknowns[i]->is_history())
        pack_value(*s.unknowns[i], s.ranks[i], s.x.data() + s.columns[i], true);
      else if (s.ranks[i] == 0)
        s.x[s.columns[i]] = s.declared->seeds[i];
  }

  void scatter(system_stage& s, column_vector const& x) {
    for (std::size_t i{0}; i < s.unknowns.size(); ++i)
      unpack_value(x.data() + s.columns[i], s.ranks[i], *s.unknowns[i]);
  }

  void gather(system_stage& s, column_vector& R) {
    for (std::size_t i{0}; i < s.residuals.size(); ++i)
      pack_value(*s.residuals[i], s.residual_ranks[i], R.data() + s.rows[i], false);
  }

  /// A scalar or a symmetric tensor into its Mandel components; `old`: a history property's old value.
  static void pack_value(property_base& prop, std::size_t rank, value_type* out, bool old) {
    auto const value{[&]<typename V>(std::type_identity<V>) -> V const& {
      if (old)
        return dynamic_cast<numsim_core::history_property<V, property_traits>&>(prop).old_value();
      return derivatives::value_in<V>(prop);
    }};
    if (rank == 0)
      *out = value(std::type_identity<value_type>{});
    else
      derivatives::pack<Traits::Dim, 2>(value(std::type_identity<tensor2>{}), out);
  }

  static void unpack_value(value_type const* in, std::size_t rank, property_base& prop) {
    if (rank == 0)
      derivatives::value_in<value_type>(prop) = *in;
    else
      derivatives::unpack<Traits::Dim, 2>(in, derivatives::value_in<tensor2>(prop));
  }

  /// A declaration with its properties: the unknowns and residuals in Mandel rows and columns.
  std::unique_ptr<system_stage> make_system(const material_interface_type& mat, const system_declaration& d) {
    auto s{std::make_unique<system_stage>()};
    s->declared = &d;
    s->name = d.own_residual ? prop_key(d.own_residual->traits().id) : mat.name();
    auto const rank{[&](property_base* prop) {
      auto const r{derivatives::property_rank<value_type, Traits::Dim>(*prop)};
      if (!r || *r > 2)
        throw std::logic_error(describe(*s) + ": '" + prop_key(prop->traits().id) + "' is neither a scalar nor a symmetric tensor");
      return *r;
    }};
    for (auto const* unknown : d.unknowns) {
      s->unknowns.push_back(find_in_graph(mat.name(), unknown->traits().id.name));
      s->ranks.push_back(rank(s->unknowns.back()));
      s->columns.push_back(s->width);
      s->width += static_cast<Eigen::Index>(derivatives::width<Traits::Dim>(s->ranks.back()));
    }
    if (d.own_residual)
      s->residuals.push_back(find_in_graph(mat.name(), d.own_residual->traits().id.name));
    for (auto const* input : d.residuals)
      s->residuals.push_back(find_in_graph(input->source_owner(), input->source_name()));
    for (auto* residual : s->residuals) {
      s->residual_ranks.push_back(rank(residual));
      s->rows.push_back(s->height);
      s->height += static_cast<Eigen::Index>(derivatives::width<Traits::Dim>(s->residual_ranks.back()));
    }
    s->R.resize(s->height);
    return s;
  }

  static std::string describe(system_stage const& s) {
    return s.declared->own_residual ? "converges(): '" + s.name + "'" : "system '" + s.name + "'";
  }

  /// Per system, by residual order. The members of a system with unknowns are the callbacks downstream of its
  /// unknowns and upstream of its residuals. converges(): the callbacks on a path from the consumer Q of a closed
  /// local edge P->Q (P downstream of Q, upstream of the residual) to the residual; its group must not share a
  /// callback with an earlier one (nested loops). Each stage runs after the callbacks it reads, before the rest;
  /// a system with unknowns shares no member.
  void form_groups(const std::vector<material_interface_type*>& interfaces) {
    m_systems.clear();
    for (auto const* mat : interfaces)
      for (auto const& d : mat->systems())
        m_systems.push_back(make_system(*mat, d));
    auto const position{[this](auto const& s) {
      return std::ranges::min(s->residuals | std::views::transform([this](auto const* r) {
                                return std::ranges::find(m_property_execution_order, r) - m_property_execution_order.begin();
                              }));
    }};
    std::ranges::stable_sort(m_systems, {}, position);

    std::unordered_map<const property_base*, std::vector<const property_base*>> downstream_edges;
    std::unordered_map<const property_base*, std::vector<const property_base*>> writers_of;
    for (auto const& e : m_edges) {
      if (e.kind != edge_kind::local)
        downstream_edges[e.from].push_back(e.to);
      if (e.kind == edge_kind::written_by)
        writers_of[e.to].push_back(e.from);
    }
    // the callbacks that fill a property: itself, or its writers
    auto const callbacks_filling{[&](const property_base* prop) {
      return has_callback(prop) ? std::vector{prop} : writers_of[prop];
    }};
    std::unordered_map<const property_base*, std::unordered_set<const property_base*>> reach_cache;
    auto const reach{[&](const property_base* from) -> const std::unordered_set<const property_base*>& {
      if (auto const it{reach_cache.find(from)}; it != reach_cache.end())
        return it->second;
      std::unordered_set<const property_base*> seen{from};
      std::vector<const property_base*> todo{from};
      while (!todo.empty()) {
        auto const* prop{todo.back()};
        todo.pop_back();
        for (auto const* next : downstream_edges[prop])
          if (seen.insert(next).second)
            todo.push_back(next);
      }
      return reach_cache.emplace(from, std::move(seen)).first->second;
    }};
    auto const callbacks{m_property_execution_order | std::views::filter(has_callback) | std::ranges::to<std::vector>()};
    std::unordered_map<const property_base*, system_stage*> member_of;
    std::vector<std::pair<system_stage*, std::vector<property_base*>>> groups;
    for (auto const& owned_system : m_systems) {
      auto& s{*owned_system};
      auto const converges{s.declared->own_residual != nullptr};
      auto const loop{s.unknowns.empty()}; // converges(), or picard iterating a local loop
      check_settings(s);
      std::unordered_set<const property_base*> upstream;
      for (auto const* residual : s.residuals)
        upstream.insert(upstream_of(residual).begin(), upstream_of(residual).end());
      std::unordered_set<const property_base*> path;
      std::unordered_set<const system_stage*> owners; // earlier groups whose loops this residual also sees
      auto const converging_owner{[&](const property_base* prop) {
        auto const it{member_of.find(prop)};
        return it != member_of.end() && it->second->unknowns.empty() ? it->second : nullptr;
      }};
      if (!loop) {
        std::vector<const property_base*> todo{s.unknowns.begin(), s.unknowns.end()};
        while (!todo.empty()) {
          auto const* prop{todo.back()};
          todo.pop_back();
          for (auto const* next : downstream_edges[prop])
            if (path.insert(next).second)
              todo.push_back(next);
        }
      } else
        for (auto const& e : m_edges) {
          if (e.kind != edge_kind::local)
            continue;
          auto const producers{callbacks_filling(e.from)};
          for (auto const* q : callbacks_filling(e.to)) {
            auto const closing{std::ranges::find_if(producers, [&](auto const* p) { return upstream.contains(p) && reach(q).contains(p); })};
            if (closing == producers.end())
              continue;
            // the callbacks of the loop q ... p
            auto const body{callbacks | std::views::filter([&](auto const* prop) { return reach(q).contains(prop) && reach(prop).contains(*closing); }) |
                            std::ranges::to<std::vector<const property_base*>>()};
            auto const owned{std::ranges::count_if(body, [&](auto const* prop) { return converging_owner(prop) != nullptr; })};
            if (owned == static_cast<std::ptrdiff_t>(body.size())) { // the loop of an earlier group
              owners.insert(converging_owner(body.front()));
              continue;
            }
            if (owned != 0) {
              auto const* other{converging_owner(*std::ranges::find_if(body, [&](auto const* prop) { return converging_owner(prop) != nullptr; }))};
              throw std::logic_error((converges ? "converges(): the loop " : "system '" + s.name + "': the loop ") + prop_key((*closing)->traits().id) +
                                     " -> " + prop_key(q->traits().id) + (converges ? " of '" + s.name + "'" : std::string{}) +
                                     " runs through the iteration group of '" + other->name + "': nested loops are not supported");
            }
            path.insert(reach(q).begin(), reach(q).end());
          }
        }
      std::vector<property_base*> members;
      for (auto* prop : callbacks) {
        if (!upstream.contains(prop) || !path.contains(prop))
          continue;
        if (auto const it{member_of.find(prop)}; it != member_of.end()) {
          if (!s.unknowns.empty() || !it->second->unknowns.empty())
            throw std::logic_error(describe(s) + ": '" + prop_key(prop->traits().id) + "' is a member of " + describe(*it->second) +
                                   " too; systems are disjoint");
          continue;
        }
        if (m_stateful.contains(prop))
          throw std::logic_error(converges ? "converges(): '" + prop_key(prop->traits().id) +
                                                 "' is stateful() and inside the iteration group of '" + s.name +
                                                 "'; a pass must not count or record"
                                           : describe(s) + ": '" + prop_key(prop->traits().id) +
                                                 "' is stateful() and between its unknowns and residuals; a pass must not count or record");
        members.push_back(prop);
      }
      if (members.empty())
        throw std::logic_error(!loop ? describe(s) + ": no callback lies between its unknowns and its residuals"
                                     : describe(s) + " " +
                                                (owners.empty() ? std::string{"has no closed local loop behind it: nothing to iterate (a local edge whose producer is "
                                                                              "downstream of its consumer and upstream of the residual)"}
                                                                : "iterates no loop outside the group of '" + (*owners.begin())->name +
                                                                      "': one residual per group"));
      for (auto* prop : members)
        member_of.emplace(prop, &s);
      for (auto const* unknown : s.unknowns)
        member_of.emplace(unknown, &s); // what reads an unknown waits for its system
      groups.emplace_back(&s, std::move(members));
    }

    // each system after the systems it reads; before it, the callbacks it reads that have not run yet
    m_stages.clear();
    m_stage_of.clear();
    std::unordered_set<const system_stage*> placed;
    auto const place{[&](auto&& props, system_stage* system) {
      for (auto* prop : props)
        m_stage_of[prop] = m_stages.size();
      m_stages.push_back({system, {}});
      if (system)
        placed.insert(system);
    }};
    while (!groups.empty()) {
      auto const reads{[&](auto const& group) {
        std::unordered_set<const property_base*> all;
        for (auto const* member : group.second)
          all.insert(upstream_of(member).begin(), upstream_of(member).end());
        return all;
      }};
      auto const next{std::ranges::find_if(groups, [&](auto const& group) {
        return std::ranges::none_of(reads(group), [&](auto const* prop) {
          auto const it{member_of.find(prop)};
          return it != member_of.end() && it->second != group.first && !placed.contains(it->second);
        });
      })};
      if (next == groups.end())
        throw std::logic_error(groups.front().first->declared->own_residual && groups.back().first->declared->own_residual
                                   ? "converges(): the iteration groups of '" + groups.front().first->name + "' and '" +
                                         groups.back().first->name + "' read each other"
                                   : describe(*groups.front().first) + " and " + describe(*groups.back().first) + " read each other");
      auto const before{reads(*next)};
      place(callbacks | std::views::filter([&](auto const* prop) {
              return before.contains(prop) && !member_of.contains(prop) && !m_stage_of.contains(prop);
            }),
            nullptr);
      place(next->second, next->first);
      groups.erase(next);
    }
    place(callbacks | std::views::filter([&](auto const* prop) { return !m_stage_of.contains(prop); }), nullptr);
  }

  /// The active callbacks into their stages.
  void distribute() {
    for (auto& s : m_stages)
      s.callbacks.clear();
    for (auto* prop : m_active)
      m_stages[m_stage_of.at(prop)].callbacks.push_back(prop);
  }

  std::vector<stage> m_stages;                                      ///< in execution order
  std::unordered_map<const property_base*, std::size_t> m_stage_of; ///< callback -> its stage

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

  // --- derivatives (#217) ---
  using read_declaration = typename material_interface_type::read_declaration;

  struct derivative_step {
    std::size_t out, in; ///< nodes of the plan
    std::size_t block;   ///< in m_blocks
    material_interface_type* material;
    const property_base* callback;
  };
  /// A residual system a derivative passes: its unknowns are seeds in `column`..., its residuals the targets from
  /// `first_target` on; fold() replaces their columns by the implicit function theorem.
  struct system_fold {
    system_stage* system;
    Eigen::Index column;
    std::size_t first_target;
  };
  /// D = d node / d seeds over the nodes between the seeds and the targets that carry one, in topological order.
  struct derivative_plan {
    const property_base* seed{nullptr};       ///< of a requested derivative, whose target is targets.front()
    std::vector<const property_base*> targets;
    std::vector<std::optional<std::size_t>> target_nodes;
    std::vector<std::pair<Eigen::Index, Eigen::Index>> seeds; ///< (first column, width) of node k
    Eigen::Index seed_width{0};                               ///< the columns of `seed`, first
    std::vector<matrix> D;
    std::vector<derivative_step> steps;  ///< in topological order of their outputs
    std::vector<std::size_t> blocks;     ///< the partials the steps read, in m_blocks
    std::vector<system_fold> folds;      ///< in the order of the systems' stages
    std::unique_ptr<derivatives::derivative_slot_base<value_type>> slot;
  };

  /// A declared system as update() solves it, in Mandel components: unknown i in the columns from columns[i],
  /// its residual in the rows from rows[i].
  struct system_stage {
    const system_declaration* declared{nullptr};
    std::string name; ///< the system's, or the residual's for converges()
    std::vector<property_base*> unknowns, residuals;
    std::vector<std::size_t> ranks, residual_ranks;
    std::vector<Eigen::Index> columns, rows;
    Eigen::Index width{0}, height{0};
    column_vector x, R;
    matrix J;
    Eigen::PartialPivLU<matrix> lu;
    std::optional<derivative_plan> jacobian; ///< newton: J from the partials, the unknowns as seeds
  };
  std::vector<std::unique_ptr<system_stage>> m_systems;

  /// D from the packed blocks, folded through the systems it passes, into the slot of a requested derivative.
  void finish(derivative_plan& plan) {
    accumulate(plan);
    fold(plan);
    if (auto const target{plan.target_nodes.front()})
      plan.slot->unpack(plan.D[*target].data());
  }

  /// Reruns the callbacks of a new plan at the values held and accumulates it. A stateful callback is not rerun: its
  /// partials come with the next update().
  void fill_now(derivative_plan& plan) {
    auto const on_plan{[&](auto const* prop) {
      return std::ranges::any_of(plan.steps, [&](auto const& step) { return step.callback == prop; });
    }};
    if (std::ranges::any_of(m_callbacks, [&](auto const* prop) { return on_plan(prop) && m_stateful.contains(prop); }))
      return;
    guarded("derivative", [&] {
      for (auto* prop : m_callbacks | std::views::filter(on_plan))
        prop->traits().update();
    });
    for (auto& [block, packed] : m_blocks)
      block->pack(packed.data());
    finish(plan);
  }

  /// Indexes the declared reads by output and runs every callback declaring partials with want_partials() as requested.
  void index_partials(const std::vector<material_interface_type*>& interfaces) {
    m_plans.clear();
    m_blocks.clear();
    m_read_declarations.clear();
    for (auto* mat : interfaces) {
      for (auto const& d : mat->declared_reads())
        for (auto const* out : d.outputs)
          m_read_declarations.emplace(out, std::pair{mat, &d});
      for (auto const& p : mat->partials())
        if (auto* prop{find_in_graph(mat->name(), p.callback->traits().id.name)}; m_partials_wrapped.insert(prop).second) {
          auto& update{prop->traits().update};
          update = [inner = std::move(update), mat, callback = p.callback] { mat->run_callback(callback, inner); };
        }
    }
  }

  /// Per system, its structural zero blocks (no path from the unknown to the residual) and, for newton, the plan
  /// of J = dR/dx; the members fill their partials on every pass.
  void plan_systems() {
    for (auto& owned : m_systems) {
      auto& s{*owned};
      if (auto* report{s.declared->report}) {
        report->zero_blocks.clear();
        for (std::size_t i{0}; i < s.residuals.size(); ++i)
          for (std::size_t j{0}; j < s.unknowns.size(); ++j)
            if (!upstream_of(s.residuals[i]).contains(s.unknowns[j]))
              report->zero_blocks.emplace_back(i, j);
      }
      if (s.declared->method != system_method::newton)
        continue;
      std::unordered_set<property_base*> up;
      std::vector<std::pair<const property_base*, Eigen::Index>> seeds;
      for (auto const* residual : s.residuals)
        up.insert(upstream_of(residual).begin(), upstream_of(residual).end());
      for (std::size_t j{0}; j < s.unknowns.size(); ++j)
        seeds.emplace_back(s.unknowns[j], static_cast<Eigen::Index>(derivatives::width<Traits::Dim>(s.ranks[j])));
      s.jacobian = plan_columns(std::vector<const property_base*>(s.residuals.begin(), s.residuals.end()), seeds, up, describe(s) + ": its Jacobian");
      for (auto const& step : s.jacobian->steps)
        step.material->request_partials(step.callback, true);
    }
  }

  /// d target / d seed: the systems whose unknowns the target reads (also through another system) add their
  /// unknowns as seeds and their residuals as targets, so fold() can apply the implicit function theorem.
  derivative_plan plan_derivative(const property_base* target, const property_base* seed, std::size_t seed_rank, std::string const& what) {
    auto const seed_width{static_cast<Eigen::Index>(derivatives::width<Traits::Dim>(seed_rank))};
    auto up{upstream_of(target)};
    std::vector<system_stage*> passed;
    for (bool grown{true}; grown;) {
      grown = false;
      for (auto const& st : m_stages) {
        auto* s{st.system};
        if (s && !std::ranges::contains(passed, s) && !std::ranges::contains(s->unknowns, seed) &&
            std::ranges::any_of(s->unknowns, [&](auto* u) { return up.contains(u); })) {
          passed.push_back(s);
          grown = true;
          for (auto const* residual : s->residuals)
            up.insert(upstream_of(residual).begin(), upstream_of(residual).end());
        }
      }
    }
    if (!up.contains(const_cast<property_base*>(seed)))
      passed.clear();
    auto const stage_index{[this](system_stage* s) { return std::ranges::find(m_stages, s, &stage::system) - m_stages.begin(); }};
    std::ranges::sort(passed, {}, stage_index);

    std::vector<const property_base*> targets{target};
    std::vector<std::pair<const property_base*, Eigen::Index>> seeds{{seed, seed_width}};
    std::vector<system_fold> folds;
    auto column{seed_width};
    for (auto* s : passed) {
      folds.push_back({s, column, targets.size()});
      for (std::size_t j{0}; j < s->unknowns.size(); ++j)
        seeds.emplace_back(s->unknowns[j], static_cast<Eigen::Index>(derivatives::width<Traits::Dim>(s->ranks[j])));
      column += s->width;
      targets.insert(targets.end(), s->residuals.begin(), s->residuals.end());
    }
    auto plan{plan_columns(std::move(targets), seeds, up, what)};
    plan.seed = seed;
    plan.seed_width = seed_width;
    plan.folds = std::move(folds);
    return plan;
  }

  /// D = d node / d seeds over `up`: seed k is the identity in its own columns. Every node a seed reaches needs
  /// the partial of each read that carries one, declared or declared zero (independent_of()).
  derivative_plan plan_columns(std::vector<const property_base*> targets, std::vector<std::pair<const property_base*, Eigen::Index>> const& seeds,
                               std::unordered_set<property_base*> const& up, std::string const& what) {
    derivative_plan plan;
    plan.targets = std::move(targets);
    std::unordered_map<const property_base*, std::size_t> node;
    std::unordered_map<const property_base*, const property_base*> via; ///< the read each node is first reached over
    Eigen::Index columns{0};
    for (auto const& [seed, width] : seeds) {
      node.emplace(seed, plan.seeds.size());
      plan.seeds.emplace_back(columns, width);
      columns += width;
    }
    for (auto const& [first, width] : plan.seeds)
      plan.D.emplace_back(width, columns);
    for (auto const* prop : m_property_execution_order | std::views::filter([&](auto* p) { return up.contains(p) && !node.contains(p); })) {
      std::vector<const property_base*> reached;
      if (auto const edges{m_reverse_edges.find(const_cast<property_base*>(prop))}; edges != m_reverse_edges.end())
        std::ranges::copy_if(edges->second, std::back_inserter(reached), [&](auto const* src) { return node.contains(src); });
      if (reached.empty())
        continue;
      auto const declared{m_read_declarations.find(prop)};
      if (declared == m_read_declarations.end())
        throw missing_partial(what, plan.targets, reached.front(), prop, via, "its callback declares no reads");
      auto* mat{declared->second.first};
      auto const& d{*declared->second.second};
      std::vector<derivative_step> steps;
      std::vector<const property_base*> sources;
      Eigen::Index width{0};
      auto const take{[&](input_wire_base const* input, const property_base* own, const property_base* src) {
        sources.push_back(src);
        if (!node.contains(src))
          return;
        auto const same_read{[&](auto const& p) { return p.input == input && p.own == own; }};
        auto const& partials{mat->partials()};
        if (auto const p{std::ranges::find_if(partials, [&](auto const& q) { return q.output == prop && same_read(q); })};
            p != partials.end()) {
          steps.push_back({0, node.at(src), block_of(*p->block), mat, p->callback});
          width = static_cast<Eigen::Index>(p->block->rows());
          via.try_emplace(prop, src);
        } else if (!std::ranges::any_of(mat->independent_reads(), [&](auto const& i) { return i.callback == d.callback && same_read(i); })) {
          throw missing_partial(what, plan.targets, src, prop, via, "its callback declares neither partials() nor independent_of() for that read");
        }
      }};
      for (auto const* input : d.inputs)
        if (input->edge_kind() == EdgeKind::Global)
          if (auto const* src{find_in_graph(input->source_owner(), input->source_name())})
            take(input, nullptr, src);
      for (auto const* own : d.own_reads)
        take(nullptr, own, own);
      for (auto const* src : reached)
        if (!std::ranges::contains(sources, src) && !std::ranges::contains(d.outputs, src))
          throw missing_partial(what, plan.targets, src, prop, via, "it is reached over a '" + std::string{to_string(kind_of(src, prop))} +
                                                                        "' edge, not a declared read, which carries no partials");
      if (steps.empty())
        continue;
      auto const index{plan.D.size()};
      plan.D.emplace_back(width, columns);
      for (auto const& step : steps) {
        plan.steps.push_back({index, step.in, step.block, step.material, step.callback});
        if (!std::ranges::contains(plan.blocks, step.block))
          plan.blocks.push_back(step.block);
      }
      node.emplace(prop, index);
    }
    for (auto const* target : plan.targets) {
      auto const it{node.find(target)};
      plan.target_nodes.push_back(it != node.end() ? std::optional{it->second} : std::nullopt);
    }
    return plan;
  }

  /// D[out] += P * D[in] from the packed partials, the seeds the identity in their columns.
  void accumulate(derivative_plan& plan) {
    for (std::size_t k{0}; k < plan.D.size(); ++k) {
      plan.D[k].setZero();
      if (k < plan.seeds.size())
        plan.D[k].middleCols(plan.seeds[k].first, plan.seeds[k].second).setIdentity();
    }
    for (auto const& step : plan.steps)
      plan.D[step.out].noalias() += m_blocks[step.block].second * plan.D[step.in];
  }

  /// Per system passed, at its converged point R(x, p) = 0: dx/dp = -J^-1 B with J = dR/dx and B = dR/dp taken
  /// with the unknowns as seeds, then every node's D_p += D_x dx/dp.
  static void fold(derivative_plan& plan) {
    auto const p{plan.seed_width};
    for (auto const& f : plan.folds) {
      auto const& s{*f.system};
      matrix J{matrix::Zero(s.height, s.width)}, B{matrix::Zero(s.height, p)};
      for (std::size_t i{0}; i < s.residuals.size(); ++i)
        if (auto const node{plan.target_nodes[f.first_target + i]}) {
          auto const& d{plan.D[*node]};
          J.middleRows(s.rows[i], d.rows()) = d.middleCols(f.column, s.width);
          B.middleRows(s.rows[i], d.rows()) = d.leftCols(p);
        }
      matrix const dx_dp{-J.partialPivLu().solve(B)};
      for (auto& d : plan.D)
        d.leftCols(p).noalias() += d.middleCols(f.column, s.width) * dx_dp;
    }
  }

  /// The packed block of a partial, shared by every plan through it.
  std::size_t block_of(const derivatives::partial_block<value_type>& block) {
    if (auto const it{std::ranges::find(m_blocks, &block, &std::pair<const derivatives::partial_block<value_type>*, matrix>::first)};
        it != m_blocks.end())
      return static_cast<std::size_t>(it - m_blocks.begin());
    m_blocks.emplace_back(&block, matrix(static_cast<Eigen::Index>(block.rows()), static_cast<Eigen::Index>(block.cols())));
    return m_blocks.size() - 1;
  }

  edge_kind kind_of(const property_base* from, const property_base* to) const {
    auto const it{std::ranges::find_if(m_edges, [&](auto const& e) { return e.from == from && e.to == to; })};
    return it != m_edges.end() ? it->kind : edge_kind::dependency;
  }

  /// Names the path seed -> ... -> from -> at -> ... -> target in the error, toward the first target behind `at`.
  std::logic_error missing_partial(std::string const& what, std::vector<const property_base*> const& targets, const property_base* from,
                                   const property_base* at, const std::unordered_map<const property_base*, const property_base*>& via,
                                   const std::string& why) const {
    std::vector<const property_base*> path{from, at};
    for (auto it{via.find(from)}; it != via.end(); it = via.find(it->second))
      path.insert(path.begin(), it->second);
    auto const target{std::ranges::find_if(targets, [&](auto const* t) { return upstream_of(t).contains(const_cast<property_base*>(at)); })};
    if (target != targets.end()) {
      auto const& up{upstream_of(*target)};
      for (auto const* p{at}; p != *target;) { // any edge into the target's upstream leads on to the target
        auto const next{std::ranges::find_if(m_edges, [&](auto const& e) {
          return e.from == p && e.kind != edge_kind::local && up.contains(const_cast<property_base*>(e.to));
        })};
        if (next == m_edges.end())
          break;
        path.push_back(p = next->to);
      }
    }
    std::string text;
    for (auto const* p : path)
      text += (text.empty() ? "" : " -> ") + prop_key(p->traits().id);
    return std::logic_error(what + " passes '" + prop_key(at->traits().id) + "' from '" + prop_key(from->traits().id) + "', but " + why +
                            "; path " + text);
  }

  std::vector<derivative_plan> m_plans;
  std::vector<std::pair<const derivatives::partial_block<value_type>*, matrix>> m_blocks; ///< packed once per differentiate()
  std::unordered_map<const property_base*, std::pair<material_interface_type*, const read_declaration*>> m_read_declarations;
  std::unordered_set<const property_base*> m_partials_wrapped; ///< callbacks wrapped by index_partials()

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
  bool m_evaluated{false}; ///< an update() ran to its end since the last failure
  bool m_failed{false};  ///< the last update threw
  std::unique_ptr<bool> m_check_reads{std::make_unique<bool>(true)}; ///< stable address for the wrapped callbacks
};

} // namespace numsim::materials

#endif // PROPERTY_ENGINE_H
