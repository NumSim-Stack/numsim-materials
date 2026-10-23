#ifndef MATERIAL_CONTEXT_H
#define MATERIAL_CONTEXT_H

#include <cstdint>
#include <fstream>
#include <ranges>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <stdexcept>
#include <unordered_set>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_property_info.h"
#include "numsim-materials/core/object_store.h"
#include "numsim-materials/core/property_engine.h"

namespace numsim::materials {

/// Facade that composes object_store (creation/storage) and property_engine
/// (execution/topo sort). This is the primary user-facing API.
///
/// Usage:
///   material_context<policy> ctx;
///   ctx.create<my_material>(params);        // compile-time
///   ctx.create("material_type", params);    // runtime factory
///   ctx.finalize();                         // wire + sort + validate
///   ctx.update();                           // execute all properties
///   ctx.commit();                           // advance history
template <typename Traits>
class material_context {
public:
  using value_type = typename Traits::value_type;
  using property_handler = typename Traits::PropertyHandler;
  using material_handler = typename Traits::MaterialHandler;
  using parameter_handler = typename Traits::ParameterHandler;
  using material_interface_type = material_interface<Traits>;

  material_context() = default;

  material_context(const material_context&) = delete;
  material_context& operator=(const material_context&) = delete;
  material_context(material_context&&) = delete;
  material_context& operator=(material_context&&) = delete;

  ~material_context() {
    if (!m_finalized) {
      try { finalize(); } catch (...) {}
    }
  }

  // --- Creation (delegated to object_store) ---

  template <typename Material>
  Material& create(parameter_handler& params) {
    check_not_finalized("create");
    return m_store.template create<Material>(params, m_properties, m_materials);
  }

  material_interface_type& create(const std::string& type_name,
                                  parameter_handler& params) {
    check_not_finalized("create");
    return m_store.create(type_name, params, m_properties, m_materials);
  }

  // --- Finalization ---

  void finalize() {
    if (m_finalized) return;
    m_materials.final_queries();
    for (auto* mat : m_store.interfaces())
      mat->wire_materials(m_materials);
    for (auto* mat : m_store.interfaces())
      mat->wire_inputs();
    m_engine.build(m_properties, m_store.interfaces());
    for (auto* mat : m_store.interfaces())
      mat->bind_revision(m_engine.revision());
    check_integrity();
    m_finalized = true;
  }

  [[nodiscard]] bool is_finalized() const noexcept { return m_finalized; }

  // --- Execution (delegated to property_engine) ---

  void update() {
    check_finalized("update");
    m_engine.update();
  }

  void update_property(const std::string& material, const std::string& property,
                       const std::unordered_set<const property_base*>& exclude = {}) {
    check_finalized("update_property");
    m_engine.update_property(material, property, exclude);
  }

  void commit() { m_engine.commit(); }
  void revert() { m_engine.revert(); }

  // --- Checkpoint / Restart ---

  /// Save all history property state to a binary file: per property its name, its byte count and its bytes.
  void save_state(const std::string& path) const {
    check_finalized("save_state");
    std::ofstream out(path, std::ios::binary);
    if (!out)
      throw std::runtime_error("save_state: cannot open '" + path + "'");
    for (auto* prop : m_engine.property_execution_order())
      if (prop->is_history()) {
        std::ostringstream bytes;
        prop->serialize(bytes);
        write_string(out, key_of(*prop));
        write_string(out, bytes.str());
      }
  }

  /// Load history property state from a binary file. Every history property of this context is restored by
  /// name, in any creation order. A property missing on either side throws before anything is restored; one of
  /// another size throws when it is reached.
  void load_state(const std::string& path) {
    check_finalized("load_state");
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw std::runtime_error("load_state: cannot open '" + path + "'");
    std::unordered_map<std::string, std::string> saved;
    while (in.peek() != std::ifstream::traits_type::eof()) {
      auto key{read_string(in, path)};
      saved[std::move(key)] = read_string(in, path);
    }
    std::vector<property_base*> restored;
    std::string missing;
    for (auto* prop : m_engine.property_execution_order())
      if (prop->is_history()) {
        if (saved.contains(key_of(*prop)))
          restored.push_back(prop);
        else
          missing += " '" + key_of(*prop) + "'";
      }
    if (!missing.empty() || restored.size() != saved.size()) {
      std::string unknown;
      for (auto const& key : saved | std::views::keys)
        if (!std::ranges::contains(restored, key, [](auto const* prop) { return key_of(*prop); }))
          unknown += " '" + key + "'";
      throw std::runtime_error("load_state: '" + path + "' does not match this context; missing:" +
                               (missing.empty() ? " none" : missing) + ", unknown here:" + (unknown.empty() ? " none" : unknown));
    }
    for (auto* prop : restored) {
      std::istringstream bytes{saved.at(key_of(*prop))};
      prop->deserialize(bytes);
      if (!bytes || bytes.peek() != std::istringstream::traits_type::eof())
        throw std::runtime_error("load_state: '" + key_of(*prop) + "' in '" + path + "' has another size");
    }
  }

  // --- Lookup ---

  material_interface_type* find(std::string_view name) const noexcept {
    return m_store.find(name);
  }

  property_base* find_property(const std::string& material,
                               const std::string& property) {
    return m_engine.find_property(m_properties, material, property);
  }

  template<typename T>
  const T& get(const std::string& material, const std::string& prop_name) {
    return m_engine.template get<T>(m_properties, material, prop_name);
  }

  /// Mutable access to a property value — bypasses the property graph.
  /// WARNING: Only for diagnostic/privileged code (e.g., numerical_diff_checker).
  /// Writing to a property without triggering update_property leaves the graph
  /// in an inconsistent state. Caller is responsible for restoring consistency.
  template<typename T>
  T& get_mutable(const std::string& material, const std::string& prop_name) {
    return m_engine.template get_mutable<T>(m_properties, material, prop_name);
  }

  // --- Accessors ---

  property_handler& properties() noexcept { return m_properties; }
  const property_handler& properties() const noexcept { return m_properties; }
  material_handler& material_hdl() noexcept { return m_materials; }

  const std::vector<material_interface_type*>& materials() const noexcept {
    return m_store.interfaces();
  }

  const std::vector<material_interface_type*>& execution_order() const noexcept {
    return m_engine.execution_order();
  }

  const std::vector<property_base*>& property_execution_order() const noexcept {
    return m_engine.property_execution_order();
  }

  void dump() const { m_engine.dump(); }

  /// Rerun the constant callbacks on every update and throw when an output changes; on with NUMSIM_MATERIALS_DEBUG_CHECKS.
  void verify_constants(bool on) noexcept { m_engine.verify_constants(on); }

  /// Throw when a callback reads an input it did not declare; compiled in with NUMSIM_MATERIALS_DEBUG_CHECKS.
  void check_reads(bool on) noexcept { m_engine.check_reads(on); }

  /// Run the constant callbacks on every update too: the reference to check folding against (on by default).
  void fold_constants(bool on) noexcept { m_engine.fold_constants(on); }

  /// Whether the engine computes material::property once (after finalize(); false for an unknown property).
  bool is_constant(const std::string& material, const std::string& property) {
    auto const* prop{find_property(material, property)};
    return prop && m_engine.is_constant(prop);
  }

  std::vector<material_property_info> build_property_infos() const {
    std::vector<material_property_info> infos;
    for (auto* mat : m_store.interfaces()) {
      if (auto result = material_property_info::from_material(*mat);
          result.has_value())
        infos.push_back(std::move(result.value()));
    }
    return infos;
  }

private:
  void check_not_finalized(const char* method) const {
    if (m_finalized)
      throw std::logic_error(
          std::string("material_context::") + method + "() called after finalize()");
  }

  void check_finalized(const char* method) const {
    if (!m_finalized)
      throw std::logic_error(
          std::string("material_context::") + method + "() called before finalize()");
  }

  static std::string key_of(const property_base& prop) { return prop.traits().id.owner + "::" + prop.traits().id.name; }
  static void write_string(std::ostream& out, const std::string& text) {
    auto const size{static_cast<std::uint64_t>(text.size())};
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
  }
  static std::string read_string(std::istream& in, const std::string& path) {
    std::uint64_t size{0};
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    std::string text(in ? size : 0, '\0');
    if (in)
      in.read(text.data(), static_cast<std::streamsize>(size));
    if (!in)
      throw std::runtime_error("load_state: '" + path + "' is truncated");
    return text;
  }

  void check_integrity() {
    // Unwired inputs already throw in wire_inputs().
    // Additional checks can be added here.
  }

  // WARNING: Declaration order below is load-bearing — DO NOT REORDER.
  // C++ destroys members in reverse declaration order.
  // m_engine holds raw pointers to materials in m_store.
  // m_store owns materials that hold raw pointers into m_properties.
  // Reordering causes use-after-free during destruction.
  property_handler m_properties;   // 1st: outlives everything
  material_handler m_materials;    // 2nd: query map
  object_store<Traits> m_store;    // 3rd: owns materials → refs into m_properties
  property_engine<Traits> m_engine;// 4th: raw ptrs into m_store
  bool m_finalized{false};
};

} // namespace numsim::materials

#endif // MATERIAL_CONTEXT_H
