#ifndef MATERIAL_CONTEXT_H
#define MATERIAL_CONTEXT_H

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
      mat->wire_inputs();
    m_engine.build(m_properties, m_store.interfaces());
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
