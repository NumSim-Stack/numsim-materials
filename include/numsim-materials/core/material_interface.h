#ifndef NUMSIM_MATERIALS_MATERIAL_INTERFACE_H
#define NUMSIM_MATERIALS_MATERIAL_INTERFACE_H

#include <any>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "numsim-materials/core/property_traits.h"
#include "numsim-materials/core/property.h"
#include "numsim-materials/core/history_property.h"
#include "numsim-materials/core/input_types.h"
#include "numsim-materials/core/property_registry_interface.h"
#include "numsim-materials/core/material_ref.h"
#include "numsim-materials/core/traits.h"

namespace numsim::materials {

/// Virtual base for all materials in the framework.
/// Provides property registration, typed input wiring, material references,
/// and the update() entry point.
template <typename Traits>
class material_interface {
public:
  using value_type = typename Traits::value_type;
  using input_parameter_controller = typename Traits::InputParameterController;
  using property_handler = typename Traits::PropertyHandler;
  using material_handler = typename Traits::MaterialHandler;
  using property_registry_type = property_registry_interface<property_handler, property_traits>;
  using parameter_handler = typename Traits::ParameterHandler;

  material_interface(parameter_handler const& param_handler,
                     property_handler& prop_handler)
      : m_parameter_handler(param_handler),
        m_property_handler(prop_handler),
        m_name(m_parameter_handler.template get<std::string>("name")) {}

  virtual ~material_interface() {}
  virtual void update() {}

  const std::string& name() const noexcept { return m_name; }

  template <typename T>
  const T& get_parameter(std::string const& key) const {
    return m_parameter_handler.template get<T>(key);
  }

  template <typename T>
  const T& get_parameter(std::string&& key) const {
    return m_parameter_handler.template get<T>(std::forward<std::string>(key));
  }

  const auto& get_property_registry() const { return m_property_handler; }

  /// Wire all property inputs. Called at finalize().
  void wire_inputs() {
    for (auto& input : m_typed_inputs) {
      auto prop = m_property_handler.find(input->source_owner(), input->source_name());
      if (!prop)
        throw std::runtime_error(
            "wire_inputs(): property '" + input->source_owner() + "::" +
            input->source_name() + "' not found (required by material '" +
            m_name + "')");
      input->wire(**prop);
    }
  }

  /// Wire all material references. Called at finalize() before wire_inputs().
  void wire_materials(material_handler& handler) {
    for (auto& ref : m_material_refs) {
      auto& any_ref = handler.get(ref->target_name());
      auto& mat = std::any_cast<
          std::reference_wrapper<material_interface> const&>(any_ref).get();
      ref->wire(mat);
    }
  }

  const std::vector<std::unique_ptr<input_wire_base>>& typed_inputs() const noexcept {
    return m_typed_inputs;
  }

protected:
  template <typename T>
  T& add_property(std::string const& property_name, T&& data = T()) {
    return m_property_handler.template add_property<T>(
        name(), property_name, std::forward<T>(data));
  }

  template <typename T>
  history_property<T>& add_history(std::string const& property_name, T&& data = T()) {
    return m_property_handler.template add_history<T>(
        name(), property_name, std::forward<T>(data));
  }

  template <typename T>
  input_property<T, property_traits>& add_input(
      std::string source_material, std::string source_property,
      EdgeKind edge_kind = EdgeKind::Global) {
    auto ptr = std::make_unique<input_property<T, property_traits>>(
        std::move(source_material), std::move(source_property), edge_kind);
    auto& ref = *ptr;
    m_typed_inputs.push_back(std::move(ptr));
    return ref;
  }

  template <typename T>
  input_history<T, property_traits>& add_input_history(
      const connection_source& src, EdgeKind edge_kind = EdgeKind::Global) {
    auto ptr = std::make_unique<input_history<T, property_traits>>(
        src.material, src.property, edge_kind);
    auto& ref = *ptr;
    m_typed_inputs.push_back(std::move(ptr));
    return ref;
  }

  /// Add a lazy reference to another material, resolved at finalize().
  template <typename T>
  material_ref<T, Traits>& add_material_ref(std::string name) {
    auto ptr = std::make_unique<material_ref<T, Traits>>(std::move(name));
    auto& ref = *ptr;
    m_material_refs.push_back(std::move(ptr));
    return ref;
  }

  parameter_handler m_parameter_handler;
  property_registry_type m_property_handler;
  std::string m_name;

private:
  std::vector<std::unique_ptr<input_wire_base>> m_typed_inputs;
  std::vector<std::unique_ptr<material_ref_base<Traits>>> m_material_refs;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_MATERIAL_INTERFACE_H
