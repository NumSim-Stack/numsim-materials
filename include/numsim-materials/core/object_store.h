#ifndef NUMSIM_MATERIALS_OBJECT_STORE_H
#define NUMSIM_MATERIALS_OBJECT_STORE_H

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "numsim-materials/core/material_base.h"
#include <numsim-core/object_registry.h>

namespace numsim::materials {

/// Owns all objects (materials, solvers, postprocessors — everything is a material).
/// Handles creation (compile-time and factory), storage, and lookup by name.
template<typename Traits>
class object_store {
public:
  using property_handler = typename Traits::PropertyHandler;
  using material_handler = typename Traits::MaterialHandler;
  using parameter_handler = typename Traits::ParameterHandler;
  using material_interface_type = material_interface<Traits>;

  using factory_type = numsim_core::object_registry<
      material_interface_type, parameter_handler, property_handler, material_handler>;

  object_store() = default;
  object_store(const object_store&) = delete;
  object_store& operator=(const object_store&) = delete;
  object_store(object_store&&) = delete;
  object_store& operator=(object_store&&) = delete;

  /// Compile-time construction.
  template <typename Material>
  Material& create(parameter_handler& params,
                   property_handler& properties, material_handler& materials) {
    auto ptr = std::make_unique<Material>(params, properties, materials);
    auto& ref = *ptr;
    m_by_name[ref.name()] = ptr.get();
    m_interfaces.push_back(ptr.get());
    m_storage.push_back(std::move(ptr));
    return ref;
  }

  /// Runtime construction from factory.
  material_interface_type& create(const std::string& type_name,
                                  parameter_handler& params,
                                  property_handler& properties,
                                  material_handler& materials) {
    auto ptr = factory_type::instance().create(
        type_name, params, properties, materials);
    auto& ref = *ptr;
    m_by_name[ref.name()] = ptr.get();
    m_interfaces.push_back(ptr.get());
    m_storage.push_back(std::move(ptr));
    return ref;
  }

  /// Find by name.
  material_interface_type* find(const std::string& name) const noexcept {
    auto it = m_by_name.find(name);
    return it != m_by_name.end() ? it->second : nullptr;
  }

  const std::vector<material_interface_type*>& interfaces() const noexcept {
    return m_interfaces;
  }

private:
  std::vector<std::unique_ptr<material_interface_type>> m_storage;
  std::vector<material_interface_type*> m_interfaces;
  std::unordered_map<std::string, material_interface_type*> m_by_name;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_OBJECT_STORE_H
