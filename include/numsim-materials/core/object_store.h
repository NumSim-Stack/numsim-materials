#ifndef OBJECT_STORE_H
#define OBJECT_STORE_H

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
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

  using input_parameter_controller = typename Traits::InputParameterController;
  using factory_type = numsim_core::object_registry<
      material_interface_type, input_parameter_controller,
      parameter_handler, property_handler, material_handler>;

  object_store() = default;
  object_store(const object_store&) = delete;
  object_store& operator=(const object_store&) = delete;
  object_store(object_store&&) = delete;
  object_store& operator=(object_store&&) = delete;

  /// Compile-time construction.
  template <typename Material>
  Material& create(parameter_handler& params,
                   property_handler& properties, material_handler& materials) {
    check_name_available(params);
    auto ptr = std::make_unique<Material>(params, properties, materials);
    auto& ref = *ptr;
    adopt(std::move(ptr), materials);
    return ref;
  }

  /// Runtime construction from factory.
  material_interface_type& create(const std::string& type_name,
                                  parameter_handler& params,
                                  property_handler& properties,
                                  material_handler& materials) {
    check_name_available(params);
    auto ptr = factory_type::instance().create(
        type_name, params, properties, materials);
    auto& ref = *ptr;
    adopt(std::move(ptr), materials);
    return ref;
  }

  /// Find by name.
  ///
  /// Transparent lookup: hashing a string_view directly keeps this allocation
  /// free, which is what lets it be noexcept. Building a std::string for the
  /// key could throw bad_alloc, and under noexcept that is std::terminate.
  material_interface_type* find(std::string_view name) const noexcept {
    auto it = m_by_name.find(name);
    return it != m_by_name.end() ? it->second : nullptr;
  }

  const std::vector<material_interface_type*>& interfaces() const noexcept {
    return m_interfaces;
  }

private:
  /// Reject a duplicate name before anything is constructed.
  ///
  /// Names are the only way materials address each other -- every input wire,
  /// every material_ref and every property lookup resolves against one. A
  /// second material under a live name used to overwrite the first in the name
  /// index and in the material handler, leaving it owned by m_storage but
  /// unreachable: consumers wired before the collision kept the first,
  /// consumers wired after silently got the second. With mismatched property
  /// types that is not a wrong answer but a wrong pointer, because lookups
  /// resolve through a static_cast -- a consumer expecting a tensor writes
  /// through a pointer to a double. ASan reports a heap-buffer-overflow;
  /// without ASan it is silent.
  ///
  /// Checked here, before construction, rather than in adopt(): a material
  /// registers its properties under its own name while constructing, so by the
  /// time adopt() sees it the collision has already happened in the property
  /// registry.
  void check_name_available(const parameter_handler& params) const {
    const auto& name = params.template get<std::string>("name");
    if (m_by_name.contains(name))
      throw std::invalid_argument(
          "object_store: a material named '" + name + "' already exists. "
          "Material names must be unique -- they are what input wires, "
          "material references and property lookups resolve against.");
  }

  /// Take ownership of a fully constructed material and publish it.
  ///
  /// Registration happens HERE and not in material_base's constructor, because
  /// a constructor that throws must leave nothing behind. material_base used to
  /// register before validating its parameters, and derived constructors can
  /// throw after that anyway -- a missing required parameter, an unknown
  /// tableau name, any deck typo. The material handler holds std::ref, and
  /// query_map has no erase, so a failed construction left a reference to a
  /// destroyed object under a live name, which wire_materials would later find
  /// and dynamic_cast. By the time make_unique returns, the object is complete.
  void adopt(std::unique_ptr<material_interface_type> ptr,
             material_handler& materials) {
    materials.set(std::ref(static_cast<material_interface_type&>(*ptr)),
                  ptr->name());
    m_by_name[ptr->name()] = ptr.get();
    m_interfaces.push_back(ptr.get());
    m_storage.push_back(std::move(ptr));
  }

  /// Lets find() look up by string_view without building a std::string.
  struct transparent_string_hash {
    using is_transparent = void;
    std::size_t operator()(std::string_view s) const noexcept {
      return std::hash<std::string_view>{}(s);
    }
  };

  std::vector<std::unique_ptr<material_interface_type>> m_storage;
  std::vector<material_interface_type*> m_interfaces;
  std::unordered_map<std::string, material_interface_type*,
                     transparent_string_hash, std::equal_to<>> m_by_name;
};

} // namespace numsim::materials

#endif // OBJECT_STORE_H
