#ifndef NUMSIM_MATERIALS_MATERIAL_BASE_H
#define NUMSIM_MATERIALS_MATERIAL_BASE_H

#include <string>
#include <numsim-core/input_parameter_controller.h>
#include <numsim-core/wrapper.h>

#include "numsim-materials/core/material_interface.h"

namespace numsim::materials {

/// CRTP material base — provides parameter validation, output/history
/// property creation with callback binding, and material registration.
template <typename Derived, typename Traits>
class material_base : public material_interface<Traits> {
public:
  using value_type = typename Traits::value_type;
  using property_handler = typename Traits::PropertyHandler;
  using parameter_handler = typename Traits::ParameterHandler;
  using material_handler = typename Traits::MaterialHandler;
  using input_parameter_controller = typename Traits::InputParameterController;

  static constexpr std::size_t Dim{Traits::Dim};

  using base = material_interface<Traits>;

  material_base() = delete;

  material_base(parameter_handler const& parameter,
                property_handler& property_handler,
                material_handler& material_handler)
      : base(parameter, property_handler),
        m_material_handler(material_handler)
  {
    register_material();
    static auto check_para{Derived::parameters()};
    check_para.check_parameter(base::m_parameter_handler);
  }

  material_base(material_base const&) = delete;
  material_base(material_base&&) = delete;
  virtual ~material_base() = default;
  const material_base& operator=(material_base const&) = delete;

  static inline input_parameter_controller parameters() {
    input_parameter_controller param;
    param.template insert<std::string>("name").template add<is_required>();
    return param;
  }

protected:
  material_handler& m_material_handler;

  /// Create an output property and optionally bind an update callback.
  template<typename T>
  T& add_output(std::string const& property_name,
                void (Derived::*callback)() = nullptr) {
    auto& ref = base::template add_property<T>(property_name);
    if (callback)
      bind_callback(property_name, callback);
    return ref;
  }

  /// Create a history output property and optionally bind an update callback.
  template<typename T>
  history_property<T>& add_history_output(std::string const& property_name,
                                           void (Derived::*callback)() = nullptr,
                                           T initial = T{}) {
    auto& ref = base::template add_history<T>(property_name, std::move(initial));
    if (callback)
      bind_callback(property_name, callback);
    return ref;
  }

private:
  /// Bind a member function as the update callback for a property.
  /// The lambda captures a reference to the derived material — the material
  /// must outlive its properties. This is guaranteed by material_context's
  /// member declaration order (m_store destroyed before m_properties).
  void bind_callback(std::string const& property_name, void (Derived::*callback)()) {
    Derived& obj = static_cast<Derived&>(*this);
    if (auto p = base::m_property_handler.find(base::m_name, property_name))
      (*p)->traits().update = [&obj, callback]() { (obj.*callback)(); };
  }

  void register_material() {
    m_material_handler.set(
        std::ref(static_cast<material_interface<Traits>&>(*this)),
        base::m_name);
  }
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_MATERIAL_BASE_H
