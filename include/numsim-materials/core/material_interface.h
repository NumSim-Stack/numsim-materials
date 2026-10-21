#ifndef MATERIAL_INTERFACE_H
#define MATERIAL_INTERFACE_H

#include <algorithm>
#include <any>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <stdexcept>
#include <type_traits>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#include <tmech/tmech.h>

#include "numsim-materials/core/property_traits.h"
#include "numsim-materials/core/property.h"
#include "numsim-materials/core/history_property.h"
#include "numsim-materials/core/input_types.h"
#include "numsim-materials/core/property_registry_interface.h"
#include "numsim-materials/core/material_ref.h"
#include "numsim-materials/core/traits.h"

namespace numsim::materials {

namespace detail {
/// How a parameter type reads in an error message.
template <typename T>
std::string parameter_type_label() {
  if constexpr (std::is_same_v<T, bool>) return "a boolean";
  else if constexpr (std::is_floating_point_v<T>) return "a real number";
  else if constexpr (std::is_integral_v<T>) return "an integer";
  else if constexpr (std::is_same_v<T, std::string>) return "a string";
  else if constexpr (requires { typename T::value_type; typename T::iterator; }) return "a list";
  else return std::string{"of type "} + typeid(T).name();
}

/// Types whose recomputed value same_value() can compare with the stored one; only these can be constant.
template <typename T>
concept verifiable = std::copy_constructible<T> &&
                     (std::is_arithmetic_v<T> || requires(T const& a) { tmech::norm(a - a); } || std::equality_comparable<T>);

/// Whether a recomputed value equals the stored one (to sqrt(eps) relative).
template <verifiable T>
bool same_value(T const& a, T const& b) {
  using std::abs;
  if constexpr (std::is_arithmetic_v<T>) {
    return abs(a - b) <= std::sqrt(std::numeric_limits<T>::epsilon()) * std::max({T{1}, abs(a), abs(b)});
  } else if constexpr (requires { tmech::norm(a - b); }) {
    auto const tol{std::sqrt(std::numeric_limits<std::decay_t<decltype(tmech::norm(a))>>::epsilon())};
    return tmech::norm(a - b) <= tol * std::max({decltype(tol){1}, tmech::norm(a), tmech::norm(b)});
  } else {
    return a == b;
  }
}
} // namespace detail

/// Virtual base for all materials: property registration, typed input wiring and material references;
/// work happens in the callbacks bound to properties, there is no virtual update() to override.
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

  virtual ~material_interface() = default;

  const std::string& name() const noexcept { return m_name; }

  /// Reads a parameter; a missing key or a wrong stored type is reported with the material, the key and the
  /// type this material reads (not as a bare bad_any_cast or the handler's "Key not found").
  template <typename T>
  const T& get_parameter(std::string const& key) const {
    try {
      return m_parameter_handler.template get<T>(key);
    } catch (const std::bad_any_cast&) {
      throw std::invalid_argument("material '" + m_name + "': parameter '" + key + "' is not " + detail::parameter_type_label<T>() +
                                  " as this material reads it (a whole number written without a decimal point is the usual cause)");
    } catch (const std::invalid_argument&) {
      throw std::invalid_argument("material '" + m_name + "': parameter '" + key + "' is missing");
    }
  }

  /// Overwrite a parameter in place so references bound by get_parameter() stay
  /// valid and see the new value. Assigns through the non-const get<T>(), not
  /// insert(): insert_or_assign replaces the whole std::any and relocates
  /// anything past its small buffer.
  ///
  /// T is not deduced -- set_parameter<double>("K", 250) compiles,
  /// set_parameter("K", 250) does not, where deducing int lost the write to a
  /// run-time bad_any_cast.
  ///
  /// Reaches only what the material re-reads through its bound reference: not
  /// state derived at construction, copied into a member, or consumed once for
  /// wiring, and not other materials, which hold their own handler copy.
  template <typename T>
  void set_parameter(std::string const& key,
                     std::type_identity_t<T> const& value) {
    // The identity is cached in m_name and used as the registry key, so a write
    // here would leave the parameter disagreeing with both.
    if (key == "name")
      throw std::invalid_argument(
          "material_interface::set_parameter(): 'name' is the material's "
          "identity and cannot be changed after construction");
    try {
      m_parameter_handler.template get<T>(key) = value;
      if (m_revision)
        ++*m_revision; // the engine recomputes its constant properties
    } catch (const std::bad_any_cast&) {
      // Otherwise this surfaces as a bare "bad any_cast" naming nothing, and
      // from a type that is neither invalid_argument nor runtime_error.
      throw std::invalid_argument(
          "material_interface::set_parameter('" + key +
          "'): the requested type does not match the stored one");
    }
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
  /// Collects ALL missing materials and reports them in one error.
  void wire_materials(material_handler& handler) {
    std::vector<std::string> missing;
    std::vector<std::string> wrong_type;
    for (auto& ref : m_material_refs) {
      // Look-up and type-check are separate failures and must stay separate.
      // A single catch(...) around both reported a wrongly-typed material as
      // MISSING, so swapping a solver for one of another type told the user
      // their solver did not exist.
      material_interface* target = nullptr;
      try {
        auto& any_ref = handler.get(ref->target_name());
        target = &std::any_cast<
            std::reference_wrapper<material_interface> const&>(any_ref).get();
      } catch (...) {
        missing.push_back(ref->target_name());
        continue;
      }
      try {
        ref->wire(*target);
      } catch (...) {
        wrong_type.push_back(ref->target_name());
      }
    }
    if (!missing.empty() || !wrong_type.empty()) {
      std::string msg = "wire_materials(): material '" + m_name + "'";
      if (!missing.empty()) {
        msg += " references materials that do not exist:";
        for (auto& name : missing) msg += " '" + name + "'";
      }
      if (!wrong_type.empty()) {
        if (!missing.empty()) msg += ";";
        msg += " references materials of the wrong type:";
        for (auto& name : wrong_type) msg += " '" + name + "'";
        msg += " (they exist, but are not the type this material requires)";
      }
      throw std::runtime_error(msg);
    }
  }

  const std::vector<std::unique_ptr<input_wire_base>>& typed_inputs() const noexcept {
    return m_typed_inputs;
  }

  using property_base_type = numsim_core::property_base<property_traits>;
  /// Copies a value; the returned check tells whether it is still the same. Empty for types never constant.
  using snapshot = std::function<std::function<bool()>()>;

  /// An output declared a function of the parameters and of `reads`, all inputs when unset
  /// (material_base::constant_output).
  struct constant_declaration {
    property_base_type const* output;
    std::string name;
    std::optional<std::vector<input_wire_base const*>> reads;
    snapshot take;
  };
  const std::vector<constant_declaration>& constant_outputs() const noexcept { return m_constant_outputs; }

  /// (output, writer): the output has no callback of its own, the writer's callback fills it.
  const auto& written_outputs() const noexcept { return m_written_by; }
  /// (earlier, later): callbacks of this material in the order they must run.
  const auto& callback_order() const noexcept { return m_callback_order; }

  /// The engine's revision counter, bumped by set_parameter(); bound at finalize().
  void bind_revision(std::size_t* revision) noexcept { m_revision = revision; }

protected:
  template <typename T>
  T& add_property(std::string const& property_name, T&& data = T()) {
    T& value{m_property_handler.template add_property<T>(name(), property_name, std::forward<T>(data))};
    snapshot take;
    if constexpr (detail::verifiable<T>)
      take = [&value] { return std::function<bool()>{[&value, copy = value] { return detail::same_value(copy, value); }}; };
    m_outputs[&value] = {*m_property_handler.find(name(), property_name), property_name, std::move(take)};
    return value;
  }

  template <typename T>
  history_property<T>& add_history(std::string const& property_name, T&& data = T()) {
    auto& value{m_property_handler.template add_history<T>(name(), property_name, std::forward<T>(data))};
    m_outputs[&value] = {*m_property_handler.find(name(), property_name), property_name, {}}; // never constant
    return value;
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

  /// `output` is the value add_property() returned, its address identifies the property.
  void declare_constant(void const* output, std::optional<std::vector<input_wire_base const*>> reads) {
    auto const it{output_entry_of("constant_output", output)};
    if (it->second.property->is_history())
      throw std::logic_error("constant_output(): '" + m_name + "::" + it->second.name + "' is history, never constant");
    if (!it->second.take)
      throw std::logic_error("constant_output(): '" + m_name + "::" + it->second.name + "' has a type that cannot be compared");
    if (reads)
      std::erase(*reads, nullptr); // an absent optional input
    if (reads)
      for (auto const* read : *reads)
        if (std::ranges::none_of(m_typed_inputs, [read](auto const& input) { return input.get() == read; }))
          throw std::logic_error("constant_output(): '" + m_name + "::" + it->second.name +
                                 "' declares a read that is not an input of the material");
    m_constant_outputs.push_back({it->second.property, it->second.name, std::move(reads), it->second.take});
  }

  void declare_written_by(void const* output, void const* writer) {
    auto const out{output_entry_of("written_by", output)};
    auto const by{output_entry_of("written_by", writer)};
    if (out->second.property->traits().update)
      throw std::logic_error("written_by(): '" + m_name + "::" + out->second.name + "' has a callback of its own");
    if (!by->second.property->traits().update)
      throw std::logic_error("written_by(): '" + m_name + "::" + by->second.name + "' has no callback to write with");
    m_written_by.emplace_back(out->second.property, by->second.property);
  }

  const std::string& output_name(void const* output) const { return output_entry_of("add_callback", output)->second.name; }
  bool has_callback(void const* output) const {
    return static_cast<bool>(output_entry_of("add_callback", output)->second.property->traits().update);
  }

  void declare_order(void const* later, void const* earlier) {
    m_callback_order.emplace_back(output_entry_of("runs_after", earlier)->second.property,
                                  output_entry_of("runs_after", later)->second.property);
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
  struct output_entry {
    property_base_type const* property;
    std::string name;
    snapshot take;
  };
  std::unordered_map<void const*, output_entry> m_outputs; ///< value address -> property
  std::vector<constant_declaration> m_constant_outputs;
  std::vector<std::pair<property_base_type const*, property_base_type const*>> m_written_by, m_callback_order;

  auto output_entry_of(char const* what, void const* output) const {
    auto const it{m_outputs.find(output)};
    if (it == m_outputs.end())
      throw std::logic_error(std::string{what} + "(): material '" + m_name + "' named a value that is not one of its outputs");
    return it;
  }
  std::size_t* m_revision{nullptr};
};

} // namespace numsim::materials

#endif // MATERIAL_INTERFACE_H
