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
#include <ranges>
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
concept verifiable = std::copy_constructible<T> && !std::is_pointer_v<T> && // a pointer compares by address
                     (std::is_arithmetic_v<T> || requires(T const& a) { tmech::norm(a - a); } || std::equality_comparable<T>);

/// Whether a recomputed value equals the stored one, to sqrt(eps) relative: independent of the units.
template <verifiable T>
bool same_value(T const& a, T const& b) {
  using std::abs;
  if constexpr (std::is_arithmetic_v<T>) {
    return abs(a - b) <= std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(abs(a), abs(b));
  } else if constexpr (requires { tmech::norm(a - b); }) {
    auto const tol{std::sqrt(std::numeric_limits<std::decay_t<decltype(tmech::norm(a))>>::epsilon())};
    return tmech::norm(a - b) <= tol * std::max(tmech::norm(a), tmech::norm(b));
  } else {
    return a == b;
  }
}

/// Whether a value is finite; types without a norm are.
template <verifiable T>
bool is_finite(T const& a) {
  if constexpr (std::is_floating_point_v<T>)
    return std::isfinite(a);
  else if constexpr (!std::is_arithmetic_v<T> && requires { tmech::norm(a - a); })
    return std::isfinite(tmech::norm(a));
  else
    return true;
}

/// A rerun constant compared with its stored value.
enum class verdict { same, changed, not_finite };
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
  /// Copies a value; the returned check compares it with the current one. Empty for types never constant.
  using snapshot = std::function<std::function<detail::verdict()>()>;

  /// An output declared a function of the parameters, of `reads` (all inputs when unset) and of the outputs
  /// `own_reads` of this material (material_base::constant_output, callback_handle::reads).
  struct constant_declaration {
    property_base_type const* output;
    std::optional<std::vector<input_wire_base const*>> reads;
    std::vector<property_base_type const*> own_reads;
    snapshot take;
  };
  const std::vector<constant_declaration>& constant_outputs() const noexcept { return m_constant_outputs; }

  /// A callback with declared reads: the inputs it reads and the outputs it fills (its own first).
  struct read_declaration {
    property_base_type const* callback;
    std::vector<input_wire_base const*> inputs;
    std::vector<property_base_type const*> outputs;
    std::vector<input_wire_base const*> drives; ///< local inputs whose source callbacks it reruns
  };
  const std::vector<read_declaration>& declared_reads() const noexcept { return m_declared_reads; }

  /// The names of the materials this one reaches through material_ref.
  auto material_ref_targets() const {
    return m_material_refs | std::views::transform([](auto const& ref) -> const std::string& { return ref->target_name(); });
  }

  /// (output, writer): the output has no callback of its own, the writer's callback fills it.
  const auto& written_outputs() const noexcept { return m_written_by; }
  /// (earlier, later): callbacks of this material in the order they must run.
  const auto& callback_order() const noexcept { return m_callback_order; }
  /// The outputs declared stateful().
  const auto& stateful_outputs() const noexcept { return m_stateful; }

  /// Throws when an output read by a callback of this material has no writer; at finalize(), when all are bound.
  void check_reads() const {
    for (auto const& [read, reader] : m_callback_order)
      if (std::ranges::contains(m_read_declared, reader) && !read->traits().update &&
          !std::ranges::contains(m_written_by, read, &decltype(m_written_by)::value_type::first))
        throw std::logic_error("reads(): '" + m_name + "::" + read->traits().id.name + "' has no callback or writer to run after");
  }

  /// Gives the callbacks reading every input the inputs the material has now; at finalize(), before the engine reads them.
  void resolve_all_inputs() {
    for (auto& d : m_declared_reads | std::views::filter([this](auto const& d) { return std::ranges::contains(m_reads_all, d.callback); })) {
      d.inputs = all_inputs();
      for (auto& c : m_constant_outputs | std::views::filter([&](auto const& c) { return std::ranges::contains(d.outputs, c.output); }))
        c.reads = d.inputs;
    }
  }

  /// Every input of the material, as a callback reading all of them declares it (callback_handle::reads_all_inputs()).
  std::vector<input_wire_base const*> all_inputs() const {
    return m_typed_inputs | std::views::transform([](auto const& input) -> input_wire_base const* { return input.get(); }) |
           std::ranges::to<std::vector>();
  }

  /// The engine's revision counter, bumped by set_parameter(); bound at finalize().
  void bind_revision(std::size_t* revision) noexcept { m_revision = revision; }

protected:
  template <typename T>
  T& add_property(std::string const& property_name, T&& data = T()) {
    T& value{m_property_handler.template add_property<T>(name(), property_name, std::forward<T>(data))};
    snapshot take;
    if constexpr (detail::verifiable<T>)
      take = [&value] {
        return std::function<detail::verdict()>{[&value, copy = value] {
          using enum detail::verdict;
          return !detail::is_finite(value) ? not_finite : detail::same_value(copy, value) ? same : changed;
        }};
      };
    m_outputs[&value] = {*m_property_handler.find(name(), property_name), std::move(take)};
    return value;
  }

  template <typename T>
  history_property<T>& add_history(std::string const& property_name, T&& data = T()) {
    auto& value{m_property_handler.template add_history<T>(name(), property_name, std::forward<T>(data))};
    m_outputs[&value] = {*m_property_handler.find(name(), property_name), {}}; // never constant
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

  /// The callback filling `outputs` reads every input, as the material has them at finalize().
  void declare_reads_all(std::vector<void const*> const& outputs, bool verifiable) {
    declare_reads(outputs, {}, {}, verifiable);
    m_reads_all.push_back(m_declared_reads.back().callback);
  }

  /// `output` is the value add_property() returned, its address identifies the property.
  void declare_constant(void const* output, std::optional<std::vector<input_wire_base const*>> reads) {
    auto const& entry{output_entry_of("constant_output", output)};
    if (entry.property->is_history())
      throw std::logic_error("constant_output(): '" + key_of(entry) + "' is history, never constant");
    if (!entry.take)
      throw std::logic_error("constant_output(): '" + key_of(entry) + "' has a type that cannot be compared");
    if (reads)
      check_inputs("constant_output", entry, *reads);
    check_undeclared("constant_output", entry);
    m_constant_outputs.push_back({entry.property, std::move(reads), {}, entry.take});
  }

  /// The callback filling `outputs` reads the parameters, `inputs` and the outputs `own` of this material only:
  /// it runs after their writers, and its outputs are constant once all reads are (when they can be constant).
  void declare_reads(std::vector<void const*> const& outputs, std::vector<input_wire_base const*> inputs,
                     std::vector<void const*> const& own, bool verifiable) {
    auto const& callback{output_entry_of("reads", outputs.front())};
    if (std::ranges::contains(m_read_declared, callback.property))
      throw std::logic_error("reads(): '" + key_of(callback) + "' declares its reads twice");
    m_read_declared.push_back(callback.property);
    check_inputs("reads", callback, inputs);
    std::vector<property_base_type const*> own_reads;
    for (auto const* output : own) {
      auto const& read{output_entry_of("reads", output)};
      if (std::ranges::contains(outputs, output))
        throw std::logic_error("reads(): '" + key_of(callback) + "' reads its own output '" + key_of(read) + "'");
      m_callback_order.emplace_back(read.property, callback.property); // after its writer: writer -> read
      own_reads.push_back(read.property);
    }
    auto const entries{outputs | std::views::transform([this](void const* output) { return &output_entry_of("reads", output); }) |
                       std::ranges::to<std::vector>()};
    m_declared_reads.push_back({callback.property, inputs,
                                entries | std::views::transform(&output_entry::property) | std::ranges::to<std::vector>(), {}});
    if (verifiable && std::ranges::none_of(entries, [this](auto const* entry) {
          return entry->property->is_history() || std::ranges::contains(m_stateful, entry->property);
        }))
      for (auto const* entry : entries) {
        check_undeclared("reads", *entry);
        m_constant_outputs.push_back({entry->property, inputs, own_reads, entry->take});
      }
  }

  /// The callback filling `callback` (after its reads()) reruns the source callbacks of the local inputs `driven`.
  void declare_drives(void const* callback, std::vector<input_wire_base const*> driven) {
    auto const& entry{output_entry_of("drives", callback)};
    auto const it{std::ranges::find(m_declared_reads, entry.property, &read_declaration::callback)};
    if (it == m_declared_reads.end())
      throw std::logic_error("drives(): '" + key_of(entry) + "' declares what it drives before its reads");
    if (!it->drives.empty())
      throw std::logic_error("drives(): '" + key_of(entry) + "' declares what it drives twice");
    check_inputs("drives", entry, driven);
    if (!std::ranges::all_of(driven, [&](auto const* input) {
          return input->edge_kind() == EdgeKind::Local && std::ranges::contains(it->inputs, input);
        }))
      throw std::logic_error("drives(): '" + key_of(entry) + "' drives an input it does not read over a local edge");
    it->drives = std::move(driven);
  }

  /// The outputs are never constant: the callback filling them keeps state or has side effects.
  void declare_stateful(std::vector<void const*> const& outputs) {
    for (auto const* output : outputs) {
      auto const* property{output_entry_of("stateful", output).property};
      m_stateful.push_back(property);
      std::erase_if(m_constant_outputs, [property](auto const& d) { return d.output == property; });
    }
  }

  void declare_written_by(void const* output, void const* writer) {
    auto const& out{output_entry_of("written_by", output)};
    auto const& by{output_entry_of("written_by", writer)};
    if (has_callback(out) || has_writer(out))
      throw std::logic_error("written_by(): '" + key_of(out) + "' has a callback or a writer already");
    if (!has_callback(by))
      throw std::logic_error("written_by(): '" + key_of(by) + "' has no callback to write with");
    m_written_by.emplace_back(out.property, by.property);
  }

  const std::string& output_name(void const* output) const {
    return output_entry_of("add_callback", output).property->traits().id.name;
  }
  /// Whether something fills the output already: its own callback or a declared writer.
  bool is_bound(void const* output) const {
    auto const& entry{output_entry_of("add_callback", output)};
    return has_callback(entry) || has_writer(entry);
  }

  void declare_order(void const* later, void const* earlier) {
    auto const& first{output_entry_of("runs_after", earlier)};
    auto const& second{output_entry_of("runs_after", later)};
    for (auto const* entry : {&first, &second})
      if (!has_callback(*entry))
        throw std::logic_error("runs_after(): '" + key_of(*entry) + "' has no callback of its own to order");
    m_callback_order.emplace_back(first.property, second.property);
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
    snapshot take;
  };
  std::unordered_map<void const*, output_entry> m_outputs; ///< value address -> property
  std::vector<constant_declaration> m_constant_outputs;
  std::vector<property_base_type const*> m_read_declared; ///< callbacks with declared reads
  std::vector<read_declaration> m_declared_reads;
  std::vector<property_base_type const*> m_stateful;      ///< outputs never constant
  std::vector<std::pair<property_base_type const*, property_base_type const*>> m_written_by, m_callback_order;

  /// The entry of an output of this material; declarations end at finalize(), when the engine reads them.
  output_entry const& output_entry_of(char const* what, void const* output) const {
    if (m_revision)
      throw std::logic_error(std::string{what} + "(): material '" + m_name + "' declares after finalize()");
    auto const it{m_outputs.find(output)};
    if (it == m_outputs.end())
      throw std::logic_error(std::string{what} + "(): material '" + m_name + "' named a value that is not one of its outputs");
    return it->second;
  }
  void check_inputs(char const* what, output_entry const& entry, std::vector<input_wire_base const*>& reads) const {
    std::erase(reads, nullptr); // an absent optional input
    auto const is_input{[this](input_wire_base const* read) {
      return std::ranges::contains(m_typed_inputs, read, [](auto const& input) { return input.get(); });
    }};
    if (!std::ranges::all_of(reads, is_input))
      throw std::logic_error(std::string{what} + "(): '" + key_of(entry) + "' declares a read that is not an input of the material");
  }
  void check_undeclared(char const* what, output_entry const& entry) const {
    if (std::ranges::contains(m_constant_outputs, entry.property, &constant_declaration::output))
      throw std::logic_error(std::string{what} + "(): '" + key_of(entry) + "' is declared twice");
  }
  std::string key_of(output_entry const& entry) const { return m_name + "::" + entry.property->traits().id.name; }
  static bool has_callback(output_entry const& entry) { return static_cast<bool>(entry.property->traits().update); }
  bool has_writer(output_entry const& entry) const {
    return std::ranges::contains(m_written_by, entry.property, &decltype(m_written_by)::value_type::first);
  }
  std::size_t* m_revision{nullptr};
  std::vector<property_base_type const*> m_reads_all; ///< callbacks reading every input (declare_reads_all)
};

} // namespace numsim::materials

#endif // MATERIAL_INTERFACE_H
