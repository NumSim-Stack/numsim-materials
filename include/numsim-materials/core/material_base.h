#ifndef MATERIAL_BASE_H
#define MATERIAL_BASE_H

#include <deque>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <numsim-core/input_parameter_controller.h>
#include <numsim-core/wrapper.h>

#include "numsim-materials/core/material_interface.h"

namespace numsim::materials {

namespace detail {
template <typename A>
concept pointer_range = std::ranges::range<A const> && std::is_pointer_v<std::ranges::range_value_t<A const>>;
} // namespace detail

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
    // Validation only. Registration in the material handler happens in
    // object_store::create, AFTER the derived constructor has returned -- see
    // the note there.
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

  /// `output` (as add_output returned it) is a function of the parameters and of the material's inputs only:
  /// once they are constant too (over global edges), the engine computes it once instead of on every update.
  template <typename T>
  void constant_output(T const& output) {
    base::declare_constant(&output, std::nullopt);
  }

  /// As above, reading only `reads` of the inputs.
  template <typename T>
  void constant_output(T const& output, std::vector<input_wire_base const*> reads) {
    base::declare_constant(&output, std::move(reads));
  }

  /// A callback bound by add_callback() and the outputs it fills.
  class callback_handle {
  public:
    explicit callback_handle(material_base& owner) : m_owner(owner) {}

    /// The outputs are functions of the parameters and of all inputs, or of `reads` (null ones skipped).
    callback_handle& constant() {
      for (auto const* output : m_outputs)
        m_owner.declare_constant(output, std::nullopt);
      return *this;
    }
    callback_handle& constant(std::vector<input_wire_base const*> const& reads) {
      for (auto const* output : m_outputs)
        m_owner.declare_constant(output, reads);
      return *this;
    }

    /// Runs after the callback of `earlier`.
    callback_handle& runs_after(callback_handle const& earlier) {
      m_owner.declare_order(m_outputs.front(), earlier.m_outputs.front());
      return *this;
    }

  private:
    friend class material_base;
    material_base& m_owner;
    std::vector<void const*> m_outputs;
  };

  /// One callback filling several outputs: references to outputs, pointers to them (null ones skipped) or
  /// ranges of such pointers, none with a callback of its own.
  template <typename... Outputs>
  callback_handle& add_callback(void (Derived::*callback)(), Outputs const&... outputs) {
    auto& handle{m_callbacks.emplace_back(*this)};
    (collect_outputs(handle.m_outputs, outputs), ...);
    if (handle.m_outputs.empty())
      throw std::logic_error("add_callback(): material '" + base::m_name + "' bound a callback to no output");
    for (auto const* output : handle.m_outputs)
      if (base::has_callback(output))
        throw std::logic_error("add_callback(): '" + base::m_name + "::" + base::output_name(output) +
                               "' has a callback already");
    bind_callback(base::output_name(handle.m_outputs.front()), callback);
    for (auto const* output : handle.m_outputs | std::views::drop(1))
      base::declare_written_by(output, handle.m_outputs.front());
    return handle;
  }

  /// `output` has no callback of its own: the callback of `writer` fills it. Lets the engine skip a constant
  /// callback while another callback of the material keeps running.
  template <typename T, typename U>
  void written_by(T const& output, U const& writer) {
    base::declare_written_by(&output, &writer);
  }

  /// The callback of `output` runs after the callback of `earlier` (both outputs of this material).
  template <typename T, typename U>
  void runs_after(T const& output, U const& earlier) {
    base::declare_order(&output, &earlier);
  }

private:
  template <typename A>
  static void collect_outputs(std::vector<void const*>& to, A const& output) {
    if constexpr (std::is_pointer_v<A>) {
      if (output)
        to.push_back(output);
    } else if constexpr (detail::pointer_range<A>) {
      for (auto const* p : output)
        if (p)
          to.push_back(p);
    } else {
      to.push_back(&output);
    }
  }

  std::deque<callback_handle> m_callbacks; ///< stable for the handles add_callback() returns

  /// Bind a member function as the update callback for a property.
  /// The lambda captures a reference to the derived material — the material
  /// must outlive its properties. This is guaranteed by material_context's
  /// member declaration order (m_store destroyed before m_properties).
  void bind_callback(std::string const& property_name, void (Derived::*callback)()) {
    Derived& obj = static_cast<Derived&>(*this);
    if (auto p = base::m_property_handler.find(base::m_name, property_name))
      (*p)->traits().update = [&obj, callback]() { (obj.*callback)(); };
  }
};

} // namespace numsim::materials

#endif // MATERIAL_BASE_H
