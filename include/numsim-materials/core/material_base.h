#ifndef MATERIAL_BASE_H
#define MATERIAL_BASE_H

#include <algorithm>
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

/// The output type behind an add_callback() argument: a reference, a pointer or a range of pointers.
template <typename A>
auto output_of() {
  if constexpr (std::is_pointer_v<A>)
    return std::type_identity<std::remove_cv_t<std::remove_pointer_t<A>>>{};
  else if constexpr (pointer_range<A>)
    return std::type_identity<std::remove_cv_t<std::remove_pointer_t<std::ranges::range_value_t<A const>>>>{};
  else
    return std::type_identity<A>{};
}
template <typename A>
using output_t = typename decltype(output_of<A>())::type;
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
  template <detail::verifiable T>
  void constant_output(T const& output) {
    base::declare_constant(&output, std::nullopt);
  }

  /// As above, reading only `reads` of the inputs.
  template <detail::verifiable T>
  void constant_output(T const& output, std::vector<input_wire_base const*> reads) {
    base::declare_constant(&output, std::move(reads));
  }

  /// A callback bound by add_callback() and the outputs it fills; constant() only when all are verifiable.
  /// Used in the constructor: the declarations it makes live in the material.
  template <bool Verifiable>
  class callback_handle {
  public:
    explicit callback_handle(material_base& owner) : m_owner(owner) {}

    /// The outputs are functions of the parameters and of all inputs: reads() of every input.
    callback_handle& constant()
      requires Verifiable
    {
      m_owner.declare_reads(m_outputs, m_owner.all_inputs(), {}, Verifiable);
      return *this;
    }

    /// The callback reads the parameters and `reads` only: inputs and outputs of this material, as references,
    /// pointers (null ones skipped) or ranges of pointers. It runs after the writers of the outputs it reads;
    /// its outputs are computed once while all reads are constant.
    template <typename... Reads>
    callback_handle& reads(Reads const&... reads) {
      std::vector<input_wire_base const*> inputs;
      std::vector<void const*> outputs;
      (sort_read(inputs, outputs, reads), ...);
      m_owner.declare_reads(m_outputs, std::move(inputs), outputs, Verifiable);
      return *this;
    }

    /// The callback keeps state of its own or acts outside its outputs (it adds to its last value, records a
    /// row): it runs on every update, however constant its reads are.
    callback_handle& stateful() {
      m_owner.declare_stateful(m_outputs);
      return *this;
    }

  private:
    friend class material_base;
    template <typename A>
    static void sort_read(std::vector<input_wire_base const*>& inputs, std::vector<void const*>& outputs, A const& read) {
      if constexpr (std::derived_from<detail::output_t<A>, input_wire_base>)
        collect(inputs, read);
      else
        collect(outputs, read);
    }
    material_base& m_owner;
    std::vector<void const*> m_outputs;
  };

  /// One callback filling several outputs: references to outputs, pointers to them (null ones skipped) or
  /// ranges of such pointers, none with a callback of its own.
  template <typename... Outputs>
  auto add_callback(void (Derived::*callback)(), Outputs const&... outputs) {
    callback_handle<(detail::verifiable<detail::output_t<Outputs>> && ...)> handle{*this};
    (collect(handle.m_outputs, outputs), ...);
    if (handle.m_outputs.empty())
      throw std::logic_error("add_callback(): material '" + base::m_name + "' bound a callback to no output");
    if (auto const bound{std::ranges::find_if(handle.m_outputs, [this](void const* output) { return base::is_bound(output); })};
        bound != handle.m_outputs.end())
      throw std::logic_error("add_callback(): '" + base::m_name + "::" + base::output_name(*bound) +
                             "' has a callback or a writer already");
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
  template <typename P, typename A>
  static void collect(std::vector<P>& to, A const& output) {
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
