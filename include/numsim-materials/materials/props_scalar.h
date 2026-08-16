#ifndef NUMSIM_MATERIALS_PROPS_SCALAR_H
#define NUMSIM_MATERIALS_PROPS_SCALAR_H

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// A scalar taken from the host's material-constants array on every call.
///
/// constant_scalar bakes its number into the graph at construction, which is
/// right when the constants are fixed for a material name — Abaqus PROPS,
/// where two *MATERIAL blocks must have distinct names. This one records only
/// WHICH slot it owns and reads the number from the host each call, for hosts
/// whose constants genuinely vary: CalculiX interpolates the *USER MATERIAL
/// constants by temperature, so they can differ from one call to the next.
///
/// Rebuilding the graph per call would also be correct — nothing is retained
/// between calls — but it measures 22x an evaluation even from a hand-written
/// builder (6.7 us against 302 ns), against nothing at all for reading a slot.
///
/// bind() DEREFERENCES. It never stores the pointer, and that is the whole
/// safety argument: a host constants array may be a per-call temporary, so a
/// kept pointer reads a dead stack slot on the next call — and usually returns
/// the right number anyway, because the slot is commonly reused. That failure
/// survives every test and breaks somewhere else.
///
/// The property is PLAIN and there is NO update callback, exactly as in
/// constant_scalar. Two consequences, both load-bearing: statev_map never sees
/// it, so it costs no STATEV slot; and the value is in place before
/// ctx.update() runs, so where the sort happens to put it cannot matter.
///
/// Parameters:
///   "name":  material name
///   "index": which host constant this publishes, 0-based
template <typename Traits>
class props_scalar final : public material_base<props_scalar<Traits>, Traits> {
public:
  using base = material_base<props_scalar<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit props_scalar(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_value(base::template add_output<value_type>("value")),
        // By value, not by reference into the parameter store: a later insert()
        // can relocate anything past the handler's small buffer.
        m_index(base::template get_parameter<std::size_t>("index")) {
    // Until the first bind(). A model that never binds would otherwise publish
    // whatever the property storage happened to hold.
    m_value = value_type{};
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::size_t>("index").template add<is_required>();
    return para;
  }

  /// Copy this material's constant out of the host's array.
  ///
  /// Callers under the UMAT layer go through material_point_evaluator, which
  /// range-checks every reader once per call and reports a short array as the
  /// setup fault it is. The check here is the backstop for direct C++ use.
  void bind(std::span<const value_type> props) {
    if (m_index >= props.size())
      throw std::out_of_range(
          "props_scalar '" + base::name() + "': wants constant " +
          std::to_string(m_index) + " but only " + std::to_string(props.size()) +
          " were supplied");
    m_value = props[m_index];
  }

  [[nodiscard]] std::size_t index() const noexcept { return m_index; }

private:
  value_type& m_value;
  const std::size_t m_index;
};

}  // namespace numsim::materials

#endif  // NUMSIM_MATERIALS_PROPS_SCALAR_H
