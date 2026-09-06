#ifndef PROPS_SCALAR_H
#define PROPS_SCALAR_H

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// A scalar taken from the host's material-constants array on every call.
///
/// constant_scalar bakes its number in at construction, which is right when the
/// constants are fixed for a material name (Abaqus PROPS). This one records only
/// WHICH slot it owns, for hosts whose constants vary — CalculiX interpolates
/// them by temperature.
///
/// Rebuilding the graph per call would also be correct, since nothing is
/// retained between calls, but it measures 22x an evaluation (6.7 us against
/// 302 ns) where reading a slot is free.
///
/// bind() DEREFERENCES and keeps no pointer: a host array may be a per-call
/// temporary, and a kept pointer would read a dead stack slot next call —
/// usually returning the right number, because the slot is commonly reused.
///
/// Plain property, no update callback, as constant_scalar: nothing reaches
/// statev_map, and the value is in place before ctx.update() so ordering
/// cannot matter.
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
        m_index(base::template get_parameter<std::size_t>("index")) {
    // Until the first bind(), rather than whatever the storage held.
    m_value = value_type{};
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::size_t>("index").template add<is_required>();
    return para;
  }

  /// Copy this material's constant out of the host's array.
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

#endif  // PROPS_SCALAR_H
