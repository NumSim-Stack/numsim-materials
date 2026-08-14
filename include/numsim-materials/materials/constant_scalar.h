#ifndef NUMSIM_MATERIALS_CONSTANT_SCALAR_H
#define NUMSIM_MATERIALS_CONSTANT_SCALAR_H

#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// A fixed scalar, published as a graph property.
///
/// Lets a quantity that some material derives from — a modulus, a yield stress —
/// be a real graph edge instead of a parameter. The consumer is then ordered
/// after it by the topological sort, which is not true of a parameter: nothing
/// connects a parameter to the material that reads it.
///
/// The property is PLAIN, not history, so statev_map never sees it and it costs
/// no STATEV slot. Use external_scalar_source instead when the value genuinely
/// changes per call and a consumer needs its old/new pair.
///
/// No update callback: the value is set once, at construction, so the engine
/// skips this property entirely.
template <typename Traits>
class constant_scalar final
    : public material_base<constant_scalar<Traits>, Traits> {
public:
  using base = material_base<constant_scalar<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit constant_scalar(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_value(base::template add_output<value_type>("value")) {
    m_value = base::template get_parameter<value_type>("value");
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<value_type>("value").template add<is_required>();
    return para;
  }

private:
  value_type& m_value;
};

}  // namespace numsim::materials

#endif  // NUMSIM_MATERIALS_CONSTANT_SCALAR_H
