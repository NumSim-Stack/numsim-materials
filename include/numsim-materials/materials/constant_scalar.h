#ifndef NUMSIM_MATERIALS_CONSTANT_SCALAR_H
#define NUMSIM_MATERIALS_CONSTANT_SCALAR_H

#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// A fixed scalar, published as a graph property.
///
/// Makes a modulus or yield stress a real edge instead of a parameter, so the
/// consumer is ordered after it. Plain, not history, so it costs no STATEV
/// slot; use external_scalar_source when a consumer needs an old/new pair.
/// No update callback — the value is set once, at construction.
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
