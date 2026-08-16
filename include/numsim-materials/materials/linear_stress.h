#ifndef NUMSIM_MATERIALS_LINEAR_STRESS_H
#define NUMSIM_MATERIALS_LINEAR_STRESS_H

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// sigma = C : eps, with C supplied by another material.
///
/// linear_elasticity with the tangent as a Global input instead of an owned
/// output. That is what makes the ordering correct: consuming another
/// material's property creates a graph edge; reading your own does not.
///
/// Pair with any tangent generator. linear_elasticity stays the better choice
/// when the moduli are fixed — one fewer material, tangent computed once.
template <typename Traits>
class linear_stress final
    : public material_base<linear_stress<Traits>, Traits> {
public:
  using base = material_base<linear_stress<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit linear_stress(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_sig(base::template add_output<tensor2>(
            "stress", &linear_stress::update_stress)),
        m_C(base::template add_input<tensor4>(
            base::template get_parameter<std::string>("tangent_source"),
            "tangent", EdgeKind::Global)),
        m_eps(base::template add_input<tensor2>(
            base::template get_parameter<std::string>("strain_source"),
            "strain", EdgeKind::Global)) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("tangent_source")
        .template add<is_required>();
    para.template insert<std::string>("strain_source")
        .template add<is_required>();
    return para;
  }

  void update_stress() { m_sig = tmech::dcontract(m_C.get(), m_eps.get()); }

private:
  tensor2& m_sig;
  const input_property<tensor4, property_traits>& m_C;
  const input_property<tensor2, property_traits>& m_eps;
};

}  // namespace numsim::materials

#endif  // NUMSIM_MATERIALS_LINEAR_STRESS_H
