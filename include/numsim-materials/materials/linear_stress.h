#ifndef NUMSIM_MATERIALS_LINEAR_STRESS_H
#define NUMSIM_MATERIALS_LINEAR_STRESS_H

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// sigma = C : eps, with C supplied by another material.
///
/// The decomposed counterpart to `linear_elasticity`: same physics, but the
/// stiffness arrives as a Global input rather than being owned and computed
/// in the constructor. That single change is what makes the ordering correct —
/// consuming another material's property creates a graph edge, so the engine
/// guarantees the stiffness is produced before this material reads it. A
/// material reading its own property has no such edge (see isotropic_tangent).
///
/// Pair it with any tangent generator. `linear_elasticity` remains the better
/// choice whenever the moduli are fixed, since it computes the tangent once and
/// costs one less material in the graph; reach for this pair when the stiffness
/// has to follow something that changes.
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
