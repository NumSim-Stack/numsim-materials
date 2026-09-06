#ifndef ISOTROPIC_TANGENT_H
#define ISOTROPIC_TANGENT_H

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/materials/plasticity_utils.h"

namespace numsim::materials {

/// Isotropic elastic stiffness as a material, with K and G as graph inputs.
///
/// A material reading its OWN property creates no edge, so linear_elasticity's
/// stress sorts before its tangent and would consume a stale stiffness. Reading
/// another material's property is a real edge the sort honours — which is why
/// the moduli are inputs rather than parameters.
///
/// Wire from constant_scalar when fixed, or any scalar producer when they vary;
/// K_property/G_property if it does not publish under "value". Which one you
/// wire IS the choice, so no flag can disagree with it.
///
/// Rebuilds on every update: no memo, so no cached state to go stale. The pair
/// costs ~315 ns per update against ~26 ns for linear_elasticity, which
/// computes its tangent once — so where the moduli are fixed and 12x matters,
/// that is still the cheaper material.
template <typename Traits>
class isotropic_tangent final
    : public material_base<isotropic_tangent<Traits>, Traits> {
public:
  using base = material_base<isotropic_tangent<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit isotropic_tangent(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_C(base::template add_output<tensor4>(
            "tangent", &isotropic_tangent::update_tangent)),
        m_K(base::template add_input<value_type>(
            base::template get_parameter<std::string>("K_source"),
            base::template get_parameter<std::string>("K_property"),
            EdgeKind::Global)),
        m_G(base::template add_input<value_type>(
            base::template get_parameter<std::string>("G_source"),
            base::template get_parameter<std::string>("G_property"),
            EdgeKind::Global)) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("K_source").template add<is_required>();
    para.template insert<std::string>("G_source").template add<is_required>();
    // Defaults to the scalar-output convention constant_scalar and
    // scalar_identity_weight follow. Override for a source that publishes under
    // another name — external_scalar_source publishes "state", for instance.
    para.template insert<std::string>("K_property")
        .template add<set_default>(std::string{"value"});
    para.template insert<std::string>("G_property")
        .template add<set_default>(std::string{"value"});
    return para;
  }

  void update_tangent() {
    const auto I{tmech::eye<value_type, Dim, 2>()};
    const auto IIvol{tmech::otimes(I, I) / Dim};
    m_C = 3 * m_K.get() * IIvol +
          2 * m_G.get() * plasticity_detail::make_IIdev<value_type, Dim>();
  }

private:
  tensor4& m_C;
  const input_property<value_type, property_traits>& m_K;
  const input_property<value_type, property_traits>& m_G;
};

}  // namespace numsim::materials

#endif  // ISOTROPIC_TANGENT_H
