#ifndef NUMSIM_MATERIALS_ISOTROPIC_DAMAGE_H
#define NUMSIM_MATERIALS_ISOTROPIC_DAMAGE_H

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Isotropic damage assembler.
///
/// Computes damaged stress and tangent from undamaged values + damage state:
///   damaged_stress  = (1 - D) * stress
///   damaged_tangent = (1 - D) * tangent - dD * otimes(stress, d_equivalent_strain)
///
/// The second term in the tangent accounts for the softening due to damage
/// evolution (consistent tangent contribution).
///
/// Outputs:
///   "damaged_stress"  — tensor2
///   "damaged_tangent" — tensor4
///
/// Inputs:
///   elastic_source::stress, elastic_source::tangent  — from undamaged material
///   damage_source::damage, damage_source::d_damage   — from propagation law
///   state_source::d_equivalent_strain                — from state function
///   yield_source::is_yielding                        — from yield function
template <typename Traits>
class isotropic_damage final
    : public material_base<isotropic_damage<Traits>, Traits> {
public:
  using base = material_base<isotropic_damage<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  static constexpr auto Dim = base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit isotropic_damage(Args&&... args)
      : base(std::forward<Args>(args)...),
        // outputs
        m_damaged_stress(base::template add_output<tensor2>(
            "damaged_stress", &isotropic_damage::compute)),
        m_damaged_tangent(base::template add_output<tensor4>("damaged_tangent")),
        // parameters
        m_elastic_source(base::template get_parameter<std::string>("elastic_source")),
        m_damage_source(base::template get_parameter<std::string>("damage_source")),
        m_state_source(base::template get_parameter<std::string>("state_source")),
        m_yield_source(base::template get_parameter<std::string>("yield_source")),
        // inputs
        m_stress(base::template add_input<tensor2>(
            m_elastic_source, "stress", EdgeKind::Global)),
        m_tangent(base::template add_input<tensor4>(
            m_elastic_source, "tangent", EdgeKind::Global)),
        m_damage(base::template add_input<value_type>(
            m_damage_source, "damage", EdgeKind::Global)),
        m_d_damage(base::template add_input<value_type>(
            m_damage_source, "d_damage", EdgeKind::Global)),
        m_deps_eq(base::template add_input<tensor2>(
            m_state_source, "d_equivalent_strain", EdgeKind::Global)),
        m_is_yielding(base::template add_input<int>(
            m_yield_source, "is_yielding", EdgeKind::Global))
  {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("elastic_source").template add<is_required>();
    para.template insert<std::string>("damage_source").template add<is_required>();
    para.template insert<std::string>("state_source").template add<is_required>();
    para.template insert<std::string>("yield_source").template add<is_required>();
    return para;
  }

  void compute() {
    const auto D = m_damage.get();
    const auto one_minus_D = value_type{1} - D;
    const auto& sig = m_stress.get();
    const auto& C = m_tangent.get();

    // Damaged stress: (1-D) * sigma
    m_damaged_stress = one_minus_D * sig;

    // Damaged tangent
    if (m_is_yielding.get() != 0) {
      // Yielding: (1-D)*C - dD * sigma ⊗ d_eps_eq
      const auto dD = m_d_damage.get();
      const auto& deps_eq = m_deps_eq.get();
      m_damaged_tangent = one_minus_D * C - dD * tmech::otimes(sig, deps_eq);
    } else {
      // Elastic: (1-D)*C
      m_damaged_tangent = one_minus_D * C;
    }
  }

private:
  tensor2& m_damaged_stress;
  tensor4& m_damaged_tangent;

  const std::string& m_elastic_source;
  const std::string& m_damage_source;
  const std::string& m_state_source;
  const std::string& m_yield_source;

  const input_property<tensor2, property_traits>& m_stress;
  const input_property<tensor4, property_traits>& m_tangent;
  const input_property<value_type, property_traits>& m_damage;
  const input_property<value_type, property_traits>& m_d_damage;
  const input_property<tensor2, property_traits>& m_deps_eq;
  const input_property<int, property_traits>& m_is_yielding;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_ISOTROPIC_DAMAGE_H
