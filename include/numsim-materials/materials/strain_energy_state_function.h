#ifndef NUMSIM_MATERIALS_STRAIN_ENERGY_STATE_FUNCTION_H
#define NUMSIM_MATERIALS_STRAIN_ENERGY_STATE_FUNCTION_H

#include <cmath>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Strain energy based equivalent strain.
///
///   eps_eq = sqrt(eps : C : eps / E)
///   d(eps_eq)/d(eps) = (C : eps) / (E * eps_eq)
///
/// Requires tangent from an elastic material as input.
template <typename Traits>
class strain_energy_state_function final
    : public material_base<strain_energy_state_function<Traits>, Traits> {
public:
  using base = material_base<strain_energy_state_function<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  static constexpr auto Dim = base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit strain_energy_state_function(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_eps_eq(base::template add_output<value_type>(
            "equivalent_strain", &strain_energy_state_function::compute)),
        m_deps_eq(base::template add_output<tensor2>("d_equivalent_strain")),
        m_E(base::template get_parameter<value_type>("youngs_modulus")),
        m_strain_source(base::template get_parameter<std::string>("strain_source")),
        m_tangent_source(base::template get_parameter<std::string>("tangent_source")),
        m_strain(base::template add_input<tensor2>(
            m_strain_source, "strain", EdgeKind::Global)),
        m_tangent(base::template add_input<tensor4>(
            m_tangent_source, "tangent", EdgeKind::Global))
  {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("strain_source").template add<is_required>();
    para.template insert<std::string>("tangent_source").template add<is_required>();
    para.template insert<value_type>("youngs_modulus").template add<is_required>();
    return para;
  }

  void compute() {
    const auto& eps = m_strain.get();
    const auto& C = m_tangent.get();
    const auto C_eps = tmech::dcontract(C, eps);
    const auto energy = tmech::dcontract(eps, C_eps);

    m_eps_eq = std::sqrt(energy / m_E);

    if (m_eps_eq > value_type{1e-30}) {
      m_deps_eq = C_eps / (m_E * m_eps_eq);
    } else {
      m_deps_eq = tensor2{};
    }
  }

private:
  value_type& m_eps_eq;
  tensor2& m_deps_eq;
  const value_type& m_E;
  const std::string& m_strain_source;
  const std::string& m_tangent_source;
  const input_property<tensor2, property_traits>& m_strain;
  const input_property<tensor4, property_traits>& m_tangent;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_STRAIN_ENERGY_STATE_FUNCTION_H
