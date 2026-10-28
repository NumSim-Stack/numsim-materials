#ifndef STRAIN_ENERGY_STATE_FUNCTION_H
#define STRAIN_ENERGY_STATE_FUNCTION_H

#include <cmath>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

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
        m_eps_eq(base::template add_output<value_type>("equivalent_strain")),
        m_deps_eq(base::template add_output<tensor2>("d_equivalent_strain")),
        m_E(base::template get_parameter<value_type>("youngs_modulus")),
        m_strain_source(base::template get_parameter<std::string>("strain_source")),
        m_tangent_source(base::template get_parameter<std::string>("tangent_source")),
        m_strain(base::template add_input<tensor2>(
            m_strain_source, base::template get_parameter<std::string>("strain_property"), EdgeKind::Global)),
        m_tangent(base::template add_input<tensor4>(
            m_tangent_source, "tangent", EdgeKind::Global))
  {
    base::add_callback(&strain_energy_state_function::compute, m_eps_eq, m_deps_eq)
        .reads(m_strain, m_tangent)
        .partials(m_eps_eq, m_strain, m_tangent);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "strain_source");
    optional<std::string>(para, "strain_property", std::string{"strain"});
    required<std::string>(para, "tangent_source");
    required<value_type>(para, "youngs_modulus");
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
    if (base::want_partials()) // d eps_eq / d C = eps (x) eps / (2 E eps_eq)
      update_partials(eps);
  }

  void update_partials(tensor2 const &eps) {
    base::partial(m_eps_eq, m_strain) = m_deps_eq;
    if (m_eps_eq > value_type{1e-30})
      base::partial(m_eps_eq, m_tangent) = tmech::otimes(eps, eps) / (2 * m_E * m_eps_eq);
    else
      base::partial(m_eps_eq, m_tangent) = tensor4{};
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

#endif // STRAIN_ENERGY_STATE_FUNCTION_H
