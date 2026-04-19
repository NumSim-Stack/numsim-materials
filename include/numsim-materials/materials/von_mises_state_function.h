#ifndef NUMSIM_MATERIALS_VON_MISES_STATE_FUNCTION_H
#define NUMSIM_MATERIALS_VON_MISES_STATE_FUNCTION_H

#include <cmath>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Computes von Mises equivalent strain from a strain tensor.
///
/// Outputs:
///   "equivalent_strain"    — scalar: sqrt(2/3 * dev(eps) : dev(eps))
///   "d_equivalent_strain"  — tensor2: gradient d(eps_eq)/d(eps)
///
/// Inputs:
///   strain_source::strain  — tensor2 strain from a stepper or FEM kernel
template <typename Traits>
class von_mises_state_function final
    : public material_base<von_mises_state_function<Traits>, Traits> {
public:
  using base = material_base<von_mises_state_function<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  static constexpr auto Dim = base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;

  template <typename... Args>
  explicit von_mises_state_function(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_eps_eq(base::template add_output<value_type>(
            "equivalent_strain", &von_mises_state_function::compute)),
        m_deps_eq(base::template add_output<tensor2>(
            "d_equivalent_strain")),
        m_strain_source(base::template get_parameter<std::string>("strain_source")),
        m_strain(base::template add_input<tensor2>(
            m_strain_source, "strain", EdgeKind::Global))
  {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("strain_source").template add<is_required>();
    return para;
  }

  void compute() {
    const auto& eps = m_strain.get();
    const auto I = tmech::eye<value_type, Dim, 2>();
    const auto trace_eps = tmech::trace(eps);
    const auto dev_eps = eps - (trace_eps / value_type{Dim}) * I;
    const auto norm_sq = tmech::dcontract(dev_eps, dev_eps);

    m_eps_eq = std::sqrt(value_type{2.0 / 3.0} * norm_sq);

    if (m_eps_eq > value_type{1e-30}) {
      m_deps_eq = (value_type{2.0 / 3.0} / m_eps_eq) * dev_eps;
    } else {
      m_deps_eq = tensor2{};
    }
  }

private:
  value_type& m_eps_eq;
  tensor2& m_deps_eq;
  const std::string& m_strain_source;
  const input_property<tensor2, property_traits>& m_strain;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_VON_MISES_STATE_FUNCTION_H
