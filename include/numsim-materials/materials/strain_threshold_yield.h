#ifndef NUMSIM_MATERIALS_STRAIN_THRESHOLD_YIELD_H
#define NUMSIM_MATERIALS_STRAIN_THRESHOLD_YIELD_H

#include <algorithm>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Strain-based yield function for isotropic damage.
///
/// Implements the Kuhn-Tucker condition via:
///   kappa_new = max(eps_eq, kappa_old, kappa_critical)
///   is_yielding = (eps_eq > max(kappa_old, kappa_critical)) ? 1 : 0
///
/// Outputs:
///   "kappa"       — history scalar: damage multiplier (monotonically increasing)
///   "is_yielding" — int: 1 if damage evolves this step, 0 otherwise
///
/// Inputs:
///   state_function::equivalent_strain — scalar from a state function material
template <typename Traits>
class strain_threshold_yield final
    : public material_base<strain_threshold_yield<Traits>, Traits> {
public:
  using base = material_base<strain_threshold_yield<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit strain_threshold_yield(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_kappa(base::template add_history_output<value_type>(
            "kappa", &strain_threshold_yield::compute)),
        m_is_yielding(base::template add_output<int>("is_yielding")),
        m_kappa_critical(base::template get_parameter<value_type>("kappa_critical")),
        m_state_source(base::template get_parameter<std::string>("state_source")),
        m_eps_eq(base::template add_input<value_type>(
            m_state_source, "equivalent_strain", EdgeKind::Global))
  {
    m_kappa.old_value() = m_kappa_critical;
    m_kappa.new_value() = m_kappa_critical;
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("state_source").template add<is_required>();
    para.template insert<value_type>("kappa_critical").template add<is_required>();
    return para;
  }

  void compute() {
    const auto threshold{std::max(m_kappa.old_value(), m_kappa_critical)};
    m_kappa.new_value() = std::max(m_eps_eq.get(), threshold);
    m_is_yielding = (m_eps_eq.get() > threshold) ? 1 : 0;
  }

private:
  history_property<value_type>& m_kappa;
  int& m_is_yielding;
  const value_type& m_kappa_critical;
  const std::string& m_state_source;
  const input_property<value_type, property_traits>& m_eps_eq;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_STRAIN_THRESHOLD_YIELD_H
