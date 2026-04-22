#ifndef NUMSIM_MATERIALS_EXPLICIT_RK_INTEGRATOR_H
#define NUMSIM_MATERIALS_EXPLICIT_RK_INTEGRATOR_H

#include <vector>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/solvers/butcher_tableau.h"

namespace numsim::materials {

/// Explicit Runge-Kutta integrator for scalar ODEs.
///
/// Integrates dy/dt = f(y) using an explicit Butcher tableau.
/// The rate function is a separate material connected via Local edges.
///
/// Outputs:
///   "state" — scalar (history): integrated state y
///
/// Inputs (Local):
///   function_source::rate — f(y) from rate function material
///
/// Parameters:
///   "function"   — name of rate function material
///   "step_size"  — h (required)
///   "tableau"    — butcher_tableau* passed via parameter handler
template<typename Traits>
class explicit_rk_integrator final
    : public material_base<explicit_rk_integrator<Traits>, Traits> {
public:
  using base = material_base<explicit_rk_integrator<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template<typename... Args>
  explicit explicit_rk_integrator(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_state(base::template add_history_output<value_type>(
            "state", &explicit_rk_integrator::compute)),
        m_h(base::template get_parameter<value_type>("step_size")),
        m_tableau(base::template get_parameter<const butcher_tableau*>("tableau")),
        m_func_name(base::template get_parameter<std::string>("function")),
        m_rate(base::template add_input<value_type>(
            m_func_name, "rate", EdgeKind::Local))
  {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("function").template add<is_required>();
    para.template insert<value_type>("step_size").template add<is_required>();
    return para;
  }

  void compute() {
    const auto& tab = *m_tableau;
    const auto y_n = m_state.old_value();
    std::vector<value_type> k(tab.stages);

    for (int i = 0; i < tab.stages; ++i) {
      // y_trial = y_n + h * Σ_{j<i} a[i][j] * k[j]
      auto y_trial = y_n;
      for (int j = 0; j < i; ++j)
        y_trial += m_h * tab.a[i][j] * k[j];

      m_state.new_value() = y_trial;
      m_rate.update_source();
      k[i] = m_rate.get();
    }

    // y_{n+1} = y_n + h * Σ_i b[i] * k[i]
    auto y_new = y_n;
    for (int i = 0; i < tab.stages; ++i)
      y_new += m_h * tab.b[i] * k[i];

    m_state.new_value() = y_new;
  }

private:
  history_property<value_type>& m_state;
  const value_type& m_h;
  const butcher_tableau* m_tableau;
  const std::string& m_func_name;
  const input_property<value_type, property_traits>& m_rate;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_EXPLICIT_RK_INTEGRATOR_H
