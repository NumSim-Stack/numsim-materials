#ifndef NUMSIM_MATERIALS_DIRK_INTEGRATOR_H
#define NUMSIM_MATERIALS_DIRK_INTEGRATOR_H

#include <cmath>
#include <Eigen/Dense>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/solvers/butcher_tableau.h"

namespace numsim::materials {

/// Diagonally Implicit Runge-Kutta (DIRK) integrator for scalar ODEs.
///
/// Each stage with a[i][i] != 0 requires solving:
///   k_i = f(y_n + h * (Σ_{j<i} a[i][j] * k[j] + a[i][i] * k_i))
///
/// Solved via Newton: residual = k_i - f(y_trial(k_i))
///                    jacobian = 1 - h * a[i][i] * df/dy
///
/// The rate function must provide both "rate" (f) and "rate_derivative" (df/dy).
///
/// Outputs:
///   "state" — scalar (history): integrated state y
///
/// Inputs (Local):
///   function_source::rate            — f(y)
///   function_source::rate_derivative — df/dy
template<typename Traits>
class dirk_integrator final
    : public material_base<dirk_integrator<Traits>, Traits> {
public:
  using base = material_base<dirk_integrator<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template<typename... Args>
  explicit dirk_integrator(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_state(base::template add_history_output<value_type>(
            "state", &dirk_integrator::compute)),
        m_h(base::template get_parameter<value_type>("step_size")),
        m_tol(base::template get_parameter<value_type>("tolerance")),
        m_max_iter(base::template get_parameter<int>("max_iter")),
        m_tableau(base::template get_parameter<const butcher_tableau*>("tableau")),
        m_func_name(base::template get_parameter<std::string>("function")),
        m_rate(base::template add_input<value_type>(
            m_func_name, "rate", EdgeKind::Local)),
        m_drate(base::template add_input<value_type>(
            m_func_name, "rate_derivative", EdgeKind::Local))
  {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("function").template add<is_required>();
    para.template insert<value_type>("step_size").template add<is_required>();
    para.template insert<value_type>("tolerance")
        .template add<set_default>(value_type{1e-12});
    para.template insert<int>("max_iter")
        .template add<set_default>(int{50});
    return para;
  }

  void compute() {
    const auto& tab = *m_tableau;
    const int s = tab.stages();
    const auto y_n = m_state.old_value();
    Eigen::VectorXd k = Eigen::VectorXd::Zero(s);

    for (int i = 0; i < s; ++i) {
      auto explicit_sum = tab.a.row(i).head(i).dot(k.head(i));

      if (std::abs(tab.a(i, i)) < 1e-30) {
        // Explicit stage
        m_state.new_value() = y_n + m_h * explicit_sum;
        m_rate.update_source();
        k[i] = m_rate.get();
      } else {
        // Implicit stage: solve k_i = f(y_n + h*(explicit_sum + a[i][i]*k_i))
        k[i] = value_type{0};
        for (int iter = 0; iter < m_max_iter; ++iter) {
          m_state.new_value() = y_n + m_h * (explicit_sum + tab.a(i, i) * k[i]);
          m_rate.update_source();

          auto residual = k[i] - m_rate.get();
          if (std::abs(residual) < m_tol) break;

          auto jacobian = value_type{1} - m_h * tab.a(i, i) * m_drate.get();
          k[i] -= residual / jacobian;
        }
      }
    }

    m_state.new_value() = y_n + m_h * tab.b.dot(k);
  }

private:
  history_property<value_type>& m_state;
  const value_type& m_h;
  const value_type& m_tol;
  const int& m_max_iter;
  const butcher_tableau* m_tableau;
  const std::string& m_func_name;
  const input_property<value_type, property_traits>& m_rate;
  const input_property<value_type, property_traits>& m_drate;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DIRK_INTEGRATOR_H
