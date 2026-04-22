#ifndef NUMSIM_MATERIALS_IMPLICIT_RK_INTEGRATOR_H
#define NUMSIM_MATERIALS_IMPLICIT_RK_INTEGRATOR_H

#include <cmath>
#include <Eigen/Dense>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/solvers/butcher_tableau.h"

namespace numsim::materials {

/// Fully implicit Runge-Kutta integrator for scalar ODEs.
///
/// All stages are coupled — solves the system simultaneously:
///   k_i = f(y_n + h * Σ_j a[i][j] * k_j)   for all i
///
/// Uses Newton iteration on the full s-dimensional system.
/// Linear system solved via Eigen's LU decomposition.
///
/// The rate function must provide "rate" and "rate_derivative".
template<typename Traits>
class implicit_rk_integrator final
    : public material_base<implicit_rk_integrator<Traits>, Traits> {
public:
  using base = material_base<implicit_rk_integrator<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template<typename... Args>
  explicit implicit_rk_integrator(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_state(base::template add_history_output<value_type>(
            "state", &implicit_rk_integrator::compute)),
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
    const int s = tab.stages;
    const auto y_n = m_state.old_value();

    Eigen::VectorXd k = Eigen::VectorXd::Zero(s);

    // Newton iteration on the coupled system:
    //   R_i(k) = k_i - f(y_n + h * Σ_j a[i][j] * k[j]) = 0
    for (int iter = 0; iter < m_max_iter; ++iter) {
      Eigen::VectorXd R(s);
      Eigen::VectorXd df_val(s);

      for (int i = 0; i < s; ++i) {
        auto y_trial = y_n;
        for (int j = 0; j < s; ++j)
          y_trial += m_h * tab.a[i][j] * k[j];

        m_state.new_value() = y_trial;
        m_rate.update_source();
        R[i] = k[i] - m_rate.get();
        df_val[i] = m_drate.get();
      }

      if (R.lpNorm<Eigen::Infinity>() < m_tol) break;

      // J[i][m] = δ_im - h * a[i][m] * df_val[i]
      Eigen::MatrixXd J = Eigen::MatrixXd::Identity(s, s);
      for (int i = 0; i < s; ++i)
        for (int m = 0; m < s; ++m)
          J(i, m) -= m_h * tab.a[i][m] * df_val[i];

      k -= J.partialPivLu().solve(R);
    }

    auto y_new = y_n;
    for (int i = 0; i < s; ++i)
      y_new += m_h * tab.b[i] * k[i];

    m_state.new_value() = y_new;
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

#endif // NUMSIM_MATERIALS_IMPLICIT_RK_INTEGRATOR_H
