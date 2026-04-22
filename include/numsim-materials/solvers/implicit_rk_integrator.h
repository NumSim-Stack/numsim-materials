#ifndef NUMSIM_MATERIALS_IMPLICIT_RK_INTEGRATOR_H
#define NUMSIM_MATERIALS_IMPLICIT_RK_INTEGRATOR_H

#include <cmath>
#include <Eigen/Dense>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/solvers/butcher_tableau.h"

namespace numsim::materials {

/// Fully implicit Runge-Kutta integrator for scalar ODEs.
///
/// All stages are coupled — solves the system simultaneously.
/// Uses Newton iteration with Eigen LU decomposition.
/// All working vectors/matrices pre-allocated — zero heap allocation per compute().
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
            m_func_name, "rate_derivative", EdgeKind::Local)),
        m_k(Eigen::VectorXd::Zero(m_tableau->stages())),
        m_R(m_tableau->stages()),
        m_df(m_tableau->stages()),
        m_J(m_tableau->stages(), m_tableau->stages())
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
    m_k.setZero();

    for (int iter = 0; iter < m_max_iter; ++iter) {
      for (int i = 0; i < s; ++i) {
        m_state.new_value() = y_n + m_h * tab.a.row(i).dot(m_k);
        m_rate.update_source();
        m_R[i] = m_k[i] - m_rate.get();
        m_df[i] = m_drate.get();
      }

      if (m_R.lpNorm<Eigen::Infinity>() < m_tol) break;

      m_J = Eigen::MatrixXd::Identity(s, s) - m_h * m_df.asDiagonal() * tab.a;
      m_k -= m_J.partialPivLu().solve(m_R);
    }

    m_state.new_value() = y_n + m_h * tab.b.dot(m_k);
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

  // Pre-allocated working storage
  Eigen::VectorXd m_k;
  Eigen::VectorXd m_R;
  Eigen::VectorXd m_df;
  Eigen::MatrixXd m_J;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_IMPLICIT_RK_INTEGRATOR_H
