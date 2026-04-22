#ifndef NUMSIM_MATERIALS_RK_INTEGRATOR_H
#define NUMSIM_MATERIALS_RK_INTEGRATOR_H

#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/solvers/butcher_tableau.h"

namespace numsim::materials {

/// Unified Runge-Kutta integrator for scalar ODEs.
///
/// Handles explicit, DIRK, and fully implicit tableaux automatically.
/// The rate function must provide "rate" and (for implicit stages)
/// "rate_derivative".
///
/// Dispatches at construction time based on tableau structure:
///   - Explicit: sequential rate evaluations, no solver
///   - DIRK: sequential stages, scalar Newton per implicit stage
///   - Fully implicit: coupled Newton system (Eigen LU)
///
/// All working storage pre-allocated. Zero heap allocation per compute().
template<typename Traits>
class rk_integrator final
    : public material_base<rk_integrator<Traits>, Traits> {
public:
  using base = material_base<rk_integrator<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template<typename... Args>
  explicit rk_integrator(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_state(base::template add_history_output<value_type>(
            "state", &rk_integrator::compute)),
        m_h(base::template get_parameter<value_type>("step_size")),
        m_tol(base::template get_parameter<value_type>("tolerance")),
        m_max_iter(base::template get_parameter<int>("max_iter")),
        m_tableau(base::template get_parameter<const butcher_tableau*>("tableau")),
        m_func_name(base::template get_parameter<std::string>("function")),
        m_rate(base::template add_input<value_type>(
            m_func_name, "rate", EdgeKind::Local)),
        m_drate(m_tableau->is_explicit()
            ? nullptr
            : &base::template add_input<value_type>(
                m_func_name, "rate_derivative", EdgeKind::Local)),
        m_k(Eigen::VectorXd::Zero(m_tableau->stages()))
  {
    const int s = m_tableau->stages();
    m_is_explicit = m_tableau->is_explicit();
    m_is_dirk = m_tableau->is_dirk();

    // Pre-compute diagonal properties for DIRK
    m_diag.resize(s);
    m_stage_implicit.resize(s);
    for (int i = 0; i < s; ++i) {
      m_diag[i] = m_tableau->a(i, i);
      m_stage_implicit[i] = std::abs(m_diag[i]) >= 1e-30;
    }

    // Pre-allocate for fully implicit Newton
    if (!m_is_dirk) {
      m_R.resize(s);
      m_df.resize(s);
      m_J.resize(s, s);
    }
  }

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
    if (m_is_explicit)       compute_explicit();
    else if (m_is_dirk)      compute_dirk();
    else                     compute_fully_implicit();
  }

private:
  void compute_explicit() {
    const auto& tab = *m_tableau;
    const int s = tab.stages();
    const auto y_n = m_state.old_value();
    m_k.setZero();

    for (int i = 0; i < s; ++i) {
      auto y_trial = y_n + m_h * tab.a.row(i).head(i).dot(m_k.head(i));
      m_state.new_value() = y_trial;
      m_rate.update_source();
      m_k[i] = m_rate.get();
    }

    m_state.new_value() = y_n + m_h * tab.b.dot(m_k);
  }

  void compute_dirk() {
    const auto& tab = *m_tableau;
    const int s = tab.stages();
    const auto y_n = m_state.old_value();
    m_k.setZero();

    for (int i = 0; i < s; ++i) {
      auto explicit_sum = tab.a.row(i).head(i).dot(m_k.head(i));

      if (!m_stage_implicit[i]) {
        m_state.new_value() = y_n + m_h * explicit_sum;
        m_rate.update_source();
        m_k[i] = m_rate.get();
      } else {
        m_k[i] = value_type{0};
        const auto aii = m_diag[i];
        for (int iter = 0; iter < m_max_iter; ++iter) {
          m_state.new_value() = y_n + m_h * (explicit_sum + aii * m_k[i]);
          m_rate.update_source();

          auto residual = m_k[i] - m_rate.get();
          if (std::abs(residual) < m_tol) break;

          auto jacobian = value_type{1} - m_h * aii * m_drate->get();
          m_k[i] -= residual / jacobian;
        }
      }
    }

    m_state.new_value() = y_n + m_h * tab.b.dot(m_k);
  }

  void compute_fully_implicit() {
    const auto& tab = *m_tableau;
    const int s = tab.stages();
    const auto y_n = m_state.old_value();
    m_k.setZero();

    for (int iter = 0; iter < m_max_iter; ++iter) {
      for (int i = 0; i < s; ++i) {
        m_state.new_value() = y_n + m_h * tab.a.row(i).dot(m_k);
        m_rate.update_source();
        m_R[i] = m_k[i] - m_rate.get();
        m_df[i] = m_drate->get();
      }

      if (m_R.lpNorm<Eigen::Infinity>() < m_tol) break;

      m_J = Eigen::MatrixXd::Identity(s, s) - m_h * m_df.asDiagonal() * tab.a;
      m_k -= m_J.partialPivLu().solve(m_R);
    }

    m_state.new_value() = y_n + m_h * tab.b.dot(m_k);
  }

  history_property<value_type>& m_state;
  const value_type& m_h;
  const value_type& m_tol;
  const int& m_max_iter;
  const butcher_tableau* m_tableau;
  const std::string& m_func_name;
  const input_property<value_type, property_traits>& m_rate;
  const input_property<value_type, property_traits>* m_drate;

  // Pre-allocated working storage
  Eigen::VectorXd m_k;
  Eigen::VectorXd m_R;
  Eigen::VectorXd m_df;
  Eigen::MatrixXd m_J;

  // Pre-computed tableau properties
  bool m_is_explicit;
  bool m_is_dirk;
  std::vector<double> m_diag;
  std::vector<bool> m_stage_implicit;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_RK_INTEGRATOR_H
