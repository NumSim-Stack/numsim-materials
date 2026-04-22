#ifndef NUMSIM_MATERIALS_IMPLICIT_RK_INTEGRATOR_H
#define NUMSIM_MATERIALS_IMPLICIT_RK_INTEGRATOR_H

#include <cmath>
#include <vector>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/solvers/butcher_tableau.h"

namespace numsim::materials {

/// Fully implicit Runge-Kutta integrator for scalar ODEs.
///
/// All stages are coupled — solves the system simultaneously:
///   k_i = f(y_n + h * Σ_j a[i][j] * k_j)   for all i
///
/// Uses Newton iteration on the full s-dimensional system.
/// For scalar ODEs, this is an s×s dense Newton system.
///
/// Handles any Butcher tableau (explicit, DIRK, fully implicit).
/// For Gauss-Legendre methods, this achieves superconvergence.
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

    // Stage values k[0..s-1], initialized to zero
    std::vector<value_type> k(s, value_type{0});

    // Newton iteration on the coupled system:
    //   R_i(k) = k_i - f(y_n + h * Σ_j a[i][j] * k[j]) = 0
    for (int iter = 0; iter < m_max_iter; ++iter) {
      // Evaluate residuals and collect f values + derivatives
      std::vector<value_type> R(s);
      std::vector<value_type> f_val(s);
      std::vector<value_type> df_val(s);

      for (int i = 0; i < s; ++i) {
        auto y_trial = y_n;
        for (int j = 0; j < s; ++j)
          y_trial += m_h * tab.a[i][j] * k[j];

        m_state.new_value() = y_trial;
        m_rate.update_source();
        f_val[i] = m_rate.get();
        df_val[i] = m_drate.get();
        R[i] = k[i] - f_val[i];
      }

      // Check convergence: max |R_i| < tol
      auto max_r = value_type{0};
      for (int i = 0; i < s; ++i)
        max_r = std::max(max_r, std::abs(R[i]));
      if (max_r < m_tol) break;

      // Build s×s Jacobian: J[i][m] = δ_im - h * a[i][m] * df_val[i]
      // Solve J · dk = -R via Gaussian elimination (small dense system)
      std::vector<std::vector<value_type>> J(s, std::vector<value_type>(s + 1));
      for (int i = 0; i < s; ++i) {
        for (int m = 0; m < s; ++m)
          J[i][m] = (i == m ? value_type{1} : value_type{0})
                    - m_h * tab.a[i][m] * df_val[i];
        J[i][s] = -R[i];  // augmented column
      }

      // Gaussian elimination with partial pivoting
      for (int col = 0; col < s; ++col) {
        // Pivot
        int pivot = col;
        for (int row = col + 1; row < s; ++row)
          if (std::abs(J[row][col]) > std::abs(J[pivot][col]))
            pivot = row;
        std::swap(J[col], J[pivot]);

        auto diag = J[col][col];
        if (std::abs(diag) < value_type{1e-30}) break;

        for (int row = col + 1; row < s; ++row) {
          auto factor = J[row][col] / diag;
          for (int c = col; c <= s; ++c)
            J[row][c] -= factor * J[col][c];
        }
      }

      // Back substitution
      std::vector<value_type> dk(s);
      for (int i = s - 1; i >= 0; --i) {
        dk[i] = J[i][s];
        for (int j = i + 1; j < s; ++j)
          dk[i] -= J[i][j] * dk[j];
        dk[i] /= J[i][i];
      }

      // Update k
      for (int i = 0; i < s; ++i)
        k[i] += dk[i];
    }

    // Final update: y_{n+1} = y_n + h * Σ_i b[i] * k[i]
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
