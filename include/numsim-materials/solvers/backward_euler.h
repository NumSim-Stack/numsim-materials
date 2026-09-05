#ifndef NUMSIM_MATERIALS_BACKWARD_EULER_H
#define NUMSIM_MATERIALS_BACKWARD_EULER_H

#include <cmath>
#include <string>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/solvers/newton_scalar.h"

namespace numsim::materials {

/// Backward-Euler update solved by a scalar Newton the PROPERTY GRAPH drives.
///
/// Reads "residual" and "jacobian" from the material named by "function", and
/// publishes the increment as "delta". The function material re-evaluates its
/// residual against the current delta through update_source(), which is the
/// circularity that makes this work at all.
///
/// "function" is REQUIRED. It used to default to an empty string, which
/// silently selected a second, callback-driven mode inside this same class: no
/// inputs were created, update() was never bound, and "delta" stayed at zero.
/// A consumer reading it -- autocatalytic_reaction does -- then froze at its
/// start value for the whole analysis with no error anywhere. Measured: a cure
/// that should reach 1.0 sat at 0.01 for 30 steps. That mode is now
/// local_newton, chosen by naming a different type rather than by omitting a
/// parameter.
///
/// Parameters:
///   "name", "function", "tolerance", "max_iter"
template<typename Traits>
class backward_euler final : public material_base<backward_euler<Traits>, Traits> {
public:
  using base = material_base<backward_euler<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template<typename... Args>
  backward_euler(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_delta(base::template add_output<value_type>("delta")),
        m_func_name(base::template get_parameter<std::string>("function")),
        m_solver(base::template get_parameter<value_type>("tolerance"),
                 base::template get_parameter<int>("max_iter")),
        m_residual(base::template add_input<value_type>(
            m_func_name, "residual", EdgeKind::Local)),
        m_jacobian(base::template add_input<value_type>(
            m_func_name, "jacobian", EdgeKind::Local))
  {
    if (auto p = base::m_property_handler.find(base::m_name, "delta"))
      (*p)->traits().update = [this]() { this->update(); };
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("function").template add<is_required>();
    para.template insert<value_type>("tolerance")
        .template add<set_default>(value_type{1e-10});
    para.template insert<int>("max_iter").template add<set_default>(50);
    return para;
  }

  /// Whether the last update() converged.
  ///
  /// Previously never set on this path at all: the loop broke on tolerance, on
  /// a singular jacobian and on exhausting its budget, and all three looked
  /// identical from outside.
  [[nodiscard]] bool converged() const noexcept { return m_converged; }

  void update() override {
    // The seed is nonzero because a jacobian evaluated at exactly zero is
    // singular for the rate laws this drives.
    const auto r = m_solver.solve(
        [this](value_type x) -> std::pair<value_type, value_type> {
          m_delta = x;
          m_residual.update_source();
          const auto res = m_residual.get();
          m_jacobian.update_source();
          return {res, m_jacobian.get()};
        },
        value_type{5e-12});

    m_converged = r.converged;
    // Sign convention owned here rather than by the solver: the quantities this
    // integrates -- degree of cure, and similar -- only increase.
    m_delta = std::abs(r.x);
  }

private:
  value_type& m_delta;
  const std::string& m_func_name;
  newton_scalar<value_type> m_solver;
  const input_property<value_type, property_traits>& m_residual;
  const input_property<value_type, property_traits>& m_jacobian;
  bool m_converged{false};
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_BACKWARD_EULER_H
