#ifndef NUMSIM_MATERIALS_BACKWARD_EULER_H
#define NUMSIM_MATERIALS_BACKWARD_EULER_H

#include <cmath>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Backward Euler solver as a material.
///
/// Consumes: function_name::residual, function_name::jacobian (Local edges)
/// Produces: "delta" (converged increment)
///
/// The Newton iteration calls update_source() on the residual/jacobian inputs
/// to re-evaluate them at each trial increment.
template<typename Traits>
class backward_euler final
    : public material_base<backward_euler<Traits>, Traits> {
public:
  using base = material_base<backward_euler<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template<typename... Args>
  backward_euler(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_delta(base::template add_output<value_type>("delta")),
        m_func_name(base::template get_parameter<std::string>("function")),
        m_tol(base::template get_parameter<value_type>("tolerance")),
        m_max_iter(base::template get_parameter<int>("max_iter"))
  {
    // If a function name is provided, set up graph-driven iteration
    if (!m_func_name.empty()) {
      m_residual = &base::template add_input<value_type>(
          m_func_name, "residual", EdgeKind::Local);
      m_jacobian = &base::template add_input<value_type>(
          m_func_name, "jacobian", EdgeKind::Local);
      // Bind update callback for graph-driven mode
      if (auto p = base::m_property_handler.find(base::m_name, "delta"))
        (*p)->traits().update = [this]() { this->update(); };
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("function")
        .template add<set_default>(std::string{});
    para.template insert<value_type>("tolerance")
        .template add<set_default>(value_type{5e-12});
    para.template insert<int>("max_iter")
        .template add<set_default>(100);
    return para;
  }

  /// Property-graph driven iteration: reads residual/jacobian via update_source.
  /// Used when the solver is in the graph (e.g., curing reaction).
  void update() override {
    if (!m_residual || !m_jacobian) return;
    m_delta = value_type{5e-12};
    for (int i = 0; i < m_max_iter; ++i) {
      m_residual->update_source();
      const auto& r = m_residual->get();
      if (std::abs(r) <= m_tol) break;
      m_jacobian->update_source();
      const auto& j = m_jacobian->get();
      if (std::abs(j) < value_type{1e-30}) break;
      auto step = r / j;
      // Damped Newton: halve step if it produces NaN
      for (int k = 0; k < 5; ++k) {
        auto candidate = m_delta - step;
        m_delta = candidate;
        m_residual->update_source();
        auto r_new = m_residual->get();
        if (!std::isnan(r_new) && !std::isinf(r_new)) break;
        m_delta = candidate + step; // restore
        step *= value_type{0.5};
      }
    }
    // Ensure positive increment (curing degree can only increase)
    m_delta = std::abs(m_delta);
  }

  /// Direct call: another material provides eval(x) → {residual, jacobian}.
  /// Used when the caller drives the iteration (e.g., plasticity return mapping).
  template<typename Eval>
  value_type solve(Eval&& eval, value_type x0 = value_type{0}) const {
    auto x = x0;
    for (int i = 0; i < m_max_iter; ++i) {
      auto [r, dr] = eval(x);
      if (std::abs(r) < m_tol) return x;
      if (std::abs(dr) < value_type{1e-30}) return x;
      x -= r / dr;
    }
    return x;
  }

private:
  value_type& m_delta;
  const std::string& m_func_name;
  const value_type& m_tol;
  const int& m_max_iter;
  const input_property<value_type, property_traits>* m_residual{nullptr};
  const input_property<value_type, property_traits>* m_jacobian{nullptr};
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_BACKWARD_EULER_H
