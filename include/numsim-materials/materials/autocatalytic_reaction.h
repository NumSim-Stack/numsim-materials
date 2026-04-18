#ifndef AUTOCATALYTIC_REACTION_H
#define AUTOCATALYTIC_REACTION_H

#include "numsim-materials/core/material_base.h"
#include <cmath>

namespace numsim::materials {

/// Autocatalytic curing reaction using the Arrhenius equation.
///
/// Produces:
///   "current_state" (history) — curing degree z
///   "residual"                — R(dz) for solver
///   "jacobian"                — dR/dz for solver
///
/// Consumes:
///   time::state, temperature::state (Global)
///   solver::delta (Local — the trial increment from the solver)
template <typename Traits>
class autocatalytic_reaction
    : public material_base<autocatalytic_reaction<Traits>, Traits> {
public:
  using base = material_base<autocatalytic_reaction<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit autocatalytic_reaction(Args&&... args)
      : base(std::forward<Args>(args)...),
        // outputs
        m_his(base::template add_history_output<value_type>(
            "current_state", &autocatalytic_reaction::update)),
        m_res(base::template add_output<value_type>(
            "residual", &autocatalytic_reaction::update_residual)),
        m_jac(base::template add_output<value_type>(
            "jacobian", &autocatalytic_reaction::update_jacobian)),
        // parameters
        m_timer_name(base::template get_parameter<std::string>("timer_name")),
        m_temp_name(base::template get_parameter<std::string>("temperature_name")),
        m_solver_name(base::template get_parameter<std::string>("solver_name")),
        // inputs
        m_time(base::template add_input_history<value_type>(
            connection_source{m_timer_name, "state"}, EdgeKind::Global)),
        m_theta(base::template add_input_history<value_type>(
            connection_source{m_temp_name, "state"}, EdgeKind::Global)),
        m_dz(base::template add_input<value_type>(
            m_solver_name, "delta", EdgeKind::Local)),
        // parameters
        m_A(base::template get_parameter<value_type>("A")),
        m_E(base::template get_parameter<value_type>("E")),
        m_n(base::template get_parameter<value_type>("n")),
        m_m(base::template get_parameter<value_type>("m"))
  {
    const auto start = base::template get_parameter<value_type>("start_value");
    m_his.old_value() = start;
    m_his.new_value() = start;
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<value_type>("A").template add<is_required>();
    para.template insert<value_type>("E").template add<is_required>();
    para.template insert<value_type>("n").template add<is_required>();
    para.template insert<value_type>("m").template add<is_required>();
    para.template insert<value_type>("start_value")
        .template add<set_default>(value_type{1e-8});
    para.template insert<std::string>("solver_name")
        .template add<set_default>("solver");
    para.template insert<std::string>("timer_name")
        .template add<set_default>("time");
    para.template insert<std::string>("temperature_name")
        .template add<set_default>("temperature");
    return para;
  }

  /// Apply solver increment to state, clamped to [0, 1].
  void update() override {
    compute_rate_constant();
    m_his.new_value() = std::min(m_his.new_value() + m_dz.get(), value_type{1});
  }

  /// Compute residual R(dz) at the current trial increment.
  void update_residual() {
    compute_rate_constant();
    const auto z_n{m_his.new_value()};
    if (z_n >= value_type{1}) { m_res = value_type{0}; return; }
    const auto dt{m_time.new_value() - m_time.old_value()};
    const auto dz{std::abs(m_dz.get())};
    const auto z_total{std::min(dz + z_n, value_type{1} - value_type{1e-15})};
    m_res = dz - dt * m_k * std::pow(value_type{1} - z_total, m_n) *
                     std::pow(z_total, m_m);
  }

  /// Compute jacobian dR/dz at the current trial increment.
  void update_jacobian() {
    compute_rate_constant();
    const auto z_n{m_his.new_value()};
    if (z_n >= value_type{1}) { m_jac = value_type{1}; return; }
    const auto dt{m_time.new_value() - m_time.old_value()};
    const auto dz{std::abs(m_dz.get())};
    const auto z_total{std::min(dz + z_n, value_type{1} - value_type{1e-15})};
    m_jac = value_type{1} + m_k * dt * std::pow(z_total, m_m - value_type{1}) *
                     std::pow(value_type{1} - z_total, m_n - value_type{1}) *
                     (m_m * (z_total - value_type{1}) + m_n * z_total);
  }

  void compute_rate_constant() {
    const auto theta{value_type{273.15} + m_theta.new_value()};
    m_k = m_A * std::exp(-m_E / (m_R * theta));
  }

private:
  history_property<value_type>& m_his;
  value_type& m_res;
  value_type& m_jac;
  value_type m_k;

  const std::string& m_timer_name;
  const std::string& m_temp_name;
  const std::string& m_solver_name;

  const input_history<value_type, property_traits>& m_time;
  const input_history<value_type, property_traits>& m_theta;
  const input_property<value_type, property_traits>& m_dz;

  const value_type& m_A;
  const value_type& m_E;
  const value_type& m_n;
  const value_type& m_m;
  static constexpr value_type m_R{8.31446261815324};
};

} // namespace numsim::materials

#endif // AUTOCATALYTIC_REACTION_H
