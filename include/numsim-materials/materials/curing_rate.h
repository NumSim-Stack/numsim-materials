#ifndef NUMSIM_MATERIALS_CURING_RATE_H
#define NUMSIM_MATERIALS_CURING_RATE_H

#include <algorithm>
#include <cmath>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Pure rate function for autocatalytic curing — no history, no solver.
///
///   rate = k(T) · z^m · (1-z)^n
///   rate_derivative = dk/dz
///
/// The state z is read from an external integrator via Local edge.
/// Used with RK integrators (explicit, DIRK, fully implicit).
template <typename Traits>
class curing_rate final
    : public material_base<curing_rate<Traits>, Traits> {
public:
  using base = material_base<curing_rate<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit curing_rate(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_rate(base::template add_output<value_type>(
            "rate", &curing_rate::compute)),
        m_drate(base::template add_output<value_type>("rate_derivative")),
        m_A(base::template get_parameter<value_type>("A")),
        m_E(base::template get_parameter<value_type>("E")),
        m_n(base::template get_parameter<value_type>("n")),
        m_m(base::template get_parameter<value_type>("m")),
        m_temp_name(base::template get_parameter<std::string>("temperature_name")),
        m_integrator_name(base::template get_parameter<std::string>("integrator_source")),
        m_theta(base::template add_input_history<value_type>(
            connection_source{m_temp_name, "state"}, EdgeKind::Global)),
        m_z(base::template add_input<value_type>(
            m_integrator_name, "state", EdgeKind::Local))
  {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<value_type>("A").template add<is_required>();
    para.template insert<value_type>("E").template add<is_required>();
    para.template insert<value_type>("n").template add<is_required>();
    para.template insert<value_type>("m").template add<is_required>();
    para.template insert<std::string>("temperature_name")
        .template add<set_default>("temperature");
    para.template insert<std::string>("integrator_source").template add<is_required>();
    return para;
  }

  void compute() {
    const auto theta = value_type{273.15} + m_theta.new_value();
    const auto k = m_A * std::exp(-m_E / (m_R * theta));
    const auto z = std::clamp(m_z.get(), value_type{1e-30}, value_type{1} - value_type{1e-15});

    if (z >= value_type{1} - value_type{1e-15} || z <= value_type{1e-30}) {
      m_rate = value_type{0};
      m_drate = value_type{0};
      return;
    }

    const auto zm = std::pow(z, m_m);
    const auto omz = std::pow(value_type{1} - z, m_n);
    m_rate = k * zm * omz;
    m_drate = k * (m_m * std::pow(z, m_m - 1) * omz
                 - m_n * zm * std::pow(value_type{1} - z, m_n - 1));
  }

private:
  value_type& m_rate;
  value_type& m_drate;
  const value_type& m_A;
  const value_type& m_E;
  const value_type& m_n;
  const value_type& m_m;
  const std::string& m_temp_name;
  const std::string& m_integrator_name;
  const input_history<value_type, property_traits>& m_theta;
  const input_property<value_type, property_traits>& m_z;
  static constexpr value_type m_R{8.31446261815324};
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CURING_RATE_H
