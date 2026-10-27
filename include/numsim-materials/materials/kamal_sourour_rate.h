#ifndef NUMSIM_MATERIALS_KAMAL_SOUROUR_RATE_H
#define NUMSIM_MATERIALS_KAMAL_SOUROUR_RATE_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

/// Kamal-Sourour cure kinetics dz/dt = (k1 + k2 z^m) (1 - z)^n, k_i = A_i exp(-E_i / (R T)); outputs "rate" and
/// "rate_derivative" for an rk_integrator. Parameters "A1", "E1", "A2", "E2", "m", "n" (all >= 0),
/// "temperature_name" (state in degrees Celsius), "integrator_source".
template <typename Traits>
class kamal_sourour_rate final : public material_base<kamal_sourour_rate<Traits>, Traits> {
public:
  using base = material_base<kamal_sourour_rate<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit kamal_sourour_rate(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_rate(base::template add_output<value_type>("rate", &kamal_sourour_rate::compute)),
        m_drate(base::template add_output<value_type>("rate_derivative")),
        m_A1(base::template get_parameter<value_type>("A1")),
        m_E1(base::template get_parameter<value_type>("E1")),
        m_A2(base::template get_parameter<value_type>("A2")),
        m_E2(base::template get_parameter<value_type>("E2")),
        m_m(base::template get_parameter<value_type>("m")),
        m_n(base::template get_parameter<value_type>("n")),
        m_theta(base::template add_input_history<value_type>(
            connection_source{base::template get_parameter<std::string>("temperature_name"), "state"}, EdgeKind::Global)),
        m_z(base::template add_input<value_type>(base::template get_parameter<std::string>("integrator_source"), "state",
                                                 EdgeKind::Local)) {
    for (auto const &[key, value] : {std::pair{"A1", m_A1}, std::pair{"E1", m_E1}, std::pair{"A2", m_A2}, std::pair{"E2", m_E2},
                                     std::pair{"m", m_m}, std::pair{"n", m_n}})
      if (!(value >= 0))
        throw std::invalid_argument("kamal_sourour_rate '" + base::name() + "': " + key + " must be non-negative");
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    for (auto const *key : {"A1", "E1", "A2", "E2", "m", "n"})
      required<value_type>(para, key);
    optional<std::string>(para, "temperature_name", std::string{"temperature"});
    required<std::string>(para, "integrator_source");
    return para;
  }

  void compute() noexcept {
    auto const theta{value_type{273.15} + m_theta.new_value()};
    auto const k1{m_A1 * std::exp(-m_E1 / (m_R * theta))};
    auto const k2{m_A2 * std::exp(-m_E2 / (m_R * theta))};
    auto const z{std::clamp(m_z.get(), value_type{0}, value_type{1})}; // an integrator stage may overshoot; the clamp absorbs it
    if (z >= value_type{1}) {
      m_rate = m_drate = value_type{0};
      return;
    }
    auto const zm{std::pow(z, m_m)}; // z^0 = 1 at z = 0: the nth-order start keeps k2 for m = 0
    auto const dzm{z > 0 || m_m >= 1 ? m_m * std::pow(z, m_m - 1) : value_type{0}}; // m < 1: the slope diverges at z = 0
    auto const omz{std::pow(1 - z, m_n)};
    m_rate = (k1 + k2 * zm) * omz;
    m_drate = k2 * dzm * omz - (k1 + k2 * zm) * m_n * std::pow(1 - z, m_n - 1);
  }

private:
  value_type &m_rate;
  value_type &m_drate;
  value_type const &m_A1, &m_E1, &m_A2, &m_E2, &m_m, &m_n;
  input_history<value_type, property_traits> const &m_theta;
  input_property<value_type, property_traits> const &m_z;
  static constexpr value_type m_R{8.31446261815324};
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_KAMAL_SOUROUR_RATE_H
