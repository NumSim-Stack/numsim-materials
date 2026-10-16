#ifndef NUMSIM_MATERIALS_REPORT_LOAD_DRIVER_H
#define NUMSIM_MATERIALS_REPORT_LOAD_DRIVER_H

#include <algorithm>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/report/load_control.h"
#include "numsim-materials/report/symmetric_basis.h"
#include "numsim-materials/umat/material_point_evaluator.h"
#include "numsim-materials/umat/tensor_conversion.h"

/// Mixed strain/stress control of one material point, for reports and checks.
namespace numsim::materials::report {

template <typename Traits>
class load_driver {
public:
  using value_type = typename Traits::value_type;
  using evaluator = umat::material_point_evaluator<Traits>;
  using tensor2 = tmech::tensor<value_type, 3, 2>;
  using tensor4 = tmech::tensor<value_type, 3, 4>;
  using control = load_control<value_type>;

  struct options {
    value_type tolerance{1e-10}; ///< on |P:(sigma - sigma*)| relative to |sigma|
    int max_iter{25};
  };

  /// Converged state at the start of a step.
  struct point_state {
    tensor2 strain{};
    std::vector<value_type> statev{};
  };

  struct step_result {
    tensor2 strain{};
    tensor2 stress{};
    tensor4 tangent{};
    std::vector<value_type> residuals{}; ///< |P:(sigma - sigma*)| before each correction
    bool converged{false};
    point_state start{};
    value_type time{0}, dt{0};
  };

  explicit load_driver(evaluator &ev, options opts = {}) : m_ev(ev), m_opts(opts), m_state{{}, std::vector<value_type>(ev.nstatv())} {}

  /// Strain-controlled components from `strain`, stress-controlled ones from `stress`; commits on convergence.
  step_result step(tensor2 const &strain, tensor2 const &stress, control const &c, value_type time, value_type dt) {
    auto const &P{c.stress_projector()};
    tensor4 const Q{c.strain_projector() + skew_identity<value_type>()}; // Iskw: A regular on the full 9x9 space
    step_result res{.strain = tmech::dcontract(c.strain_projector(), strain) + tmech::dcontract(P, m_state.strain),
                    .start = m_state, .time = time, .dt = dt};
    for (int it{0}; it <= m_opts.max_iter; ++it) {
      evaluate(res.start, res.strain, time, dt, res.stress, res.tangent);
      tensor2 const r{tmech::dcontract(P, tensor2{res.stress - stress})};
      res.residuals.push_back(tmech::norm(r));
      if (res.residuals.back() <= m_opts.tolerance * std::max(tmech::norm(res.stress), tmech::norm(tensor2{tmech::dcontract(P, stress)}))) {
        res.converged = true;
        break;
      }
      tensor4 const A{tmech::dcontract(tensor4{tmech::dcontract(P, res.tangent)}, P) + Q};
      res.strain = tmech::eval(res.strain - tmech::dcontract(tensor4{tmech::invf(A)}, r));
    }
    if (res.converged) {
      m_state.strain = res.strain;
      m_ev.store_statev(m_state.statev.data());
    }
    return res;
  }

  /// Relative error of the tangent of `res` against central differences from its start state.
  value_type tangent_error(step_result const &res, value_type h = 1e-7) {
    tensor4 fd, unused;
    tensor2 sp, sm;
    for (auto const &E : symmetric_basis<value_type>()) {
      evaluate(res.start, tensor2{res.strain + h * E}, res.time, res.dt, sp, unused);
      evaluate(res.start, tensor2{res.strain - h * E}, res.time, res.dt, sm, unused);
      fd += tmech::otimes(tensor2{(sp - sm) / (2 * h)}, E);
    }
    return tmech::norm(tensor4{fd - res.tangent}) / tmech::norm(res.tangent);
  }

  [[nodiscard]] point_state const &state() const noexcept { return m_state; }

private:
  void evaluate(point_state const &start, tensor2 const &strain, value_type time, value_type dt, tensor2 &stress,
                tensor4 &tangent) {
    value_type old6[umat::canonical_width], new6[umat::canonical_width], sig6[umat::canonical_width];
    value_type tan36[umat::canonical_width * umat::canonical_width];
    umat::strain_to_buffer(start.strain, old6);
    umat::strain_to_buffer(strain, new6);
    m_ev.evaluate_canonical(start.statev.data(), old6, new6, time, dt, sig6, tan36);
    stress = umat::stress_from_buffer(sig6);
    tangent = umat::tangent_from_buffer(tan36);
  }

  evaluator &m_ev;
  options m_opts;
  point_state m_state;
};

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_LOAD_DRIVER_H
