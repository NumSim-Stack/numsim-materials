#ifndef NUMSIM_MATERIALS_UMAT_DEFORMATION_GRADIENT_DRIVER_H
#define NUMSIM_MATERIALS_UMAT_DEFORMATION_GRADIENT_DRIVER_H

#include <algorithm>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/umat/load_control.h"
#include "numsim-materials/umat/material_point_evaluator.h"

/// One material point at finite strain under mixed F / P control: Newton on the free components of F.
namespace numsim::materials::umat {

template <typename Traits>
class deformation_gradient_driver {
public:
  using value_type = typename Traits::value_type;
  using evaluator = material_point_evaluator<Traits>;
  using tensor2 = tmech::tensor<value_type, 3, 2>;
  using tensor4 = tmech::tensor<value_type, 3, 4>;
  using control = deformation_control<value_type>;

  struct options {
    value_type tolerance{1e-10}; ///< on |M:(P - P*)| relative to max(|P|, |M:P*|, |A| |F - F_n|)
    int max_iter{25};
    value_type fd_step{1e-7};    ///< perturbation of F in tangent_error()
  };

  /// Converged state at the start of a step.
  struct point_state {
    tensor2 F{tmech::eye<value_type, 3, 2>()};
    std::vector<value_type> statev{};
  };

  struct step_result {
    tensor2 F{};
    tensor2 stress{};  ///< first Piola-Kirchhoff
    tensor4 tangent{}; ///< dP/dF
    std::vector<value_type> residuals{}; ///< |M:(P - P*)| before each correction
    bool converged{false};
    point_state start{};
    value_type time{0}, dt{0};
  };

  explicit deformation_gradient_driver(evaluator &ev, options opts = {})
      : m_ev(ev), m_opts(opts), m_state{tmech::eye<value_type, 3, 2>(), std::vector<value_type>(ev.nstatv())} {}

  /// Prescribed components of F from `F`, stress-controlled ones from `P`; commits on convergence.
  step_result step(tensor2 const &F, tensor2 const &P, control const &c, value_type time, value_type dt) {
    auto const &M{c.stress_projector()};
    tensor4 const N{c.deformation_projector()};
    step_result res{.F = tmech::dcontract(N, F) + tmech::dcontract(M, m_state.F), .start = m_state, .time = time, .dt = dt};
    for (int it{0}; it <= m_opts.max_iter; ++it) {
      evaluate(res.start, res.F, time, dt, res.stress, res.tangent);
      tensor2 const r{tmech::dcontract(M, tensor2{res.stress - P})};
      res.residuals.push_back(tmech::norm(r));
      auto const scale{std::max({tmech::norm(res.stress), tmech::norm(tensor2{tmech::dcontract(M, P)}),
                                 tmech::norm(res.tangent) * tmech::norm(tensor2{res.F - res.start.F})})};
      if (res.residuals.back() <= m_opts.tolerance * scale) {
        res.converged = true;
        break;
      }
      tensor4 const A{tmech::dcontract(tensor4{tmech::dcontract(M, res.tangent)}, M) + N};
      res.F = tmech::eval(res.F - tmech::dcontract(tensor4{tmech::invf(A)}, r));
    }
    if (res.converged) {
      m_state.F = res.F;
      m_ev.store_statev(m_state.statev.data());
    }
    return res;
  }

  /// Relative error of dP/dF of `res` against central differences from its start state.
  value_type tangent_error(step_result const &res) {
    auto const h{m_opts.fd_step};
    tensor4 fd, unused;
    tensor2 Pp, Pm;
    for (auto const &E : materials::full_basis<value_type>()) {
      evaluate(res.start, tensor2{res.F + h * E}, res.time, res.dt, Pp, unused);
      evaluate(res.start, tensor2{res.F - h * E}, res.time, res.dt, Pm, unused);
      fd += tmech::otimes(tensor2{(Pp - Pm) / (2 * h)}, E);
    }
    return tmech::norm(tensor4{fd - res.tangent}) / tmech::norm(res.tangent);
  }

  [[nodiscard]] point_state const &state() const noexcept { return m_state; }

private:
  void evaluate(point_state const &start, tensor2 const &F, value_type time, value_type dt, tensor2 &P, tensor4 &A) {
    m_ev.evaluate_deformation_gradient(start.statev.data(), start.F, F, time, dt);
    P = m_ev.stress();
    A = m_ev.tangent();
  }

  evaluator &m_ev;
  options m_opts;
  point_state m_state;
};

} // namespace numsim::materials::umat

#endif // NUMSIM_MATERIALS_UMAT_DEFORMATION_GRADIENT_DRIVER_H
