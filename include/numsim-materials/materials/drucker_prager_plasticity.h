#ifndef DRUCKER_PRAGER_PLASTICITY_H
#define DRUCKER_PRAGER_PLASTICITY_H

#include <cmath>
#include <concepts>
#include <stdexcept>
#include <utility>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_ref.h"
#include "numsim-materials/materials/drucker_prager_yield_function.h"
#include "numsim-materials/materials/plasticity_utils.h"
#include "numsim-materials/solvers/backward_euler.h"

namespace numsim::materials {

/// Single-stage implicit Euler plasticity (classical return mapping).
///
/// Uses solver.solve() for the Newton iteration. No tableau, no stage
/// vectors, no overhead. This is the standard radial return for J2.
template<typename Traits>
class drucker_prager_plasticity final
    : public material_base<drucker_prager_plasticity<Traits>, Traits> {
public:
  using base = material_base<drucker_prager_plasticity<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  static constexpr auto Dim = base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;
  using yield_fn = drucker_prager_yield_function<value_type, base::Dim>;
  using solver_type = backward_euler<Traits>;

  template <typename... Args>
  explicit drucker_prager_plasticity(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_stress(base::template add_output<tensor2>(
            "stress", &drucker_prager_plasticity::compute)),
        m_tangent(base::template add_output<tensor4>("tangent")),
        m_eps_p(base::template add_history_output<tensor2>("plastic_strain")),
        m_kappa(base::template add_history_output<value_type>("equivalent_plastic_strain")),
        m_G(base::template get_parameter<value_type>("G")),
        m_sigma_0(base::template get_parameter<value_type>("sigma_0")),
        m_solver(base::template add_material_ref<solver_type>(
            base::template get_parameter<std::string>("solver_source"))),
        m_C_e(base::template add_input<tensor4>(
            base::template get_parameter<std::string>("elastic_source"),
            "tangent", EdgeKind::Global)),
        m_strain(base::template add_input<tensor2>(
            base::template get_parameter<std::string>("strain_source"),
            "strain", EdgeKind::Global)),
        m_H(base::template add_input<value_type>(
            base::template get_parameter<std::string>("hardening_source"),
            "hardening_stress", EdgeKind::Local)),
        m_dH(base::template add_input<value_type>(
            base::template get_parameter<std::string>("hardening_source"),
            "hardening_modulus", EdgeKind::Local))
  {
    m_yf = yield_fn(base::template get_parameter<value_type>("eta"),
                    base::template get_parameter<value_type>("beta"),
                    base::template get_parameter<value_type>("K_bulk"));
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("elastic_source").template add<is_required>();
    para.template insert<std::string>("hardening_source").template add<is_required>();
    para.template insert<std::string>("strain_source").template add<is_required>();
    para.template insert<std::string>("solver_source").template add<is_required>();
    para.template insert<value_type>("G").template add<is_required>();
    para.template insert<value_type>("sigma_0").template add<is_required>();
    // The cone's friction, dilatancy and bulk modulus, as plain scalars.
    // They used to arrive inside a "yield_function" C++ object, which the JSON
    // reader cannot convert -- so the material could not be configured from a
    // document at all (see #33).
    para.template insert<value_type>("eta").template add<is_required>();
    para.template insert<value_type>("beta").template add<is_required>();
    para.template insert<value_type>("K_bulk").template add<is_required>();
    return para;
  }

  void compute() {
    const auto& C_e = m_C_e.get();
    const auto kappa_n = m_kappa.old_value();

    m_kappa.new_value() = kappa_n;
    m_H.update_source();

    auto ts = plasticity_detail::compute_trial<value_type, Dim>(
        m_yf, m_strain.get(), m_eps_p.old_value(), C_e, m_sigma_0, m_H.get());

    if (!ts.yielding) {
      do_elastic(ts.eval.sig, C_e);
      return;
    }

    // For yield functions with an apex (DP cone), avoid the wasted smooth
    // Newton when the trial state already guarantees apex overshoot.
    // Conservative pre-check using the zero-hardening dlambda bound:
    //   dlambda_max = F_trial / G_eff  ≥  true dlambda  (for H' ≥ 0)
    // If even this upper bound triggers apex, smooth will also.
    const auto G_eff_pre = m_yf.effective_modulus(m_G);
    if (m_yf.needs_apex_return(m_G, ts.eval.F / G_eff_pre, ts.eval.sig_eq)) {
      do_apex_return(ts.eval.sig, C_e, kappa_n);
      return;
    }

    const auto dlambda = solve_smooth_newton(ts.eval.modified_sig_eq, kappa_n);

    // If smooth Newton fails and apex is available, try apex as fallback.
    if (!m_solver.get().converged()) {
      do_apex_return(ts.eval.sig, C_e, kappa_n);
      if (!m_solver.get().converged())
        throw std::runtime_error(
            "drucker_prager_plasticity: both smooth and apex Newton failed");
      return;
    }

    do_smooth_return(ts.eval, C_e, kappa_n, dlambda);
  }

private:
  /// Elastic step: stress = C:(ε - ε_p_old), tangent = C, history unchanged.
  void do_elastic(const tensor2& sig_trial, const tensor4& C_e) {
    m_stress = sig_trial;
    m_tangent = C_e;
    m_eps_p.new_value() = m_eps_p.old_value();
    m_kappa.new_value() = m_kappa.old_value();
  }

  /// Scalar Newton solve: r(Δλ) = phi - G_eff·Δλ - Y0 - H(κ_n + Δλ) = 0.
  /// Used for both the smooth and apex returns with different (phi, G_eff).
  value_type solve_scalar_return(value_type phi_trial, value_type G_eff,
                                  value_type kappa_n) {
    auto eval = [&](value_type dl) -> std::pair<value_type, value_type> {
      m_kappa.new_value() = kappa_n + dl;
      m_H.update_source();
      return {m_yf.residual(phi_trial, dl, G_eff, m_sigma_0, m_H.get()),
              m_yf.jacobian(G_eff, m_dH.get())};
    };
    return m_solver.get().solve(eval);
  }

  /// Smooth-cone return Newton: phi = modified_sig_eq, G_eff from yield function.
  value_type solve_smooth_newton(value_type phi_trial, value_type kappa_n) {
    return solve_scalar_return(phi_trial, m_yf.effective_modulus(m_G), kappa_n);
  }

  /// Smooth-cone return: ε_p update with N_trial, tangent via implicit function thm.
  void do_smooth_return(const plasticity_detail::state_eval<value_type, Dim>& ts,
                        const tensor4& C_e, value_type kappa_n, value_type dlambda) {
    m_eps_p.new_value() = m_eps_p.old_value() + dlambda * ts.N;
    m_kappa.new_value() = kappa_n + dlambda;
    m_stress = tmech::dcontract(C_e, m_strain.get() - m_eps_p.new_value());

    // Tangent at trial state (return mapping uses N_trial).
    // For J2, trial = converged. For DP, they differ.
    m_H.update_source();
    m_tangent = plasticity_detail::compute_tangent<value_type, Dim>(
        m_yf, ts.sig_dev, ts.N, ts.sig_eq, dlambda, m_dH.get(), C_e, m_G);
  }

  /// Apex return: deviatoric stress vanishes, only volumetric Newton.
  /// dev(ε_p) = dev(ε), tr(ε_p) += β·Δκ. Tangent is rank-1 volumetric.
  void do_apex_return(const tensor2& sig_trial, const tensor4& C_e,
                      value_type kappa_n) {
    const auto phi_apex = m_yf.apex_modified_sig_eq(sig_trial);
    const auto G_eff_apex = m_yf.apex_effective_modulus();
    const auto dkappa = solve_scalar_return(phi_apex, G_eff_apex, kappa_n);

    m_eps_p.new_value() = m_yf.apex_plastic_strain(
        m_strain.get(), m_eps_p.old_value(), dkappa);
    m_kappa.new_value() = kappa_n + dkappa;
    m_stress = tmech::dcontract(C_e, m_strain.get() - m_eps_p.new_value());

    // Apex tangent is a branch tangent: valid only for perturbations that
    // remain on the active apex branch (q stays at 0).
    m_H.update_source();
    m_tangent = m_yf.apex_tangent(m_dH.get());
  }

private:
  tensor2& m_stress;
  tensor4& m_tangent;
  history_property<tensor2>& m_eps_p;
  history_property<value_type>& m_kappa;

  const value_type& m_G;
  const value_type& m_sigma_0;
  material_ref<solver_type, Traits>& m_solver;

  const input_property<tensor4, property_traits>& m_C_e;
  const input_property<tensor2, property_traits>& m_strain;
  const input_property<value_type, property_traits>& m_H;
  const input_property<value_type, property_traits>& m_dH;
  yield_fn m_yf{};
};


} // namespace numsim::materials

#endif // DRUCKER_PRAGER_PLASTICITY_H
