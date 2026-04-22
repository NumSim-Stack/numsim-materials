#ifndef NUMSIM_MATERIALS_RK_PLASTICITY_H
#define NUMSIM_MATERIALS_RK_PLASTICITY_H

#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/materials/yield_functions.h"
#include "numsim-materials/solvers/butcher_tableau.h"

namespace numsim::materials {

/// Multi-stage Runge-Kutta return mapping for small-strain plasticity.
///
/// Applies a Butcher tableau to the plasticity evolution equations:
///   dε_p/dλ = N(σ)
///   dα/dλ = 1
///   F(σ, α) = 0  (constraint at each implicit stage)
///
/// For each stage i:
///   ε_p^(i) = ε_p_n + Σ_j a_ij · Δλ_j · N_j
///   α^(i) = α_n + Σ_j a_ij · Δλ_j
///   σ^(i) = C : (ε - ε_p^(i))
///   Implicit: solve F(σ^(i), α^(i)) = 0 for Δλ_i
///   Explicit: Δλ_i from consistency condition
///
/// Final update:
///   ε_p_{n+1} = ε_p_n + Σ_i b_i · Δλ_i · N_i
///   α_{n+1} = α_n + Σ_i b_i · Δλ_i
///
/// With implicit_euler() tableau → classical return mapping (1st order).
/// With sdirk3() tableau → 3rd order return mapping.
/// With gauss_legendre_4() → 4th order (coupled Newton system).
template<typename Traits, typename YieldFunction>
class rk_plasticity final
    : public material_base<rk_plasticity<Traits, YieldFunction>, Traits> {
public:
  using base = material_base<rk_plasticity<Traits, YieldFunction>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  static constexpr auto Dim = base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;
  using yield_fn = YieldFunction;

  template <typename... Args>
  explicit rk_plasticity(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_stress(base::template add_output<tensor2>(
            "stress", &rk_plasticity::compute)),
        m_tangent(base::template add_output<tensor4>("tangent")),
        m_eps_p(base::template add_history_output<tensor2>("plastic_strain")),
        m_alpha(base::template add_history_output<value_type>("equivalent_plastic_strain")),
        m_G(base::template get_parameter<value_type>("G")),
        m_sigma_0(base::template get_parameter<value_type>("sigma_0")),
        m_tol(base::template get_parameter<value_type>("tolerance")),
        m_max_iter(base::template get_parameter<int>("max_iter")),
        m_tableau(base::template get_parameter<const butcher_tableau*>("tableau")),
        m_elastic_source(base::template get_parameter<std::string>("elastic_source")),
        m_hardening_source(base::template get_parameter<std::string>("hardening_source")),
        m_strain_source(base::template get_parameter<std::string>("strain_source")),
        m_C_e(base::template add_input<tensor4>(
            m_elastic_source, "tangent", EdgeKind::Global)),
        m_strain(base::template add_input<tensor2>(
            m_strain_source, "strain", EdgeKind::Global)),
        m_H(base::template add_input<value_type>(
            m_hardening_source, "hardening_stress", EdgeKind::Local)),
        m_dH(base::template add_input<value_type>(
            m_hardening_source, "hardening_modulus", EdgeKind::Local))
  {
    const int s = m_tableau->stages();
    m_dlambda.resize(s, value_type{0});
    m_N_stage.resize(s);
    m_is_implicit.resize(s);
    m_diag.resize(s);
    for (int i = 0; i < s; ++i) {
      m_diag[i] = m_tableau->a(i, i);
      m_is_implicit[i] = std::abs(m_diag[i]) >= 1e-30;
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("elastic_source").template add<is_required>();
    para.template insert<std::string>("hardening_source").template add<is_required>();
    para.template insert<std::string>("strain_source").template add<is_required>();
    para.template insert<value_type>("G").template add<is_required>();
    para.template insert<value_type>("sigma_0").template add<is_required>();
    para.template insert<value_type>("tolerance")
        .template add<set_default>(value_type{1e-12});
    para.template insert<int>("max_iter")
        .template add<set_default>(int{50});
    return para;
  }

  void compute() {
    const auto& tab = *m_tableau;
    const int s = tab.stages();
    const auto& eps = m_strain.get();
    const auto& C_e = m_C_e.get();
    const auto I = tmech::eye<value_type, Dim, 2>();
    const auto eps_p_n = m_eps_p.old_value();
    const auto alpha_n = m_alpha.old_value();

    // Trial stress at initial state (for elastic check + tangent)
    const tensor2 sig_trial_0{tmech::dcontract(C_e, eps - eps_p_n)};
    const auto trace_sig = tmech::trace(sig_trial_0);
    const tensor2 sig_dev_0{sig_trial_0 - (trace_sig / value_type{Dim}) * I};
    const auto sig_eq_0 = yield_fn::equivalent_stress(sig_dev_0);

    // Elastic check with hardening at α_n
    m_alpha.new_value() = alpha_n;
    m_H.update_source();
    const auto F_trial = yield_fn::trial_yield(sig_eq_0, m_sigma_0, m_H.get());

    if (F_trial <= value_type{0}) {
      m_stress = sig_trial_0;
      m_tangent = C_e;
      m_eps_p.new_value() = eps_p_n;
      m_alpha.new_value() = alpha_n;
      return;
    }

    // Cache trial N for tangent computation
    const tensor2 N_trial{yield_fn::flow_normal(sig_dev_0, sig_eq_0)};

    // --- Multi-stage return mapping ---
    for (int i = 0; i < s; ++i)
      m_dlambda[i] = value_type{0};

    for (int i = 0; i < s; ++i) {
      // Accumulated state from previous stages
      tensor2 eps_p_acc{eps_p_n};
      auto alpha_acc = alpha_n;
      for (int j = 0; j < i; ++j) {
        eps_p_acc = eps_p_acc + tab.a(i, j) * m_dlambda[j] * m_N_stage[j];
        alpha_acc += tab.a(i, j) * m_dlambda[j];
      }

      if (!m_is_implicit[i]) {
        // Explicit stage: evaluate N and compute Δλ from consistency
        const tensor2 sig_i{tmech::dcontract(C_e, eps - eps_p_acc)};
        const auto tr_i = tmech::trace(sig_i);
        const tensor2 dev_i{sig_i - (tr_i / value_type{Dim}) * I};
        const auto seq_i = yield_fn::equivalent_stress(dev_i);

        m_alpha.new_value() = alpha_acc;
        m_H.update_source();
        const auto F_i = yield_fn::trial_yield(seq_i, m_sigma_0, m_H.get());
        const auto dH_i = m_dH.get();

        if (F_i > value_type{0} && seq_i > value_type{1e-30}) {
          m_N_stage[i] = yield_fn::flow_normal(dev_i, seq_i);
          m_dlambda[i] = F_i / (value_type{3} * m_G + dH_i);
        } else {
          m_N_stage[i] = tensor2{};
          m_dlambda[i] = value_type{0};
        }
      } else {
        // Implicit stage: Newton solve for Δλ_i
        const auto aii = m_diag[i];
        m_dlambda[i] = value_type{0};

        for (int iter = 0; iter < m_max_iter; ++iter) {
          // Trial state including current stage contribution
          tensor2 eps_p_i{eps_p_acc + aii * m_dlambda[i] * N_trial};
          auto alpha_i = alpha_acc + aii * m_dlambda[i];

          const tensor2 sig_i{tmech::dcontract(C_e, eps - eps_p_i)};
          const auto tr_i = tmech::trace(sig_i);
          const tensor2 dev_i{sig_i - (tr_i / value_type{Dim}) * I};
          const auto seq_i = yield_fn::equivalent_stress(dev_i);

          m_alpha.new_value() = alpha_i;
          m_H.update_source();
          const auto H_i = m_H.get();
          const auto dH_i = m_dH.get();

          const auto F_i = yield_fn::trial_yield(seq_i, m_sigma_0, H_i);
          if (std::abs(F_i) < m_tol) {
            if (seq_i > value_type{1e-30})
              m_N_stage[i] = yield_fn::flow_normal(dev_i, seq_i);
            break;
          }

          const auto dF_i = -aii * (value_type{3} * m_G + dH_i);
          m_dlambda[i] -= F_i / dF_i;
        }
      }
    }

    // Final update using b weights
    tensor2 eps_p_new{eps_p_n};
    auto alpha_new = alpha_n;
    auto total_dlambda = value_type{0};
    for (int i = 0; i < s; ++i) {
      eps_p_new = eps_p_new + tab.b[i] * m_dlambda[i] * m_N_stage[i];
      alpha_new += tab.b[i] * m_dlambda[i];
      total_dlambda += tab.b[i] * m_dlambda[i];
    }

    m_eps_p.new_value() = eps_p_new;
    m_alpha.new_value() = alpha_new;

    // Stress at converged state
    m_stress = tmech::dcontract(C_e, eps - eps_p_new);

    // Algorithmic tangent — uses trial N and sig_eq (radial return for J2)
    m_H.update_source();
    const auto dH_val = m_dH.get();
    const auto dr_ddlambda = yield_fn::jacobian(m_G, dH_val);
    const tensor2 dr_deps{tmech::dcontract(N_trial, C_e)};
    const tensor2 dlambda_deps{-dr_deps / dr_ddlambda};

    const tensor4 dN_dsig{yield_fn::flow_normal_stress_derivative(N_trial, sig_eq_0)};
    const tensor4 dN_deps{tmech::dcontract(dN_dsig, C_e)};
    const tensor4 dsig_deps{C_e - value_type{2} * m_G * total_dlambda * dN_deps};
    const tensor2 dsig_ddlambda{-value_type{2} * m_G * N_trial};

    m_tangent = dsig_deps + tmech::otimes(dsig_ddlambda, dlambda_deps);
  }

private:
  tensor2& m_stress;
  tensor4& m_tangent;
  history_property<tensor2>& m_eps_p;
  history_property<value_type>& m_alpha;

  const value_type& m_G;
  const value_type& m_sigma_0;
  const value_type& m_tol;
  const int& m_max_iter;
  const butcher_tableau* m_tableau;
  const std::string& m_elastic_source;
  const std::string& m_hardening_source;
  const std::string& m_strain_source;

  const input_property<tensor4, property_traits>& m_C_e;
  const input_property<tensor2, property_traits>& m_strain;
  const input_property<value_type, property_traits>& m_H;
  const input_property<value_type, property_traits>& m_dH;

  // Pre-allocated stage storage
  std::vector<value_type> m_dlambda;
  std::vector<tensor2> m_N_stage;
  std::vector<bool> m_is_implicit;
  std::vector<double> m_diag;
};

// --- Convenience aliases ---

template<typename Traits>
using j2_rk_plasticity = rk_plasticity<Traits,
    j2_yield_function<typename Traits::value_type, Traits::Dim>>;

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_RK_PLASTICITY_H
