#ifndef NUMSIM_MATERIALS_SMALL_STRAIN_PLASTICITY_H
#define NUMSIM_MATERIALS_SMALL_STRAIN_PLASTICITY_H

#include <cmath>
#include <utility>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_ref.h"
#include "numsim-materials/materials/yield_functions.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/solvers/butcher_tableau.h"

namespace numsim::materials {

/// Small-strain plasticity with pluggable yield function and optional
/// multi-stage Butcher tableau for the return mapping.
///
/// Without a tableau (default): uses solver.solve() for a single-stage
/// implicit Euler return mapping — the classical radial return.
///
/// With a tableau: multi-stage RK return mapping. Each implicit stage
/// solves F(σ^(i), α^(i)) = 0 for Δλ_i. Explicit stages use the
/// consistency condition. Higher-order accuracy for large strain increments.
///
/// Consistent tangent derived via implicit function theorem at the trial state.
template<typename Traits, typename YieldFunction>
class small_strain_plasticity final
    : public material_base<small_strain_plasticity<Traits, YieldFunction>, Traits> {
public:
  using base = material_base<small_strain_plasticity<Traits, YieldFunction>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  static constexpr auto Dim = base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;
  using yield_fn = YieldFunction;
  using solver_type = backward_euler<Traits>;

  template <typename... Args>
  explicit small_strain_plasticity(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_stress(base::template add_output<tensor2>(
            "stress", &small_strain_plasticity::compute)),
        m_tangent(base::template add_output<tensor4>("tangent")),
        m_eps_p(base::template add_history_output<tensor2>("plastic_strain")),
        m_alpha(base::template add_history_output<value_type>("equivalent_plastic_strain")),
        m_G(base::template get_parameter<value_type>("G")),
        m_sigma_0(base::template get_parameter<value_type>("sigma_0")),
        m_solver_name(base::template get_parameter<std::string>("solver_source")),
        m_solver(m_solver_name.empty() ? nullptr
            : &base::template add_material_ref<solver_type>(m_solver_name)),
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
    // Optional multi-stage tableau
    if (base::m_parameter_handler.contains("tableau")) {
      m_tableau = base::template get_parameter<const butcher_tableau*>("tableau");
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
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("elastic_source").template add<is_required>();
    para.template insert<std::string>("hardening_source").template add<is_required>();
    para.template insert<std::string>("strain_source").template add<is_required>();
    para.template insert<std::string>("solver_source")
        .template add<set_default>(std::string{});
    para.template insert<value_type>("G").template add<is_required>();
    para.template insert<value_type>("sigma_0").template add<is_required>();
    para.template insert<value_type>("tolerance")
        .template add<set_default>(value_type{1e-12});
    para.template insert<int>("max_iter")
        .template add<set_default>(int{50});
    return para;
  }

  void compute() {
    const auto& eps = m_strain.get();
    const auto& C_e = m_C_e.get();
    const auto I = tmech::eye<value_type, Dim, 2>();
    const auto alpha_n = m_alpha.old_value();
    const auto eps_p_n = m_eps_p.old_value();

    // Trial stress
    const tensor2 sig_trial{tmech::dcontract(C_e, eps - eps_p_n)};
    const auto trace_sig = tmech::trace(sig_trial);
    const tensor2 sig_dev{sig_trial - (trace_sig / value_type{Dim}) * I};
    const auto sig_eq = yield_fn::equivalent_stress(sig_dev);

    // Elastic check
    m_alpha.new_value() = alpha_n;
    m_H.update_source();
    const auto F_trial = yield_fn::trial_yield(sig_eq, m_sigma_0, m_H.get());

    if (F_trial <= value_type{0}) {
      m_stress = sig_trial;
      m_tangent = C_e;
      m_eps_p.new_value() = eps_p_n;
      m_alpha.new_value() = alpha_n;
      return;
    }

    // Flow normal at trial state (cached for tangent)
    const tensor2 N_trial{yield_fn::flow_normal(sig_dev, sig_eq)};

    // Return mapping — single stage or multi-stage
    value_type total_dlambda;
    tensor2 eps_p_new;
    value_type alpha_new;

    if (!m_tableau) {
      // Single-stage: solver.solve() with lambda
      auto eval = [&](value_type dl) -> std::pair<value_type, value_type> {
        m_alpha.new_value() = alpha_n + dl;
        m_H.update_source();
        return {yield_fn::residual(sig_eq, dl, m_G, m_sigma_0, m_H.get()),
                yield_fn::jacobian(m_G, m_dH.get())};
      };
      total_dlambda = m_solver->get().solve(eval);
      eps_p_new = eps_p_n + total_dlambda * N_trial;
      alpha_new = alpha_n + total_dlambda;
    } else {
      // Multi-stage RK return mapping
      compute_rk_stages(C_e, eps, eps_p_n, alpha_n, sig_eq, N_trial, I);
      total_dlambda = value_type{0};
      eps_p_new = eps_p_n;
      alpha_new = alpha_n;
      const int s = m_tableau->stages();
      for (int i = 0; i < s; ++i) {
        eps_p_new = eps_p_new + m_tableau->b[i] * m_dlambda[i] * m_N_stage[i];
        alpha_new += m_tableau->b[i] * m_dlambda[i];
        total_dlambda += m_tableau->b[i] * m_dlambda[i];
      }
    }

    // Finalize
    m_eps_p.new_value() = eps_p_new;
    m_alpha.new_value() = alpha_new;
    m_stress = tmech::dcontract(C_e, eps - eps_p_new);

    // Consistent tangent via implicit function theorem (trial state)
    m_H.update_source();
    const auto dH_val = m_dH.get();
    const auto dr_ddlambda = yield_fn::jacobian(m_G, dH_val);
    const tensor2 dr_deps{tmech::dcontract(N_trial, C_e)};
    const tensor2 dlambda_deps{-dr_deps / dr_ddlambda};

    const tensor4 dN_dsig{yield_fn::flow_normal_stress_derivative(N_trial, sig_eq)};
    const tensor4 dN_deps{tmech::dcontract(dN_dsig, C_e)};
    const tensor4 dsig_deps{C_e - value_type{2} * m_G * total_dlambda * dN_deps};
    const tensor2 dsig_ddlambda{-value_type{2} * m_G * N_trial};

    m_tangent = dsig_deps + tmech::otimes(dsig_ddlambda, dlambda_deps);
  }

private:
  /// Multi-stage RK return mapping — fills m_dlambda and m_N_stage
  void compute_rk_stages(const tensor4& C_e, const tensor2& eps,
                          const tensor2& eps_p_n, value_type alpha_n,
                          value_type sig_eq_trial, const tensor2& N_trial,
                          const tensor2& I) {
    const auto& tab = *m_tableau;
    const int s = tab.stages();
    const auto tol = base::template get_parameter<value_type>("tolerance");
    const auto max_iter = base::template get_parameter<int>("max_iter");

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
        // Explicit stage
        const tensor2 sig_i{tmech::dcontract(C_e, eps - eps_p_acc)};
        const auto tr_i = tmech::trace(sig_i);
        const tensor2 dev_i{sig_i - (tr_i / value_type{Dim}) * I};
        const auto seq_i = yield_fn::equivalent_stress(dev_i);

        m_alpha.new_value() = alpha_acc;
        m_H.update_source();
        const auto F_i = yield_fn::trial_yield(seq_i, m_sigma_0, m_H.get());

        if (F_i > value_type{0} && seq_i > value_type{1e-30}) {
          m_N_stage[i] = yield_fn::flow_normal(dev_i, seq_i);
          m_dlambda[i] = F_i / (value_type{3} * m_G + m_dH.get());
        } else {
          m_N_stage[i] = tensor2{};
          m_dlambda[i] = value_type{0};
        }
      } else {
        // Implicit stage: Newton on F = 0
        const auto aii = m_diag[i];
        m_dlambda[i] = value_type{0};

        for (int iter = 0; iter < max_iter; ++iter) {
          tensor2 eps_p_i{eps_p_acc + aii * m_dlambda[i] * N_trial};
          auto alpha_i = alpha_acc + aii * m_dlambda[i];

          const tensor2 sig_i{tmech::dcontract(C_e, eps - eps_p_i)};
          const auto tr_i = tmech::trace(sig_i);
          const tensor2 dev_i{sig_i - (tr_i / value_type{Dim}) * I};
          const auto seq_i = yield_fn::equivalent_stress(dev_i);

          m_alpha.new_value() = alpha_i;
          m_H.update_source();

          const auto F_i = yield_fn::trial_yield(seq_i, m_sigma_0, m_H.get());
          if (std::abs(F_i) < tol) {
            if (seq_i > value_type{1e-30})
              m_N_stage[i] = yield_fn::flow_normal(dev_i, seq_i);
            break;
          }

          const auto dF_i = -aii * (value_type{3} * m_G + m_dH.get());
          m_dlambda[i] -= F_i / dF_i;
        }
      }
    }
  }

  tensor2& m_stress;
  tensor4& m_tangent;
  history_property<tensor2>& m_eps_p;
  history_property<value_type>& m_alpha;

  const value_type& m_G;
  const value_type& m_sigma_0;
  const std::string& m_solver_name;
  material_ref<solver_type, Traits>* m_solver;
  const std::string& m_elastic_source;
  const std::string& m_hardening_source;
  const std::string& m_strain_source;

  const input_property<tensor4, property_traits>& m_C_e;
  const input_property<tensor2, property_traits>& m_strain;
  const input_property<value_type, property_traits>& m_H;
  const input_property<value_type, property_traits>& m_dH;

  // Multi-stage storage (empty if no tableau)
  const butcher_tableau* m_tableau{nullptr};
  std::vector<value_type> m_dlambda;
  std::vector<tensor2> m_N_stage;
  std::vector<bool> m_is_implicit;
  std::vector<double> m_diag;
};

template<typename Traits>
using j2_plasticity = small_strain_plasticity<Traits,
    j2_yield_function<typename Traits::value_type, Traits::Dim>>;

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_SMALL_STRAIN_PLASTICITY_H
