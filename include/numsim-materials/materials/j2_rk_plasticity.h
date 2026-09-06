#ifndef NUMSIM_MATERIALS_J2_RK_PLASTICITY_H
#define NUMSIM_MATERIALS_J2_RK_PLASTICITY_H

#include <cmath>
#include <stdexcept>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/materials/yield_functions.h"
#include "numsim-materials/materials/plasticity_utils.h"
#include "numsim-materials/solvers/butcher_tableau.h"

namespace numsim::materials {

/// Multi-stage Runge-Kutta return mapping for small-strain plasticity.
///
/// Applies a Butcher tableau to the plasticity evolution equations.
/// Each implicit stage solves F = 0 for Δλ_i. Explicit stages use
/// the consistency condition. Shares trial/tangent code with
/// small_strain_plasticity via plasticity_utils.h.
template<typename Traits>
class j2_rk_plasticity final
    : public material_base<j2_rk_plasticity<Traits>, Traits> {
public:
  using base = material_base<j2_rk_plasticity<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  static constexpr auto Dim = base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;
  using yield_fn = j2_yield_function<value_type, base::Dim>;

  template <typename... Args>
  explicit j2_rk_plasticity(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_stress(base::template add_output<tensor2>(
            "stress", &j2_rk_plasticity::compute)),
        m_tangent(base::template add_output<tensor4>("tangent")),
        m_eps_p(base::template add_history_output<tensor2>("plastic_strain")),
        m_kappa(base::template add_history_output<value_type>("equivalent_plastic_strain")),
        m_K(base::template get_parameter<value_type>("K")),
        m_G(base::template get_parameter<value_type>("G")),
        m_sigma_0(base::template get_parameter<value_type>("sigma_0")),
        m_tol(base::template get_parameter<value_type>("tolerance")),
        m_max_iter(base::template get_parameter<int>("max_iter")),
        m_tableau(tableau_by_name(
            base::template get_parameter<std::string>("tableau"))),
        m_hardening_source(base::template get_parameter<std::string>("hardening_source")),
        m_strain_source(base::template get_parameter<std::string>("strain_source")),
        m_strain(base::template add_input<tensor2>(
            m_strain_source, "strain", EdgeKind::Global)),
        m_H(base::template add_input<value_type>(
            m_hardening_source, "hardening_stress", EdgeKind::Local)),
        m_dH(base::template add_input<value_type>(
            m_hardening_source, "hardening_modulus", EdgeKind::Local))
  {
    const auto I = tmech::eye<value_type, Dim, 2>();
    const tensor4 IIvol{tmech::otimes(I, I) / value_type{Dim}};
    m_C_e = value_type{3} * m_K * IIvol +
            value_type{2} * m_G * plasticity_detail::make_IIdev<value_type, Dim>();

    // The stage loop below accumulates only a(i,j) for j < i, plus the
    // diagonal. A tableau with a(i,j) != 0 for j > i would have those terms
    // silently DROPPED -- integrating with a method that is not the one named.
    // Measured with gauss_legendre_4 before this guard: equivalent plastic
    // strain of -8.8 (negative) and a yield residual of +10880.
    //
    // rk_integrator dispatches that case to a fully-implicit solve; this class
    // has no such path, so it refuses rather than pretending.
    if (!m_tableau.is_dirk())
      throw std::invalid_argument(
          "j2_rk_plasticity: the scheme '" +
          base::template get_parameter<std::string>("tableau") +
          "' is fully implicit (it has coupling above the diagonal), which this "
          "return map cannot integrate -- its stage loop sums only j < i. Use a "
          "DIRK or explicit scheme: forward_euler, explicit_midpoint, rk4, "
          "implicit_euler, implicit_midpoint, crank_nicolson, sdirk3");

    const int s = m_tableau.stages();
    m_dlambda.resize(s, value_type{0});
    m_N_stage.resize(s);
    m_is_implicit.resize(s);
    m_diag.resize(s);
    for (int i = 0; i < s; ++i) {
      m_diag[i] = m_tableau.a(i, i);
      m_is_implicit[i] = std::abs(m_diag[i]) >= 1e-30;
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("hardening_source").template add<is_required>();
    para.template insert<std::string>("strain_source").template add<is_required>();
    para.template insert<value_type>("K").template add<is_required>();
    para.template insert<value_type>("G").template add<is_required>();
    para.template insert<value_type>("sigma_0").template add<is_required>();
    // The scheme by NAME. This parameter was never in the schema at all --
    // the constructor read it while parameters() never declared it, so it
    // could only ever be supplied from C++, where insert() bypasses the
    // schema. That is why the explicit/implicit choice, which is the whole
    // point of a tableau, was unreachable from a document.
    para.template insert<std::string>("tableau").template add<is_required>();
    para.template insert<value_type>("tolerance")
        .template add<set_default>(value_type{1e-12});
    para.template insert<int>("max_iter")
        .template add<set_default>(int{50});
    return para;
  }

  void compute() {
    const auto& C_e = m_C_e;
    const auto& eps = m_strain.get();
    const auto kappa_n = m_kappa.old_value();
    const auto eps_p_n = m_eps_p.old_value();
    m_kappa.new_value() = kappa_n;
    m_H.update_source();

    auto ts = plasticity_detail::compute_trial<value_type, Dim>(
        m_yf, eps, eps_p_n, C_e, m_sigma_0, m_H.get());

    if (!ts.yielding) {
      m_stress = ts.eval.sig;
      m_tangent = C_e;
      m_eps_p.new_value() = eps_p_n;
      m_kappa.new_value() = kappa_n;
      return;
    }

    // Multi-stage return mapping
    const auto& tab = m_tableau;
    const int s = tab.stages();

    for (int i = 0; i < s; ++i)
      m_dlambda[i] = value_type{0};

    for (int i = 0; i < s; ++i) {
      tensor2 eps_p_acc{eps_p_n};
      auto kappa_acc = kappa_n;
      for (int j = 0; j < i; ++j) {
        eps_p_acc = eps_p_acc + tab.a(i, j) * m_dlambda[j] * m_N_stage[j];
        kappa_acc += tab.a(i, j) * m_dlambda[j];
      }

      if (!m_is_implicit[i]) {
        // Explicit stage
        m_kappa.new_value() = kappa_acc;
        m_H.update_source();
        auto se = plasticity_detail::evaluate_at_state<value_type, Dim>(
            m_yf, eps, eps_p_acc, C_e, m_sigma_0, m_H.get());

        if (se.F > value_type{0} && se.sig_eq > value_type{1e-30}) {
          m_N_stage[i] = se.N;
          m_dlambda[i] = -se.F / m_yf.jacobian(m_G, m_dH.get());
        } else {
          m_N_stage[i] = tensor2{};
          m_dlambda[i] = value_type{0};
        }
      } else {
        // Implicit stage: Newton on F = 0
        const auto aii = m_diag[i];
        m_dlambda[i] = value_type{0};

        for (int iter = 0; iter < m_max_iter; ++iter) {
          tensor2 eps_p_i{eps_p_acc + aii * m_dlambda[i] * ts.eval.N};
          auto kappa_i = kappa_acc + aii * m_dlambda[i];

          m_kappa.new_value() = kappa_i;
          m_H.update_source();
          auto se = plasticity_detail::evaluate_at_state<value_type, Dim>(
              m_yf, eps, eps_p_i, C_e, m_sigma_0, m_H.get());

          if (std::abs(se.F) < m_tol) {
            m_N_stage[i] = se.N;
            break;
          }

          const auto dF_i = aii * m_yf.jacobian(m_yf.effective_modulus(m_G), m_dH.get());
          m_dlambda[i] -= se.F / dF_i;
        }
      }
    }

    // Final update
    tensor2 eps_p_new{eps_p_n};
    auto kappa_new = kappa_n;
    auto total_dlambda = value_type{0};
    for (int i = 0; i < s; ++i) {
      eps_p_new = eps_p_new + tab.b[i] * m_dlambda[i] * m_N_stage[i];
      kappa_new += tab.b[i] * m_dlambda[i];
      total_dlambda += tab.b[i] * m_dlambda[i];
    }

    m_eps_p.new_value() = eps_p_new;
    m_kappa.new_value() = kappa_new;
    m_stress = tmech::dcontract(C_e, eps - eps_p_new);

    m_H.update_source();
    // Evaluate at converged state for correct tangent (non-associative)
    auto converged = plasticity_detail::evaluate_at_state<value_type, Dim>(
        m_yf, eps, eps_p_new, C_e, m_sigma_0, m_H.get());
    m_tangent = plasticity_detail::compute_tangent<value_type, Dim>(
        m_yf, converged.sig_dev, converged.N, converged.sig_eq,
        total_dlambda, m_dH.get(), C_e, m_G);
  }

private:
  tensor2& m_stress;
  tensor4& m_tangent;
  history_property<tensor2>& m_eps_p;
  history_property<value_type>& m_kappa;

  const value_type& m_K;
  const value_type& m_G;
  const value_type& m_sigma_0;
  const value_type& m_tol;
  const int& m_max_iter;
  const butcher_tableau m_tableau;
  const std::string& m_hardening_source;
  const std::string& m_strain_source;

  const input_property<tensor2, property_traits>& m_strain;
  const input_property<value_type, property_traits>& m_H;
  const input_property<value_type, property_traits>& m_dH;

  /// Built here, not read from an elastic_source. compute_tangent's collapse
  /// (C_e : X = 2G X) and effective_modulus(G) = 3G both require an isotropic
  /// C_e, so taking an arbitrary rank-4 tangent promised more than this class
  /// can deliver -- and pulled a linear_elasticity into the graph whose own
  /// "stress" output is meaningless once eps_p is nonzero.
  tensor4 m_C_e{};

  yield_fn m_yf{};
  std::vector<value_type> m_dlambda;
  std::vector<tensor2> m_N_stage;
  std::vector<bool> m_is_implicit;
  std::vector<double> m_diag;
};


} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_J2_RK_PLASTICITY_H
