#ifndef NUMSIM_MATERIALS_SMALL_STRAIN_PLASTICITY_H
#define NUMSIM_MATERIALS_SMALL_STRAIN_PLASTICITY_H

#include <cmath>
#include <utility>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/materials/yield_functions.h"
#include "numsim-materials/solvers/backward_euler.h"

namespace numsim::materials {

/// Generic small-strain plasticity with pluggable yield function.
///
/// The return mapping calls an external solver material's solve() method.
/// Consistent tangent derived via implicit function theorem.
///
/// Outputs:
///   "stress"                      — tensor2: corrected stress
///   "tangent"                     — tensor4: consistent (algorithmic) tangent
///   "plastic_strain"              — tensor2 (history): ε_p
///   "equivalent_plastic_strain"   — scalar (history): α
///
/// Inputs (Global):
///   elastic_source::tangent       — C_e (elastic tangent)
///   strain_source::strain         — total strain ε
///
/// Inputs (Local — re-evaluated in inner Newton loop):
///   hardening_source::hardening_stress      — H(α)
///   hardening_source::hardening_modulus     — dH/dα
///
/// Parameters:
///   "solver" — pointer to a solver material (e.g., newton_raphson)
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
        m_solver(base::template add_material_ref<solver_type>(
            base::template get_parameter<std::string>("solver_source"))),
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
  {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("elastic_source").template add<is_required>();
    para.template insert<std::string>("hardening_source").template add<is_required>();
    para.template insert<std::string>("strain_source").template add<is_required>();
    para.template insert<std::string>("solver_source").template add<is_required>();
    para.template insert<value_type>("G").template add<is_required>();
    para.template insert<value_type>("sigma_0").template add<is_required>();
    return para;
  }

  void compute() {
    const auto& eps = m_strain.get();
    const auto& C_e = m_C_e.get();
    const auto I = tmech::eye<value_type, Dim, 2>();
    const auto alpha_n = m_alpha.old_value();

    // 1. Trial stress: σ_trial = C_e : (ε - ε_p_old)
    const tensor2 eps_elastic{eps - m_eps_p.old_value()};
    const tensor2 sig_trial{tmech::dcontract(C_e, eps_elastic)};

    // 2. Deviatoric trial stress and equivalent stress
    const auto trace_sig = tmech::trace(sig_trial);
    const tensor2 sig_dev{sig_trial - (trace_sig / value_type{Dim}) * I};
    const auto sig_eq = yield_fn::equivalent_stress(sig_dev);

    // 3. Elastic check
    m_alpha.new_value() = alpha_n;
    m_H.update_source();
    const auto F_trial = yield_fn::trial_yield(sig_eq, m_sigma_0, m_H.get());

    if (F_trial <= value_type{0}) {
      m_stress = sig_trial;
      m_tangent = C_e;
      m_eps_p.new_value() = m_eps_p.old_value();
      m_alpha.new_value() = alpha_n;
      return;
    }

    // 4. Return mapping via external solver
    auto eval = [&](value_type dlambda) -> std::pair<value_type, value_type> {
      m_alpha.new_value() = alpha_n + dlambda;
      m_H.update_source();
      auto r = yield_fn::residual(sig_eq, dlambda, m_G, m_sigma_0, m_H.get());
      auto dr = yield_fn::jacobian(m_G, m_dH.get());
      return {r, dr};
    };

    const auto dlambda = m_solver.get().solve(eval);

    // 5. Final hardening values at converged state
    m_alpha.new_value() = alpha_n + dlambda;
    m_H.update_source();
    const auto dH_val = m_dH.get();

    // 6. Converged state
    const tensor2 N{yield_fn::flow_normal(sig_dev, sig_eq)};
    m_stress = sig_trial - value_type{2} * m_G * dlambda * N;
    m_eps_p.new_value() = m_eps_p.old_value() + dlambda * N;

    // 7. Consistent tangent via implicit function theorem
    const auto dr_ddlambda = yield_fn::jacobian(m_G, dH_val);
    const tensor2 dr_deps{tmech::dcontract(N, C_e)};
    const tensor2 dlambda_deps{-dr_deps / dr_ddlambda};

    const tensor4 dN_dsig{yield_fn::flow_normal_stress_derivative(N, sig_eq)};
    const tensor4 dN_deps{tmech::dcontract(dN_dsig, C_e)};
    const tensor4 dsig_deps{C_e - value_type{2} * m_G * dlambda * dN_deps};
    const tensor2 dsig_ddlambda{-value_type{2} * m_G * N};

    m_tangent = dsig_deps + tmech::otimes(dsig_ddlambda, dlambda_deps);
  }

private:
  tensor2& m_stress;
  tensor4& m_tangent;
  history_property<tensor2>& m_eps_p;
  history_property<value_type>& m_alpha;

  const value_type& m_G;
  const value_type& m_sigma_0;
  material_ref<solver_type, Traits>& m_solver;
  const std::string& m_elastic_source;
  const std::string& m_hardening_source;
  const std::string& m_strain_source;

  const input_property<tensor4, property_traits>& m_C_e;
  const input_property<tensor2, property_traits>& m_strain;
  const input_property<value_type, property_traits>& m_H;
  const input_property<value_type, property_traits>& m_dH;
};

template<typename Traits>
using j2_plasticity = small_strain_plasticity<Traits,
    j2_yield_function<typename Traits::value_type, Traits::Dim>>;

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_SMALL_STRAIN_PLASTICITY_H
