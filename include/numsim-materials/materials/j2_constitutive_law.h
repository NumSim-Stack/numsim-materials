#ifndef NUMSIM_MATERIALS_J2_CONSTITUTIVE_LAW_H
#define NUMSIM_MATERIALS_J2_CONSTITUTIVE_LAW_H

#include <cmath>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/materials/yield_functions.h"

namespace numsim::materials {

/// Pure constitutive law for J2 plasticity — no history, no solver.
///
/// Evaluates the yield function and flow direction at a trial state
/// (eps_p, alpha) provided by the integrator via Local edges.
///
/// Outputs:
///   "sigma"          — tensor2: stress at trial state
///   "flow_normal"    — tensor2: N = 3/2 · dev(σ) / σ_eq
///   "yield_function" — scalar: F = σ_eq - σ_0 - H(α)
///   "yield_jacobian" — scalar: dF/dΔλ = -3G - dH/dα
///   "sig_eq"         — scalar: von Mises equivalent stress
///   "yield_active"   — int: 1 if F > 0, 0 otherwise
///
/// Inputs (Global): strain, elastic tangent
/// Inputs (Local): eps_p, alpha from integrator; H, dH from hardening
template<typename Traits,
         typename YieldFunction = j2_yield_function<typename Traits::value_type, Traits::Dim>>
class j2_constitutive_law final
    : public material_base<j2_constitutive_law<Traits, YieldFunction>, Traits> {
public:
  using base = material_base<j2_constitutive_law<Traits, YieldFunction>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  static constexpr auto Dim = base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;
  using yield_fn = YieldFunction;

  template <typename... Args>
  explicit j2_constitutive_law(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_sigma(base::template add_output<tensor2>(
            "sigma", &j2_constitutive_law::compute)),
        m_N(base::template add_output<tensor2>("flow_normal")),
        m_F(base::template add_output<value_type>("yield_function")),
        m_dF(base::template add_output<value_type>("yield_jacobian")),
        m_sig_eq(base::template add_output<value_type>("sig_eq")),
        m_yield_active(base::template add_output<int>("yield_active")),
        m_G(base::template get_parameter<value_type>("G")),
        m_sigma_0(base::template get_parameter<value_type>("sigma_0")),
        m_elastic_source(base::template get_parameter<std::string>("elastic_source")),
        m_hardening_source(base::template get_parameter<std::string>("hardening_source")),
        m_strain_source(base::template get_parameter<std::string>("strain_source")),
        m_integrator_source(base::template get_parameter<std::string>("integrator_source")),
        m_C_e(base::template add_input<tensor4>(
            m_elastic_source, "tangent", EdgeKind::Global)),
        m_strain(base::template add_input<tensor2>(
            m_strain_source, "strain", EdgeKind::Global)),
        m_eps_p(base::template add_input<tensor2>(
            m_integrator_source, "plastic_strain", EdgeKind::Local)),
        m_alpha(base::template add_input<value_type>(
            m_integrator_source, "equivalent_plastic_strain", EdgeKind::Local)),
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
    para.template insert<std::string>("integrator_source").template add<is_required>();
    para.template insert<value_type>("G").template add<is_required>();
    para.template insert<value_type>("sigma_0").template add<is_required>();
    return para;
  }

  void compute() {
    const auto& eps = m_strain.get();
    const auto& C_e = m_C_e.get();
    const auto I = tmech::eye<value_type, Dim, 2>();

    const tensor2 eps_elastic{eps - m_eps_p.get()};
    m_sigma = tmech::dcontract(C_e, eps_elastic);

    const auto trace_sig = tmech::trace(m_sigma);
    const tensor2 sig_dev{m_sigma - (trace_sig / value_type{Dim}) * I};
    m_sig_eq = yield_fn::equivalent_stress(sig_dev);

    m_H.update_source();
    m_F = yield_fn::trial_yield(m_sig_eq, m_sigma_0, m_H.get());
    m_dF = yield_fn::jacobian(m_G, m_dH.get());

    m_yield_active = (m_F > value_type{0}) ? 1 : 0;

    if (m_sig_eq > value_type{1e-30})
      m_N = yield_fn::flow_normal(sig_dev, m_sig_eq);
    else
      m_N = tensor2{};
  }

private:
  tensor2& m_sigma;
  tensor2& m_N;
  value_type& m_F;
  value_type& m_dF;
  value_type& m_sig_eq;
  int& m_yield_active;

  const value_type& m_G;
  const value_type& m_sigma_0;
  const std::string& m_elastic_source;
  const std::string& m_hardening_source;
  const std::string& m_strain_source;
  const std::string& m_integrator_source;

  const input_property<tensor4, property_traits>& m_C_e;
  const input_property<tensor2, property_traits>& m_strain;
  const input_property<tensor2, property_traits>& m_eps_p;
  const input_property<value_type, property_traits>& m_alpha;
  const input_property<value_type, property_traits>& m_H;
  const input_property<value_type, property_traits>& m_dH;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_J2_CONSTITUTIVE_LAW_H
