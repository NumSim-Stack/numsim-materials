#ifndef NUMSIM_MATERIALS_SMALL_STRAIN_PLASTICITY_H
#define NUMSIM_MATERIALS_SMALL_STRAIN_PLASTICITY_H

#include <cmath>
#include <utility>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_ref.h"
#include "numsim-materials/materials/yield_functions.h"
#include "numsim-materials/materials/plasticity_utils.h"
#include "numsim-materials/solvers/backward_euler.h"

namespace numsim::materials {

/// Single-stage implicit Euler plasticity (classical return mapping).
///
/// Uses solver.solve() for the Newton iteration. No tableau, no stage
/// vectors, no overhead. This is the standard radial return for J2.
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
  {
    if (base::m_parameter_handler.contains("yield_function"))
      m_yf = base::template get_parameter<yield_fn>("yield_function");
  }

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
    const auto& C_e = m_C_e.get();
    const auto alpha_n = m_alpha.old_value();

    m_alpha.new_value() = alpha_n;
    m_H.update_source();

    auto ts = plasticity_detail::compute_trial<value_type, Dim>(
        m_yf, m_strain.get(), m_eps_p.old_value(), C_e, m_sigma_0, m_H.get());

    if (!ts.yielding) {
      m_stress = ts.eval.sig;
      m_tangent = C_e;
      m_eps_p.new_value() = m_eps_p.old_value();
      m_alpha.new_value() = alpha_n;
      return;
    }

    // Return mapping via solver
    auto eval = [&](value_type dl) -> std::pair<value_type, value_type> {
      m_alpha.new_value() = alpha_n + dl;
      m_H.update_source();
      return {m_yf.residual(ts.eval.sig_eq, dl, m_G, m_sigma_0, m_H.get()),
              m_yf.jacobian(m_G, m_dH.get())};
    };

    const auto dlambda = m_solver.get().solve(eval);

    m_eps_p.new_value() = m_eps_p.old_value() + dlambda * ts.eval.N;
    m_alpha.new_value() = alpha_n + dlambda;
    m_stress = tmech::dcontract(C_e, m_strain.get() - m_eps_p.new_value());

    m_H.update_source();
    m_tangent = plasticity_detail::compute_tangent<value_type, Dim>(
        m_yf, ts.eval.N, ts.eval.sig_eq, dlambda, m_G, m_dH.get(), C_e);
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
  yield_fn m_yf{};
};

template<typename Traits>
using j2_plasticity = small_strain_plasticity<Traits,
    j2_yield_function<typename Traits::value_type, Traits::Dim>>;

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_SMALL_STRAIN_PLASTICITY_H
