#ifndef NUMSIM_MATERIALS_IMPLICIT_EULER_PLASTICITY_H
#define NUMSIM_MATERIALS_IMPLICIT_EULER_PLASTICITY_H

#include <cmath>
#include <utility>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_ref.h"
#include "numsim-materials/materials/yield_functions.h"
#include "numsim-materials/solvers/backward_euler.h"

namespace numsim::materials {

/// Implicit Euler (return mapping) integrator for plasticity.
///
/// Owns the history state (ε_p, α). Calls the solver's solve() method
/// with a lambda that re-evaluates the constitutive law at each Newton step.
///
/// The constitutive law is a pure function evaluated via Local update_source().
/// The solver is accessed via material_ref (resolved at finalize).
///
/// Outputs:
///   "stress"                      — tensor2: corrected stress
///   "tangent"                     — tensor4: algorithmic tangent
///   "plastic_strain"              — tensor2 (history): ε_p
///   "equivalent_plastic_strain"   — scalar (history): α
///
/// Inputs (Global):
///   elastic_source::tangent       — C_e
///   law_source::sigma, flow_normal, yield_function, yield_jacobian, sig_eq, yield_active
///
/// Solver accessed via material_ref (not property graph edges).
template<typename Traits>
class implicit_euler_plasticity final
    : public material_base<implicit_euler_plasticity<Traits>, Traits> {
public:
  using base = material_base<implicit_euler_plasticity<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  static constexpr auto Dim = base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;
  using yield_fn = j2_yield_function<value_type, Dim>;
  using solver_type = backward_euler<Traits>;

  template <typename... Args>
  explicit implicit_euler_plasticity(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_stress(base::template add_output<tensor2>(
            "stress", &implicit_euler_plasticity::compute)),
        m_tangent(base::template add_output<tensor4>("tangent")),
        m_eps_p(base::template add_history_output<tensor2>("plastic_strain")),
        m_alpha(base::template add_history_output<value_type>("equivalent_plastic_strain")),
        m_G(base::template get_parameter<value_type>("G")),
        m_solver(base::template add_material_ref<solver_type>(
            base::template get_parameter<std::string>("solver_source"))),
        m_law_source(base::template get_parameter<std::string>("law_source")),
        m_elastic_source(base::template get_parameter<std::string>("elastic_source")),
        m_C_e(base::template add_input<tensor4>(
            m_elastic_source, "tangent", EdgeKind::Global)),
        m_law_sigma(base::template add_input<tensor2>(
            m_law_source, "sigma", EdgeKind::Global)),
        m_law_N(base::template add_input<tensor2>(
            m_law_source, "flow_normal", EdgeKind::Global)),
        m_law_F(base::template add_input<value_type>(
            m_law_source, "yield_function", EdgeKind::Global)),
        m_law_dF(base::template add_input<value_type>(
            m_law_source, "yield_jacobian", EdgeKind::Global)),
        m_law_active(base::template add_input<int>(
            m_law_source, "yield_active", EdgeKind::Global)),
        m_law_sig_eq(base::template add_input<value_type>(
            m_law_source, "sig_eq", EdgeKind::Global))
  {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("law_source").template add<is_required>();
    para.template insert<std::string>("elastic_source").template add<is_required>();
    para.template insert<std::string>("solver_source").template add<is_required>();
    para.template insert<value_type>("G").template add<is_required>();
    return para;
  }

  void compute() {
    const auto& C_e = m_C_e.get();
    const auto alpha_n = m_alpha.old_value();

    if (m_law_active.get() == 0) {
      m_stress = m_law_sigma.get();
      m_tangent = C_e;
      m_eps_p.new_value() = m_eps_p.old_value();
      m_alpha.new_value() = alpha_n;
      return;
    }

    // Return mapping via solver.solve()
    // Lambda re-evaluates constitutive law at each trial (α_n + Δλ)
    auto eval = [&](value_type dlambda) -> std::pair<value_type, value_type> {
      // Write trial state so constitutive law re-evaluates
      m_alpha.new_value() = alpha_n + dlambda;
      m_eps_p.new_value() = m_eps_p.old_value() + dlambda * m_law_N.get();
      m_law_sigma.update_source();  // triggers law::compute()
      return {m_law_F.get(), m_law_dF.get()};
    };

    const auto dlambda = m_solver.get().solve(eval);

    // Finalize at converged state
    const auto& N = m_law_N.get();
    m_stress = m_law_sigma.get() - value_type{2} * m_G * dlambda * N;
    m_eps_p.new_value() = m_eps_p.old_value() + dlambda * N;
    m_alpha.new_value() = alpha_n + dlambda;

    // Algorithmic tangent via implicit function theorem
    const auto sig_eq = m_law_sig_eq.get();
    const auto dr_ddlambda = m_law_dF.get();
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
  material_ref<solver_type, Traits>& m_solver;
  const std::string& m_law_source;
  const std::string& m_elastic_source;

  const input_property<tensor4, property_traits>& m_C_e;
  const input_property<tensor2, property_traits>& m_law_sigma;
  const input_property<tensor2, property_traits>& m_law_N;
  const input_property<value_type, property_traits>& m_law_F;
  const input_property<value_type, property_traits>& m_law_dF;
  const input_property<int, property_traits>& m_law_active;
  const input_property<value_type, property_traits>& m_law_sig_eq;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_IMPLICIT_EULER_PLASTICITY_H
