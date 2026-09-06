#ifndef J2_PLASTICITY_H
#define J2_PLASTICITY_H

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_ref.h"
#include "numsim-materials/materials/plasticity_utils.h"
#include "numsim-materials/solvers/local_newton.h"

namespace numsim::materials {

/// J2 (von Mises) plasticity with isotropic hardening, radial return.
///
/// Split out of a class templated on the yield function, which took both this
/// model and Drucker-Prager. That class
/// is parameterised over a yield function so it can also serve Drucker-Prager,
/// and J2 paid for the generality without using it:
///
///   - a yield normal distinct from the flow normal, which for associative J2
///     is the same tensor;
///   - a "modified" equivalent stress carrying pressure coupling J2 does not
///     have;
///   - an apex branch a cylinder cannot reach.
///
/// The cost was not only readability. The general consistent tangent forms
/// C_e : (dN/dsigma) : C_e -- two rank-4 x rank-4 contractions per plastic
/// step. For isotropic elasticity N is deviatoric, so C_e : N = 2G N and
/// N : C_e : N = 3G, and the whole expression collapses:
///
///   C = C_e - (6G^2 dl / sig_eq) IIdev
///           + (4G^2 dl / sig_eq - 4G^2 / (3G + H')) N (x) N
///
/// which is the standard closed form. It is algebraically identical to what
/// the general path computes, not an approximation of it.
///
/// ISOTROPY. The closed form uses C_e : N = 2G N. That is not a new
/// restriction: the class this was split from already assumed isotropic elasticity
/// through effective_modulus(G) = 3G in its residual, so both paths are
/// equally limited. Here it is stated rather than implied.
///
/// The scalar Newton is kept. Linear hardening converges in one iteration and
/// has a closed form, but the hardening material is a separate graph node and
/// may be nonlinear (see exponential_isotropic_hardening).
///
/// Parameters:
///   "name", "hardening_source", "strain_source", "solver_source",
///   "K", "G", "sigma_0"
///
/// No elastic_source: the stiffness is built from K and G here.
template <typename Traits>
class j2_plasticity final
    : public material_base<j2_plasticity<Traits>, Traits> {
public:
  using base = material_base<j2_plasticity<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;
  using solver_type = local_newton<Traits>;

  template <typename... Args>
  explicit j2_plasticity(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_stress(base::template add_output<tensor2>(
            "stress", &j2_plasticity::compute)),
        m_tangent(base::template add_output<tensor4>("tangent")),
        m_eps_p(base::template add_history_output<tensor2>("plastic_strain")),
        m_kappa(base::template add_history_output<value_type>(
            "equivalent_plastic_strain")),
        m_G(base::template get_parameter<value_type>("G")),
        m_sigma_0(base::template get_parameter<value_type>("sigma_0")),
        m_K(base::template get_parameter<value_type>("K")),
        m_solver(base::template add_material_ref<solver_type>(
            base::template get_parameter<std::string>("solver_source"))),
        m_strain(base::template add_input<tensor2>(
            base::template get_parameter<std::string>("strain_source"),
            "strain", EdgeKind::Global)),
        m_H(base::template add_input<value_type>(
            base::template get_parameter<std::string>("hardening_source"),
            "hardening_stress", EdgeKind::Local)),
        m_dH(base::template add_input<value_type>(
            base::template get_parameter<std::string>("hardening_source"),
            "hardening_modulus", EdgeKind::Local)),
        m_IIdev(plasticity_detail::make_IIdev<value_type, Dim>()),
        m_C_e(plasticity_detail::make_isotropic_tangent<value_type, Dim>(
            m_K, m_G))
  {}

  /// m_C_e above is the elastic stiffness, built here rather than read from
  /// another material.
  ///
  /// The closed forms below REQUIRE an isotropic C_e -- that is what makes
  /// C_e : N = 2G N and N : C_e : N = 3G true. Accepting an arbitrary rank-4
  /// tangent from an elastic_source advertised a generality this material
  /// cannot honour, and dragged a linear_elasticity into every plasticity graph
  /// whose own "stress" output (C : eps, ignoring eps_p) is meaningless once
  /// yielding starts.
  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("hardening_source")
        .template add<is_required>();
    para.template insert<std::string>("strain_source")
        .template add<is_required>();
    para.template insert<std::string>("solver_source")
        .template add<is_required>();
    para.template insert<value_type>("K").template add<is_required>();
    para.template insert<value_type>("G").template add<is_required>();
    para.template insert<value_type>("sigma_0").template add<is_required>();
    return para;
  }

  void compute() {
    const auto& C_e = m_C_e;
    const auto kappa_n = m_kappa.old_value();

    m_kappa.new_value() = kappa_n;
    m_H.update_source();

    // Trial state: freeze the plastic strain and load elastically.
    const tensor2 sig_trial{
        tmech::dcontract(C_e, tensor2(m_strain.get() - m_eps_p.old_value()))};
    const tensor2 s{tmech::dev(sig_trial)};
    const auto sig_eq =
        std::sqrt(value_type{1.5} * tmech::dcontract(s, s));

    if (sig_eq - m_sigma_0 - m_H.get() <= value_type{0}) {
      m_stress = sig_trial;
      m_tangent = C_e;
      m_eps_p.new_value() = m_eps_p.old_value();
      m_kappa.new_value() = kappa_n;
      return;
    }

    // Radial return: the flow direction is fixed by the trial state, so only
    // the magnitude is solved for.
    const tensor2 N{value_type{1.5} * s / sig_eq};
    const auto G_eff = value_type{3} * m_G;

    auto eval = [&](value_type dl) -> std::pair<value_type, value_type> {
      m_kappa.new_value() = kappa_n + dl;
      m_H.update_source();
      return {sig_eq - G_eff * dl - m_sigma_0 - m_H.get(),
              -G_eff - m_dH.get()};
    };
    const auto sol = m_solver.get().solve(eval);
    if (!sol.converged)
      throw std::runtime_error(
          "j2_plasticity: return-mapping Newton failed to converge");
    // A negative multiplier is not a rounding artefact to be clamped away.
    // The return map is only entered with F_trial > 0, and the root is
    // F_trial / (G_eff + H'), so dlambda < 0 means G_eff + H' < 0: the material
    // softens faster than it can unload elastically, the yield residual has
    // positive slope, and the local problem has no admissible solution.
    // Clamping to zero returns the elastic stress -- outside the yield surface,
    // with no plastic flow and no indication. Measured on a uniaxial path with
    // H' = -300 against 3G = 230.8, the equivalent stress climbed past
    // sigma_0 = 50 to 76.9 with alpha identically zero.
    // Moderate softening, -G_eff < H' < 0, still gives a positive root and is
    // untouched.
    if (sol.x < value_type{0})
      throw std::runtime_error(
          "j2_plasticity: the return map produced a negative plastic "
          "multiplier, which means the hardening modulus is at or below "
          "-3G. The softening branch is unstable and this local return map "
          "cannot resolve it; use a smaller magnitude, or a formulation with "
          "viscous or gradient regularisation.");
    // dlambda >= 0 is a statement about the plastic multiplier, so it is
    // enforced here rather than inside a general scalar solver (see #13).
    const auto dlambda = std::max(sol.x, value_type{0});

    m_eps_p.new_value() = m_eps_p.old_value() + dlambda * N;
    m_kappa.new_value() = kappa_n + dlambda;

    // sigma = C_e : (eps - eps_p_new)
    //       = C_e : (eps - eps_p_old) - dl (C_e : N)
    //       = sig_trial - 2G dl N,           since C_e : N = 2G N.
    // The same identity the tangent uses. Avoids a second rank-4 : rank-2
    // contraction, which measures 33 ns against ~245 for the whole step.
    m_stress = sig_trial - value_type{2} * m_G * dlambda * N;

    m_H.update_source();
    const auto GG = m_G * m_G;
    const auto a = value_type{6} * GG * dlambda / sig_eq;
    const auto b = value_type{4} * GG * dlambda / sig_eq -
                   value_type{4} * GG / (G_eff + m_dH.get());
    m_tangent = C_e - a * m_IIdev + b * tmech::otimes(N, N);
  }

private:
  tensor2& m_stress;
  tensor4& m_tangent;
  history_property<tensor2>& m_eps_p;
  history_property<value_type>& m_kappa;

  const value_type& m_G;
  const value_type& m_sigma_0;
  const value_type& m_K;
  material_ref<solver_type, Traits>& m_solver;

  const input_property<tensor2, property_traits>& m_strain;
  const input_property<value_type, property_traits>& m_H;
  const input_property<value_type, property_traits>& m_dH;

  const tensor4 m_IIdev;
  const tensor4 m_C_e;
};

}  // namespace numsim::materials

#endif  // J2_PLASTICITY_H
