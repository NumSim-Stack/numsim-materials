#ifndef J2_KINEMATIC_PLASTICITY_H
#define J2_KINEMATIC_PLASTICITY_H

#include <cmath>
#include <stdexcept>
#include <string>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_ref.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/materials/plasticity_utils.h"
#include "numsim-materials/solvers/vector_newton.h"
#include "numsim-materials/core/errors.h"

namespace numsim::materials {

/// J2 plasticity with Armstrong-Frederick kinematic hardening, returned as ONE
/// coupled system over {dlambda (scalar), dbeta (symmetric tensor)}.
///
/// The tensor unknown is the INCREMENT dbeta = beta - beta_n, so the solver's
/// zero seed is the trial state. With s_tr = 2G dev(eps - eps_p,n),
/// eta = s_tr - dev(beta), q = sigma_eq(eta) and N = (3/2) eta / q:
///
///   R_l = q - 3G dl - sigma_0 - H(kappa_n + dl)         yield on the RETURNED state
///   R_b = (1 + gamma dl) beta - beta_n - (2/3) C dl N    Armstrong-Frederick
///
/// The first line uses sigma_eq(s - beta) = q - 3G dl, which is exact because
/// s = s_tr - 2G dl N and N is parallel to eta.
///
/// gamma = 0 is Prager hardening. The return is then radial in the shifted
/// space, dl = (q_tr - sigma_0 - H_n)/(3G + C + H') in closed form, and Newton
/// lands on it in one step from the trial seed. gamma > 0 scales beta_n and
/// the hardening term differently, so eta rotates away from eta_tr during the
/// step whenever beta_n is not parallel to s_tr, and the system iterates.
///
/// Jacobian, with A = ((3/2) IIdev - N (x) N) / q = dN/deta:
///
///   J_ll = -3G - H'          J_lb = -N
///   J_bl = gamma beta - (2/3) C N
///   J_bb = (1 + gamma dl) IIsym + (2/3) C dl A
///
/// The symmetric unknown carries the trace of dbeta as well, so the system is
/// 1 + Dim(Dim+1)/2 wide and its trace row reads (1 + gamma dl) tr(dbeta) =
/// -gamma dl tr(beta_n): well posed, since the diagonal entry is 1 + gamma dl,
/// and a back stress that starts traceless stays traceless.
///
/// and the partial strain derivatives at fixed unknowns, from ds_tr/deps = 2G IIdev:
///
///   dR_l/deps = 2G N         dR_b/deps = -(2/3) C dl 2G A
///
/// The solver is a vector_newton held by material_ref and run only when the
/// trial state yields, as the other return maps do with local_newton
/// (docs/plasticity.md, section 9). It must be created with
///
///   function = <this material>, graph_driven = false, strain_derivative = true,
///   unknowns = [{dlambda, scalar}, {dbeta, sym_tensor}]
///
/// graph_driven = false is checked at the first update, elastic or not; the
/// other three fail to wire at finalize.
///
/// compute_residual() is bound to residual_dlambda so the solver can pull it,
/// which also means the engine fires it once per update like every property
/// callback, at whatever the solver's unknowns hold. It writes kappa_n + dl
/// into the history the hardening source reads, so an elastic step zeroes the
/// unknowns (vector_newton::reset) rather than leave the previous point's dl
/// there to be added to this point's kappa.
///
/// Parameters: strain_source, hardening_source, solver_source, K, G, sigma_0,
///             C (kinematic modulus), gamma (recovery, default 0).
/// Outputs: stress, tangent; history plastic_strain, equivalent_plastic_strain,
///          back_stress. back_stress is stress-like: statev_map exports it with
///          the strain shear factor until #91 lands.
template <typename Traits>
class j2_kinematic_plasticity final
    : public material_base<j2_kinematic_plasticity<Traits>, Traits> {
public:
  using base = material_base<j2_kinematic_plasticity<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;
  using solver_type = vector_newton<Traits>;

  template <typename... Args>
  explicit j2_kinematic_plasticity(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_stress(base::template add_output<tensor2>("stress")),
        m_tangent(base::template add_output<tensor4>("tangent")),
        m_r_l(base::template add_output<value_type>("residual_dlambda")),
        m_r_b(base::template add_output<tensor2>("residual_dbeta")),
        m_j_ll(base::template add_output<value_type>("jacobian_dlambda_dlambda")),
        m_j_lb(base::template add_output<tensor2>("jacobian_dlambda_dbeta")),
        m_j_bl(base::template add_output<tensor2>("jacobian_dbeta_dlambda")),
        m_j_bb(base::template add_output<tensor4>("jacobian_dbeta_dbeta")),
        m_dr_l(base::template add_output<tensor2>("d_residual_dlambda_d_strain")),
        m_dr_b(base::template add_output<tensor4>("d_residual_dbeta_d_strain")),
        m_eps_p(base::template add_history_output<tensor2>("plastic_strain")),
        m_kappa(base::template add_history_output<value_type>(
            "equivalent_plastic_strain")),
        m_beta(base::template add_history_output<tensor2>("back_stress")),
        m_K(base::template get_parameter<value_type>("K")),
        m_G(base::template get_parameter<value_type>("G")),
        m_sigma_0(base::template get_parameter<value_type>("sigma_0")),
        m_C(base::template get_parameter<value_type>("C")),
        m_gamma(base::template get_parameter<value_type>("gamma")),
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
        m_dl(base::template add_input<value_type>(
            base::template get_parameter<std::string>("solver_source"),
            "dlambda", EdgeKind::Local)),
        m_db(base::template add_input<tensor2>(
            base::template get_parameter<std::string>("solver_source"),
            "dbeta", EdgeKind::Local)),
        m_dl_de(base::template add_input<tensor2>(
            base::template get_parameter<std::string>("solver_source"),
            "d_dlambda_d_strain", EdgeKind::Local)),
        m_db_de(base::template add_input<tensor4>(
            base::template get_parameter<std::string>("solver_source"),
            "d_dbeta_d_strain", EdgeKind::Local)),
        m_IIdev(dev_projector<value_type, Dim>()),
        m_IIsym(m_IIdev + tmech::otimes(tmech::eye<value_type, Dim, 2>(),
                                        tmech::eye<value_type, Dim, 2>()) /
                              value_type{Dim}),
        m_C_e(plasticity_detail::make_isotropic_tangent<value_type, Dim>(m_K, m_G))
  {
    base::add_callback(&j2_kinematic_plasticity::compute, m_stress, m_tangent, m_eps_p, m_kappa, m_beta)
        .reads(m_strain, m_H, m_dH, m_dl, m_db, m_dl_de, m_db_de)
        .drives(m_H, m_dH);
    base::add_callback(&j2_kinematic_plasticity::compute_residual, m_r_l, m_r_b, m_j_ll, m_j_lb, m_j_bl,
                       m_j_bb, m_dr_l, m_dr_b)
        .reads(m_dl, m_db, m_H, m_dH)
        .drives(m_H, m_dH);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "hardening_source");
    required<std::string>(para, "strain_source");
    required<std::string>(para, "solver_source");
    required<value_type>(para, "K");
    required<value_type>(para, "G");
    required<value_type>(para, "sigma_0");
    required<value_type>(para, "C");
    optional<value_type>(para, "gamma", value_type{0});
    return para;
  }

  /// Trial state, conditional solve, update. Bound to "stress".
  void compute() {
    auto& solver = m_solver.get();
    if (solver.graph_driven())
      throw std::runtime_error(
          "j2_kinematic_plasticity '" + std::string(base::name()) +
          "': solver_source must be a vector_newton with graph_driven = false. "
          "A graph-driven solver also runs a cold-start solve at every point, "
          "elastic ones included, where the yield residual has a negative root.");

    const auto I = tmech::eye<value_type, Dim, 2>();
    const auto kappa_n = m_kappa.old_value();
    const tensor2 beta_n{m_beta.old_value()};

    // Trial: plastic strain and back stress frozen at the start of the step.
    // s_tr is cached for compute_residual(), which the solver calls back.
    m_s_tr = tensor2{value_type{2} * m_G *
                     tmech::dev(tensor2(m_strain.get() - m_eps_p.old_value()))};
    m_kappa.new_value() = kappa_n;
    m_H.update_source();
    const auto q_tr = equivalent(tensor2{m_s_tr - beta_n});

    if (q_tr - m_sigma_0 - m_H.get() <= value_type{0}) {
      solver.reset();   // dl = 0, dbeta = 0, dx/deps = 0: see the class comment
      m_eps_p.new_value() = m_eps_p.old_value();
      m_beta.new_value() = beta_n;
      m_stress = tensor2{
          m_s_tr +
          m_K * tmech::trace(tensor2(m_strain.get() - m_eps_p.old_value())) * I};
      m_tangent = m_C_e;
      return;
    }

    solver.solve();
    if (!solver.converged())
      throw cutback_error("j2_kinematic_plasticity '" +
                                std::string(base::name()) +
                                "': the coupled return map did not converge");

    const auto dl = m_dl.get();
    // Same reasoning as j2_plasticity: the return map is only entered with
    // F_tr > 0, so a negative multiplier means the hardening is at or below
    // -(3G + C) and the local problem has no admissible solution.
    if (dl < value_type{0})
      throw cutback_error(
          "j2_kinematic_plasticity '" + std::string(base::name()) +
          "': the return map produced a negative plastic multiplier: the "
          "hardening softens faster than 3G + C can absorb (for gamma = 0 that "
          "is H' <= -(3G + C)). That branch is unstable and this local return "
          "map cannot resolve it.");

    const tensor2 beta{beta_n + m_db.get()};
    const tensor2 eta{m_s_tr - tmech::dev(beta)};
    const auto q = equivalent(eta);
    const tensor2 N{value_type{1.5} * eta / q};

    m_eps_p.new_value() = m_eps_p.old_value() + dl * N;
    m_kappa.new_value() = kappa_n + dl;
    m_beta.new_value() = beta;

    // sigma = C_e : (eps - eps_p,new) = K tr(eps - eps_p,new) I + s_tr - 2G dl N
    m_stress = tensor2{
        m_K * tmech::trace(tensor2(m_strain.get() - m_eps_p.new_value())) * I +
        m_s_tr - value_type{2} * m_G * dl * N};

    // Consistent tangent by the chain rule through the solver's dx/deps:
    //   dsigma/deps = C_e - 2G [ N (x) dl/deps + dl dN/deps ]
    //   dN/deps     = A : deta/deps,   deta/deps = 2G IIdev - dbeta/deps
    // A is deviatoric in its second index pair, so A : deta/deps discards the
    // spherical part of dbeta/deps by itself; no dev() is applied to it.
    const tensor4 A{(value_type{1.5} * m_IIdev - tmech::otimes(N, N)) / q};
    const tensor4 deta_de{value_type{2} * m_G * m_IIdev - m_db_de.get()};
    m_tangent = tensor4{m_C_e - value_type{2} * m_G *
                                  (tmech::otimes(N, m_dl_de.get()) +
                                   dl * tmech::dcontract(A, deta_de))};
  }

  /// Residual, Jacobian and strain derivative at the solver's current iterate.
  /// Bound to "residual_dlambda"; the solver pulls it through update_source().
  void compute_residual() {
    const auto dl = m_dl.get();
    const tensor2 beta{m_beta.old_value() + m_db.get()};
    const tensor2 eta{m_s_tr - tmech::dev(beta)};
    const auto q = equivalent(eta);

    m_kappa.new_value() = m_kappa.old_value() + dl;
    m_H.update_source();

    // Inside solve() the seed is the trial state, where q = q_tr > sigma_0 + H,
    // and no iterate reaches q = 0 in practice. The engine's own call of this
    // callback can, at an unloaded point with zero history; the flow direction
    // is undefined there and the outputs are not read, so they are set finite
    // rather than inf. Same threshold as plasticity_utils::evaluate_at_state.
    const bool has_direction = q > value_type{1e-10} * std::abs(m_sigma_0);
    const tensor2 N{has_direction ? tensor2{value_type{1.5} * eta / q} : tensor2{}};
    const tensor4 A{has_direction
                        ? tensor4{(value_type{1.5} * m_IIdev - tmech::otimes(N, N)) / q}
                        : tensor4{}};

    const auto f = value_type{1} + m_gamma * dl;
    const auto c = value_type{2} / value_type{3} * m_C;

    m_r_l = q - value_type{3} * m_G * dl - m_sigma_0 - m_H.get();
    m_r_b = tensor2{f * beta - m_beta.old_value() - (c * dl) * N};

    m_j_ll = -value_type{3} * m_G - m_dH.get();
    m_j_lb = tensor2{-N};
    m_j_bl = tensor2{m_gamma * beta - c * N};
    m_j_bb = tensor4{f * m_IIsym + (c * dl) * A};

    m_dr_l = tensor2{value_type{2} * m_G * N};
    m_dr_b = tensor4{-(c * dl * value_type{2} * m_G) * A};
  }

private:
  static value_type equivalent(const tensor2& x) {
    return std::sqrt(value_type{1.5} * tmech::dcontract(x, x));
  }

  tensor2& m_stress;
  tensor4& m_tangent;
  value_type& m_r_l;
  tensor2& m_r_b;
  value_type& m_j_ll;
  tensor2& m_j_lb;
  tensor2& m_j_bl;
  tensor4& m_j_bb;
  tensor2& m_dr_l;
  tensor4& m_dr_b;
  history_property<tensor2>& m_eps_p;
  history_property<value_type>& m_kappa;
  history_property<tensor2>& m_beta;

  const value_type& m_K;
  const value_type& m_G;
  const value_type& m_sigma_0;
  const value_type& m_C;
  const value_type& m_gamma;
  material_ref<solver_type, Traits>& m_solver;

  const input_property<tensor2, property_traits>& m_strain;
  const input_property<value_type, property_traits>& m_H;
  const input_property<value_type, property_traits>& m_dH;
  const input_property<value_type, property_traits>& m_dl;
  const input_property<tensor2, property_traits>& m_db;
  const input_property<tensor2, property_traits>& m_dl_de;
  const input_property<tensor4, property_traits>& m_db_de;

  const tensor4& m_IIdev;
  const tensor4 m_IIsym;
  const tensor4 m_C_e;
  tensor2 m_s_tr{};
};

}  // namespace numsim::materials

#endif  // J2_KINEMATIC_PLASTICITY_H
