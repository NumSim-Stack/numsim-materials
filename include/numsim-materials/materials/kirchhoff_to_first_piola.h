#ifndef NUMSIM_MATERIALS_KIRCHHOFF_TO_FIRST_PIOLA_H
#define NUMSIM_MATERIALS_KIRCHHOFF_TO_FIRST_PIOLA_H

#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

/// Push-forward of a Kirchhoff stress tau of the logarithmic strain to the first Piola-Kirchhoff stress
/// P = tau F^-T and its consistent dP/dF. Reads tau ("stress_source"), d tau / d eps_e ("tangent_source"),
/// d ln V / dF ("strain_source": a log_strain) and F^-1 ("kinematics_source": a deformation_gradient_kinematics).
/// Outputs "stress" with its partials by tau and F^-1, and "tangent"; both unsymmetric.
/// Valid for an isotropic stress law only; this is not checked (docs/finite-strain.md).
template <typename Traits>
class kirchhoff_to_first_piola final : public material_base<kirchhoff_to_first_piola<Traits>, Traits> {
public:
  using base = material_base<kirchhoff_to_first_piola<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit kirchhoff_to_first_piola(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_P(base::template add_output<tensor2>("stress")),
        m_A(base::template add_output<tensor4>("tangent")),
        m_tau(base::template add_input<tensor2>(base::template get_parameter<std::string>("stress_source"),
                                                base::template get_parameter<std::string>("stress_property"), EdgeKind::Global)),
        m_C(base::template add_input<tensor4>(base::template get_parameter<std::string>("tangent_source"),
                                              base::template get_parameter<std::string>("tangent_property"), EdgeKind::Global)),
        m_dlog(base::template add_input<tensor4>(base::template get_parameter<std::string>("strain_source"),
                                                 "d_strain_d_deformation_gradient", EdgeKind::Global)),
        m_F_inv(base::template add_input<tensor2>(base::template get_parameter<std::string>("kinematics_source"),
                                                  "inverse", EdgeKind::Global)) {
    base::add_callback(&kirchhoff_to_first_piola::compute_stress, m_P).reads(m_tau, m_F_inv).partials(m_P, m_tau, m_F_inv);
    base::add_callback(&kirchhoff_to_first_piola::compute_tangent, m_A).reads(m_C, m_dlog, m_F_inv, m_P);
    base::unsymmetric_output(m_P);
    base::unsymmetric_output(m_A);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "stress_source");
    optional<std::string>(para, "stress_property", std::string{"stress"});
    required<std::string>(para, "tangent_source");
    optional<std::string>(para, "tangent_property", std::string{"tangent"});
    required<std::string>(para, "strain_source");
    required<std::string>(para, "kinematics_source");
    return para;
  }

  void compute_stress() noexcept {
    auto const &Finv{m_F_inv.get()};
    m_P = m_tau.get() * tmech::trans(Finv);
    if (base::want_partials()) {
      auto const I{tmech::eye<value_type, Dim, 2>()};
      // P_ij = tau_ik F^-1_jk; tau is symmetric, so is its partial
      base::partial(m_P, m_tau) = value_type{0.5} * (tmech::otimesu(I, Finv) + tmech::otimesl(I, Finv));
      base::partial(m_P, m_F_inv) = tmech::otimesl(m_tau.get(), I);
    }
  }

  void compute_tangent() noexcept {
    auto const &Finv{m_F_inv.get()};
    auto const I{tmech::eye<value_type, Dim, 2>()};
    // d eps_e / d ln V is the identity: a strain-independent eigenstrain does not enter the tangent
    tensor4 const dtau{tmech::dcontract(m_C.get(), m_dlog.get())};
    m_A = tmech::dcontract(tmech::otimesu(I, Finv), dtau) - tmech::otimesl(m_P, Finv);
  }

private:
  tensor2 &m_P;
  tensor4 &m_A;
  input_property<tensor2, property_traits> const &m_tau;
  input_property<tensor4, property_traits> const &m_C;
  input_property<tensor4, property_traits> const &m_dlog;
  input_property<tensor2, property_traits> const &m_F_inv;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_KIRCHHOFF_TO_FIRST_PIOLA_H
