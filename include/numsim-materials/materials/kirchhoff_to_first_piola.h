#ifndef NUMSIM_MATERIALS_KIRCHHOFF_TO_FIRST_PIOLA_H
#define NUMSIM_MATERIALS_KIRCHHOFF_TO_FIRST_PIOLA_H

#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

/// Push-forward of a Kirchhoff stress tau of the logarithmic strain to the first Piola-Kirchhoff stress
/// P = tau F^-T and its consistent dP/dF. Reads tau ("stress_source"), d tau / d eps_e ("tangent_source"),
/// d ln V / dF ("strain_source": a log_strain) and F^-1 ("deformation_gradient_source"). Outputs "stress", "tangent".
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
        m_F_inv(base::template add_input<tensor2>(base::template get_parameter<std::string>("deformation_gradient_source"),
                                                  "inverse", EdgeKind::Global)) {
    base::add_callback(&kirchhoff_to_first_piola::compute, m_P, m_A).reads(m_tau, m_C, m_dlog, m_F_inv);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "stress_source");
    optional<std::string>(para, "stress_property", std::string{"stress"});
    required<std::string>(para, "tangent_source");
    optional<std::string>(para, "tangent_property", std::string{"tangent"});
    required<std::string>(para, "strain_source");
    required<std::string>(para, "deformation_gradient_source");
    return para;
  }

  void compute() noexcept {
    auto const &Finv{m_F_inv.get()};
    auto const I{tmech::eye<value_type, Dim, 2>()};
    m_P = m_tau.get() * tmech::trans(Finv);
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
