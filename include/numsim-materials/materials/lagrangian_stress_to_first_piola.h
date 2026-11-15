#ifndef NUMSIM_MATERIALS_LAGRANGIAN_STRESS_TO_FIRST_PIOLA_H
#define NUMSIM_MATERIALS_LAGRANGIAN_STRESS_TO_FIRST_PIOLA_H

#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

/// Pull-back of the stress T work-conjugate to the Lagrangian log strain E = ln(C) / 2 to the first Piola-Kirchhoff
/// stress P = T : dE/dF and its consistent dP/dF. Reads T ("stress_source"), dT/dE ("tangent_source"), dE/dF
/// ("strain_source": a lagrangian_log_strain) and F ("deformation_gradient_source"). Outputs "stress", "tangent".
/// Objective for any (also anisotropic) stress law.
template <typename Traits>
class lagrangian_stress_to_first_piola final : public material_base<lagrangian_stress_to_first_piola<Traits>, Traits> {
public:
  using base = material_base<lagrangian_stress_to_first_piola<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit lagrangian_stress_to_first_piola(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_P(base::template add_output<tensor2>("stress")),
        m_A(base::template add_output<tensor4>("tangent")),
        m_T(base::template add_input<tensor2>(base::template get_parameter<std::string>("stress_source"),
                                              base::template get_parameter<std::string>("stress_property"), EdgeKind::Global)),
        m_C(base::template add_input<tensor4>(base::template get_parameter<std::string>("tangent_source"),
                                              base::template get_parameter<std::string>("tangent_property"), EdgeKind::Global)),
        m_dE(base::template add_input<tensor4>(base::template get_parameter<std::string>("strain_source"),
                                               "d_strain_d_deformation_gradient", EdgeKind::Global)),
        m_F(base::template add_input<tensor2>(base::template get_parameter<std::string>("deformation_gradient_source"),
                                              "deformation_gradient", EdgeKind::Global)) {
    base::add_callback(&lagrangian_stress_to_first_piola::compute, m_P, m_A).reads(m_T, m_C, m_dE, m_F);
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
    auto const &F{m_F.get()};
    auto const &T{m_T.get()};
    auto const &dE{m_dE.get()};
    auto const I{tmech::eye<value_type, Dim, 2>()};
    tensor2 const Ft{tmech::trans(F)};
    auto ln_C{tmech::log(tensor2{Ft * F})};
    tensor4 const dC{tmech::otimesl(I, Ft) + tmech::otimesu(Ft, I)};
    m_P = tmech::dcontract(T, dE); // = F S
    // dP/dF = dE^T : dT/dE : dE + dC^T : (T : d2E/dC2) : dC + I (x) S, with S = T : d ln C / dC
    tensor2 const S{tmech::dcontract(T, tensor4{ln_C.derivative()})};
    tensor4 const M{value_type{0.5} * tmech::inner_product<tmech::sequence<1, 2>, tmech::sequence<1, 2>>(
                                          T, tmech::tensor<value_type, Dim, 6>{ln_C.second_derivative()})};
    m_A = tmech::dcontract(major(dE), tensor4{tmech::dcontract(m_C.get(), dE)}) +
          tmech::dcontract(major(dC), tensor4{tmech::dcontract(M, dC)}) + tmech::otimesu(I, S);
  }

private:
  static tensor4 major(tensor4 const &A) { return tmech::basis_change<tmech::sequence<3, 4, 1, 2>>(A); }

  tensor2 &m_P;
  tensor4 &m_A;
  input_property<tensor2, property_traits> const &m_T;
  input_property<tensor4, property_traits> const &m_C;
  input_property<tensor4, property_traits> const &m_dE;
  input_property<tensor2, property_traits> const &m_F;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_LAGRANGIAN_STRESS_TO_FIRST_PIOLA_H
