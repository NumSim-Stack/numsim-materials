#ifndef SAINT_VENANT_KIRCHHOFF_H
#define SAINT_VENANT_KIRCHHOFF_H

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Saint Venant-Kirchhoff hyperelasticity: the linear isotropic law
/// S = C : E in the Green-Lagrange strain E = (F^T F - I) / 2, with
/// C = K I (x) I + 2 G (I_s - I (x) I / 3).
///
/// Consumes the "deformation_gradient" history of the material named by
/// "deformation_gradient_source" (external_deformation_gradient_source) and
/// publishes the first Piola-Kirchhoff stress P = F S as "stress" and its
/// derivative dP_ij / dF_kl = d_ik S_lj + F_im C_mjlb F_kb as "tangent" (a
/// rank-4 tensor without minor symmetries).
template <typename Traits>
class saint_venant_kirchhoff
    : public material_base<saint_venant_kirchhoff<Traits>, Traits> {
public:
  using base = material_base<saint_venant_kirchhoff<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit saint_venant_kirchhoff(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_P(base::template add_output<tensor2>("stress", &saint_venant_kirchhoff::update)),
        m_A(base::template add_output<tensor4>("tangent")),
        m_K(base::template get_parameter<value_type>("K")),
        m_G(base::template get_parameter<value_type>("G")),
        m_F_name(base::template get_parameter<std::string>("deformation_gradient_source")),
        m_F(base::template add_input<tensor2>(m_F_name, "deformation_gradient", EdgeKind::Global)) {
    const auto I{tmech::eye<value_type, Dim, 2>()};
    const auto IIsym{(tmech::otimesu(I, I) + tmech::otimesl(I, I)) * 0.5};
    const auto IIvol{tmech::otimes(I, I) / Dim};
    m_C = 3 * m_K * IIvol + 2 * m_G * (IIsym - IIvol);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<value_type>("K").template add<is_required>();
    para.template insert<value_type>("G").template add<is_required>();
    para.template insert<std::string>("deformation_gradient_source").template add<is_required>();
    return para;
  }

  void update() {
    const auto& F{m_F.get()};
    const auto I{tmech::eye<value_type, Dim, 2>()};
    const tensor2 E{0.5 * (tmech::trans(F) * F - I)};
    const tensor2 S{tmech::dcontract(m_C, E)};
    m_P = F * S;
    // d_ik S_lj + F_im C_mjlb F_kb: contract F with the first index of C,
    // then the last index with F (giving i j l k), and swap the last two
    const tensor4 FC{tmech::inner_product<tmech::sequence<2>, tmech::sequence<1>>(F, m_C)};
    m_A = tmech::otimesu(I, tmech::trans(S)) +
          tmech::basis_change<tmech::sequence<1, 2, 4, 3>>(
              tmech::inner_product<tmech::sequence<4>, tmech::sequence<2>>(FC, F));
  }

private:
  tensor2& m_P;
  tensor4& m_A;
  const value_type& m_K;
  const value_type& m_G;
  const std::string& m_F_name;
  const input_property<tensor2, property_traits>& m_F;
  tensor4 m_C;
};

}  // namespace numsim::materials
#endif  // SAINT_VENANT_KIRCHHOFF_H
