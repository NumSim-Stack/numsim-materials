#ifndef SAINT_VENANT_KIRCHHOFF_H
#define SAINT_VENANT_KIRCHHOFF_H

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/tensor_identities.h"

namespace numsim::materials {

/// Saint Venant-Kirchhoff: S = C : E, E = (F^T F - I) / 2.
/// "stress": P = F S, "tangent": dP/dF.
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
        m_P(base::template add_output<tensor2>("stress")),
        m_A(base::template add_output<tensor4>("tangent")),
        m_K(base::template get_parameter<value_type>("K")),
        m_G(base::template get_parameter<value_type>("G")),
        m_F_name(base::template get_parameter<std::string>("deformation_gradient_source")),
        m_F(base::template add_input<tensor2>(m_F_name, "deformation_gradient", EdgeKind::Global)) {
    m_C = 3 * m_K * vol_projector<value_type, Dim>() + 2 * m_G * dev_projector<value_type, Dim>();
    base::add_callback(&saint_venant_kirchhoff::update, m_P, m_A).reads(m_F);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<value_type>(para, "K");
    required<value_type>(para, "G");
    required<std::string>(para, "deformation_gradient_source");
    return para;
  }

  void update() {
    const auto& F{m_F.get()};
    const auto I{tmech::eye<value_type, Dim, 2>()};
    const tensor2 E{0.5 * (tmech::trans(F) * F - I)};
    const tensor2 S{tmech::dcontract(m_C, E)};
    m_P = F * S;
    // d_ik S_lj + F_im C_mjlb F_kb
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
