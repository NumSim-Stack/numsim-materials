#ifndef NEO_HOOKE_H
#define NEO_HOOKE_H

#include <cmath>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/errors.h"

namespace numsim::materials {

/// Compressible neo-Hooke: W = G/2 (tr(F^T F) - 3) - G ln J + lambda/2 (ln J)^2.
/// "stress": P with its partials by F, F^-1 and det F, "tangent": dP/dF; both unsymmetric. F from
/// "deformation_gradient_source", F^-1 and det F from "kinematics_source" (a deformation_gradient_kinematics).
template <typename Traits>
class neo_hooke : public material_base<neo_hooke<Traits>, Traits> {
public:
  using base = material_base<neo_hooke<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit neo_hooke(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_P(base::template add_output<tensor2>("stress")),
        m_A(base::template add_output<tensor4>("tangent")),
        m_K(base::template get_parameter<value_type>("K")),
        m_G(base::template get_parameter<value_type>("G")),
        m_F(base::template add_input<tensor2>(base::template get_parameter<std::string>("deformation_gradient_source"),
                                              "deformation_gradient", EdgeKind::Global)),
        m_kinematics(base::template get_parameter<std::string>("kinematics_source")),
        m_F_inv(base::template add_input<tensor2>(m_kinematics, "inverse", EdgeKind::Global)),
        m_J(base::template add_input<value_type>(m_kinematics, "determinant", EdgeKind::Global)) {
    base::add_callback(&neo_hooke::update, m_P, m_A).reads(m_F, m_F_inv, m_J).partials(m_P, m_F, m_F_inv, m_J);
    base::unsymmetric_output(m_P);
    base::unsymmetric_output(m_A);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<value_type>("K").template add<is_required>();
    para.template insert<value_type>("G").template add<is_required>();
    para.template insert<std::string>("deformation_gradient_source").template add<is_required>();
    para.template insert<std::string>("kinematics_source").template add<is_required>();
    return para;
  }

  void update() {
    const auto& F{m_F.get()};
    const auto& Finv{m_F_inv.get()};
    const value_type J{m_J.get()};
    if (!(J > value_type(0)))
      throw cutback_error("neo_hooke: det F must be positive");
    const value_type lambda{m_K - value_type(2) / value_type(3) * m_G};
    const value_type lnJ{std::log(J)};
    const auto FinvT{tmech::trans(Finv)};
    const auto I{tmech::eye<value_type, Dim, 2>()};
    m_P = m_G * (F - FinvT) + lambda * lnJ * FinvT;
    m_A = m_G * tmech::otimesu(I, I) + (m_G - lambda * lnJ) * tmech::otimesl(FinvT, Finv) +
          lambda * tmech::otimes(FinvT, FinvT);
    if (base::want_partials()) {
      base::partial(m_P, m_F) = m_G * tmech::otimesu(I, I);
      base::partial(m_P, m_F_inv) = (lambda * lnJ - m_G) * tmech::otimesl(I, I); // P reads F^-T
      base::partial(m_P, m_J) = lambda / J * FinvT;
    }
  }

private:
  tensor2& m_P;
  tensor4& m_A;
  const value_type& m_K;
  const value_type& m_G;
  const input_property<tensor2, property_traits>& m_F;
  const std::string& m_kinematics;
  const input_property<tensor2, property_traits>& m_F_inv;
  const input_property<value_type, property_traits>& m_J;
};

}  // namespace numsim::materials
#endif  // NEO_HOOKE_H
