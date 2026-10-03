#ifndef NEO_HOOKE_H
#define NEO_HOOKE_H

#include <cmath>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Compressible neo-Hookean hyperelasticity with the strain energy
///   W = G / 2 (tr(F^T F) - 3) - G ln J + lambda / 2 (ln J)^2,
/// lambda = K - 2 G / 3, J = det F, which reduces to isotropic linear
/// elasticity (K, G) at small strain.
///
/// Consumes the "deformation_gradient" history of the material named by
/// "deformation_gradient_source" and publishes the first Piola-Kirchhoff
/// stress P = G (F - F^-T) + lambda ln J F^-T as "stress" and
///   dP_ij / dF_kl = G d_ik d_jl + (G - lambda ln J) F^-1_jk F^-1_li
///                 + lambda F^-1_ji F^-1_lk
/// as "tangent".
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
        m_P(base::template add_output<tensor2>("stress", &neo_hooke::update)),
        m_A(base::template add_output<tensor4>("tangent")),
        m_K(base::template get_parameter<value_type>("K")),
        m_G(base::template get_parameter<value_type>("G")),
        m_F_name(base::template get_parameter<std::string>("deformation_gradient_source")),
        m_F(base::template add_input<tensor2>(m_F_name, "deformation_gradient", EdgeKind::Global)) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<value_type>("K").template add<is_required>();
    para.template insert<value_type>("G").template add<is_required>();
    para.template insert<std::string>("deformation_gradient_source").template add<is_required>();
    return para;
  }

  void update() {
    const tensor2 F{m_F.get()};
    const value_type J{tmech::det(F)};
    if (!(J > value_type(0)))
      throw std::domain_error("neo_hooke: det F must be positive");
    const value_type lambda{m_K - value_type(2) / value_type(3) * m_G};
    const value_type lnJ{std::log(J)};
    const tensor2 Finv{tmech::inv(F)};
    const tensor2 FinvT{tmech::trans(Finv)};
    m_P = m_G * (F - FinvT) + lambda * lnJ * FinvT;
    for (std::size_t i = 0; i < Dim; ++i)
      for (std::size_t j = 0; j < Dim; ++j)
        for (std::size_t k = 0; k < Dim; ++k)
          for (std::size_t l = 0; l < Dim; ++l)
            m_A(i, j, k, l) = (i == k && j == l ? m_G : value_type(0)) +
                              (m_G - lambda * lnJ) * Finv(j, k) * Finv(l, i) +
                              lambda * Finv(j, i) * Finv(l, k);
  }

private:
  tensor2& m_P;
  tensor4& m_A;
  const value_type& m_K;
  const value_type& m_G;
  const std::string& m_F_name;
  const input_property<tensor2, property_traits>& m_F;
};

}  // namespace numsim::materials
#endif  // NEO_HOOKE_H
