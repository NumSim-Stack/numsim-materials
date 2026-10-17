#ifndef NUMSIM_MATERIALS_HENCKY_ELASTICITY_H
#define NUMSIM_MATERIALS_HENCKY_ELASTICITY_H

#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/log_strain.h"
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Linear elasticity in the logarithmic strain: eps = ln V, tau = C : (eps - eps_eig), P = tau F^-T,
/// with the consistent dP/dF through d ln V / dF (core/log_strain.h). Outputs "stress" (P),
/// "tangent" (dP/dF), "log_strain".
///
/// Parameters: "K", "G", "deformation_gradient_source", optional "eigenstrain_source" (an
/// eigenstrain to subtract from the log strain: thermal, chemical).
template <typename Traits>
class hencky_elasticity final : public material_base<hencky_elasticity<Traits>, Traits> {
public:
  using base = material_base<hencky_elasticity<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit hencky_elasticity(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_P(base::template add_output<tensor2>("stress", &hencky_elasticity::update)),
        m_A(base::template add_output<tensor4>("tangent")),
        m_log_strain(base::template add_output<tensor2>("log_strain")),
        m_C(isotropic_stiffness<value_type, Dim>(base::template get_parameter<value_type>("K"),
                                                 base::template get_parameter<value_type>("G"))),
        m_F(base::template add_input<tensor2>(base::template get_parameter<std::string>("deformation_gradient_source"),
                                              "deformation_gradient", EdgeKind::Global)),
        m_F_inv(base::template add_input<tensor2>(base::template get_parameter<std::string>("deformation_gradient_source"),
                                                  "inverse", EdgeKind::Global)),
        m_eigenstrain(eigenstrain_input()) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<value_type>("K").template add<is_required>();
    para.template insert<value_type>("G").template add<is_required>();
    para.template insert<std::string>("deformation_gradient_source").template add<is_required>();
    para.template insert<std::string>("eigenstrain_source").template add<set_default>(std::string{});
    return para;
  }

  void update() {
    auto const &Finv{m_F_inv.get()};
    auto const [log_V, dlog_dF]{log_strain(m_F.get())};
    m_log_strain = log_V;
    tensor2 const eps{m_eigenstrain ? tensor2{m_log_strain - m_eigenstrain->get()} : m_log_strain};
    tensor2 const tau{tmech::dcontract(m_C, eps)};
    m_P = tau * tmech::trans(Finv);
    // dP/dF = (dtau/dF) . F^-T - P_il F^-1_jk, dtau/dF = C : d ln V / dF
    tensor4 const dtau{tmech::dcontract(m_C, dlog_dF)};
    auto const I{tmech::eye<value_type, Dim, 2>()};
    m_A = tmech::dcontract(tmech::otimesu(I, Finv), dtau) - tmech::otimesl(m_P, Finv);
  }

private:
  input_property<tensor2, property_traits> const *eigenstrain_input() {
    auto const &source{base::template get_parameter<std::string>("eigenstrain_source")};
    return source.empty() ? nullptr : &base::template add_input<tensor2>(source, "eigenstrain", EdgeKind::Global);
  }

  tensor2 &m_P;
  tensor4 &m_A;
  tensor2 &m_log_strain;
  tensor4 m_C;
  input_property<tensor2, property_traits> const &m_F;
  input_property<tensor2, property_traits> const &m_F_inv;
  input_property<tensor2, property_traits> const *m_eigenstrain; ///< null without an eigenstrain_source
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_HENCKY_ELASTICITY_H
