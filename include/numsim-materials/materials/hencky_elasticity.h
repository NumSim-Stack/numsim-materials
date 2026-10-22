#ifndef NUMSIM_MATERIALS_HENCKY_ELASTICITY_H
#define NUMSIM_MATERIALS_HENCKY_ELASTICITY_H

#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

/// Linear elasticity in the logarithmic strain eps = ln V = ln(F F^T) / 2 with a produced stiffness C
/// ("tangent_source", e.g. isotropic_tangent): tau = C : (eps - eps_eig), P = tau F^-T and the consistent
/// dP/dF. Outputs "stress" (P), "tangent" (dP/dF), "log_strain".
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
        m_P(base::template add_output<tensor2>("stress")),
        m_A(base::template add_output<tensor4>("tangent")),
        m_log_strain(base::template add_output<tensor2>("log_strain")),
        m_C(base::template add_input<tensor4>(base::template get_parameter<std::string>("tangent_source"),
                                              base::template get_parameter<std::string>("tangent_property"), EdgeKind::Global)),
        m_F_name(base::template get_parameter<std::string>("deformation_gradient_source")),
        m_F(base::template add_input<tensor2>(m_F_name, "deformation_gradient", EdgeKind::Global)),
        m_F_inv(base::template add_input<tensor2>(m_F_name, "inverse", EdgeKind::Global)),
        m_eigenstrain(eigenstrain_input()) {
    base::add_callback(&hencky_elasticity::compute, m_P, m_A, m_log_strain).reads(m_C, m_F, m_F_inv, m_eigenstrain);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "tangent_source");
    optional<std::string>(para, "tangent_property", std::string{"tangent"});
    required<std::string>(para, "deformation_gradient_source");
    optional<std::string>(para, "eigenstrain_source", std::string{});
    return para;
  }

  void compute() noexcept {
    auto const &F{m_F.get()};
    auto const &Finv{m_F_inv.get()};
    auto const &C{m_C.get()};
    auto const I{tmech::eye<value_type, Dim, 2>()};
    tensor2 const b{F * tmech::trans(F)};
    auto ln_b{tmech::log(b)};
    m_log_strain = value_type{0.5} * ln_b;
    tensor2 const eps{m_eigenstrain ? tensor2{m_log_strain - m_eigenstrain->get()} : m_log_strain};
    tensor2 const tau{tmech::dcontract(C, eps)};
    m_P = tau * tmech::trans(Finv);
    // d ln V / dF = 1/2 (d ln b / db) : (db / dF), db_ij / dF_kl = d_ik F_jl + F_il d_jk
    tensor4 const dlog_dF{value_type{0.5} * tmech::dcontract(tensor4{ln_b.derivative()}, tensor4{tmech::otimesu(I, F) + tmech::otimesl(F, I)})};
    tensor4 const dtau{tmech::dcontract(C, dlog_dF)};
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
  input_property<tensor4, property_traits> const &m_C;
  std::string const &m_F_name;
  input_property<tensor2, property_traits> const &m_F;
  input_property<tensor2, property_traits> const &m_F_inv;
  input_property<tensor2, property_traits> const *m_eigenstrain; ///< null without an eigenstrain_source
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_HENCKY_ELASTICITY_H
