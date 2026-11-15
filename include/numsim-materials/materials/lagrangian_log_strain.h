#ifndef NUMSIM_MATERIALS_LAGRANGIAN_LOG_STRAIN_H
#define NUMSIM_MATERIALS_LAGRANGIAN_LOG_STRAIN_H

#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

/// Lagrangian logarithmic strain E = ln(F^T F) / 2 of the deformation gradient ("deformation_gradient_source").
/// Outputs "strain" and "d_strain_d_deformation_gradient" (dE/dF). For anisotropic stress laws (docs/finite-strain.md).
template <typename Traits>
class lagrangian_log_strain final : public material_base<lagrangian_log_strain<Traits>, Traits> {
public:
  using base = material_base<lagrangian_log_strain<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit lagrangian_log_strain(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_strain(base::template add_output<tensor2>("strain")),
        m_dstrain(base::template add_output<tensor4>("d_strain_d_deformation_gradient")),
        m_F(base::template add_input<tensor2>(base::template get_parameter<std::string>("deformation_gradient_source"),
                                              "deformation_gradient", EdgeKind::Global)) {
    // no partials(): F is not symmetric and the declared derivatives cover symmetric tensors only
    base::add_callback(&lagrangian_log_strain::compute, m_strain, m_dstrain).reads(m_F);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "deformation_gradient_source");
    return para;
  }

  void compute() noexcept {
    auto const &F{m_F.get()};
    auto const I{tmech::eye<value_type, Dim, 2>()};
    tensor2 const Ft{tmech::trans(F)};
    auto ln_C{tmech::log(tensor2{Ft * F})};
    m_strain = value_type{0.5} * ln_C;
    // dE/dF = 1/2 (d ln C / dC) : (dC / dF), dC_IJ / dF_kL = d_IL F_kJ + F_kI d_JL
    m_dstrain = value_type{0.5} * tmech::dcontract(tensor4{ln_C.derivative()}, tensor4{tmech::otimesl(I, Ft) + tmech::otimesu(Ft, I)});
  }

private:
  tensor2 &m_strain;
  tensor4 &m_dstrain;
  input_property<tensor2, property_traits> const &m_F;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_LAGRANGIAN_LOG_STRAIN_H
