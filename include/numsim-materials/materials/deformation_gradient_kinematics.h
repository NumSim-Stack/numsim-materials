#ifndef NUMSIM_MATERIALS_DEFORMATION_GRADIENT_KINEMATICS_H
#define NUMSIM_MATERIALS_DEFORMATION_GRADIENT_KINEMATICS_H

#include <limits>
#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

/// What the consumers of a deformation gradient share, computed once: "inverse" F^-1 and "determinant" det F
/// of "deformation_gradient_source". A singular F gives a NaN inverse; the consumers check det F.
template <typename Traits>
class deformation_gradient_kinematics final : public material_base<deformation_gradient_kinematics<Traits>, Traits> {
public:
  using base = material_base<deformation_gradient_kinematics<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;

  template <typename... Args>
  explicit deformation_gradient_kinematics(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_F_inv(base::template add_output<tensor2>("inverse")),
        m_J(base::template add_output<value_type>("determinant")),
        m_F(base::template add_input<tensor2>(base::template get_parameter<std::string>("deformation_gradient_source"),
                                              "deformation_gradient", EdgeKind::Global)) {
    base::add_callback(&deformation_gradient_kinematics::compute, m_F_inv, m_J).reads(m_F);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "deformation_gradient_source");
    return para;
  }

  void compute() {
    auto const &F{m_F.get()};
    m_J = tmech::det(F);
    if (m_J != value_type{0})
      m_F_inv = tmech::inv(F);
    else
      m_F_inv = std::numeric_limits<value_type>::quiet_NaN() * tmech::eye<value_type, Dim, 2>();
  }

private:
  tensor2 &m_F_inv;
  value_type &m_J;
  input_property<tensor2, property_traits> const &m_F;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DEFORMATION_GRADIENT_KINEMATICS_H
