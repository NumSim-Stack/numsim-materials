#ifndef NUMSIM_MATERIALS_ESHELBY_CYLINDER_H
#define NUMSIM_MATERIALS_ESHELBY_CYLINDER_H

#include <vector>

#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/materials/detail/eshelby_tensor_base.h"

namespace numsim::materials {

/// Eshelby tensor of a circular cylinder along "direction" in the isotropic "matrix"; output "tensor".
template <typename Traits>
class eshelby_cylinder final : public detail::eshelby_tensor_base<eshelby_cylinder<Traits>, Traits> {
public:
  using base = detail::eshelby_tensor_base<eshelby_cylinder<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using tensor4 = typename base::tensor4;
  using tensor2 = tmech::tensor<value_type, 3, 2>;

  template <typename... Args>
  explicit eshelby_cylinder(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_rotation(rotation_to(detail::unit_direction(base::template get_parameter<std::vector<value_type>>("direction"), base::name()))) {
    base::register_output();
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    optional<std::vector<value_type>>(para, "direction", std::vector<value_type>{1, 0, 0});
    return para;
  }

  [[nodiscard]] tensor4 shape(value_type nu) const { return rotate(eshelby::cylinder(nu, std::size_t{0}), m_rotation); }

private:
  tensor2 m_rotation; ///< x1 onto the direction
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_ESHELBY_CYLINDER_H
