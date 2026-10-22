#ifndef NUMSIM_MATERIALS_DEPOLARISATION_CYLINDER_H
#define NUMSIM_MATERIALS_DEPOLARISATION_CYLINDER_H

#include <vector>

#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/materials/detail/depolarisation_tensor_base.h"
#include "numsim-materials/materials/detail/direction.h"

namespace numsim::materials {

/// Depolarisation tensor of a circular cylinder along "direction", (I - d (x) d) / 2; output "tensor".
template <typename Traits>
class depolarisation_cylinder final : public detail::depolarisation_tensor_base<depolarisation_cylinder<Traits>, Traits> {
public:
  using base = detail::depolarisation_tensor_base<depolarisation_cylinder<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using tensor2 = typename base::tensor2;

  template <typename... Args>
  explicit depolarisation_cylinder(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_direction(detail::unit_direction(base::template get_parameter<std::vector<value_type>>("direction"), base::name())) {
    base::register_output();
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    optional<std::vector<value_type>>(para, "direction", std::vector<value_type>{1, 0, 0});
    return para;
  }

  [[nodiscard]] tensor2 shape() const noexcept { return eshelby::depolarisation_cylinder(m_direction); }

private:
  tmech::tensor<value_type, 3, 1> m_direction;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DEPOLARISATION_CYLINDER_H
