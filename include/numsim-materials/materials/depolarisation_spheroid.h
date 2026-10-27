#ifndef NUMSIM_MATERIALS_DEPOLARISATION_SPHEROID_H
#define NUMSIM_MATERIALS_DEPOLARISATION_SPHEROID_H

#include <stdexcept>
#include <vector>

#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/materials/detail/depolarisation_tensor_base.h"
#include "numsim-materials/materials/detail/direction.h"

namespace numsim::materials {

/// Depolarisation tensor of a spheroid with "aspect_ratio" (length over diameter) along "direction";
/// output "tensor".
template <typename Traits>
class depolarisation_spheroid final : public detail::depolarisation_tensor_base<depolarisation_spheroid<Traits>, Traits> {
public:
  using base = detail::depolarisation_tensor_base<depolarisation_spheroid<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using tensor2 = typename base::tensor2;

  template <typename... Args>
  explicit depolarisation_spheroid(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_aspect_ratio(base::template get_parameter<value_type>("aspect_ratio")),
        m_rotation(rotation_to(detail::unit_direction(base::template get_parameter<std::vector<value_type>>("direction"), base::name()))) {
    if (!(m_aspect_ratio > 0))
      throw std::invalid_argument(base::name() + ": aspect_ratio must be positive");
    base::register_output();
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<value_type>(para, "aspect_ratio");
    optional<std::vector<value_type>>(para, "direction", std::vector<value_type>{1, 0, 0});
    return para;
  }

  /// The symmetry axis along x1 in the principal frame, then onto the direction.
  [[nodiscard]] tensor2 shape() const {
    return rotate(eshelby::depolarisation(m_aspect_ratio, value_type{1}, value_type{1}), m_rotation);
  }

private:
  value_type const &m_aspect_ratio;
  tensor2 m_rotation;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DEPOLARISATION_SPHEROID_H
