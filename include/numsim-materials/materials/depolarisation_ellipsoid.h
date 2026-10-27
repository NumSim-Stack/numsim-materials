#ifndef NUMSIM_MATERIALS_DEPOLARISATION_ELLIPSOID_H
#define NUMSIM_MATERIALS_DEPOLARISATION_ELLIPSOID_H

#include <stdexcept>
#include <vector>

#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/materials/detail/depolarisation_tensor_base.h"

namespace numsim::materials {

/// Depolarisation tensor of an ellipsoid with "semi_axes" along the global axes; output "tensor".
template <typename Traits>
class depolarisation_ellipsoid final : public detail::depolarisation_tensor_base<depolarisation_ellipsoid<Traits>, Traits> {
public:
  using base = detail::depolarisation_tensor_base<depolarisation_ellipsoid<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using tensor2 = typename base::tensor2;

  template <typename... Args>
  explicit depolarisation_ellipsoid(Args &&...args)
      : base(std::forward<Args>(args)...), m_semi_axes(base::template get_parameter<std::vector<value_type>>("semi_axes")) {
    if (m_semi_axes.size() != 3)
      throw std::invalid_argument(base::name() + ": semi_axes takes three components");
    for (auto const a : m_semi_axes)
      if (!(a > 0))
        throw std::invalid_argument(base::name() + ": semi_axes must be positive");
    base::publish();
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::vector<value_type>>(para, "semi_axes");
    return para;
  }

  [[nodiscard]] tensor2 shape() const { return eshelby::depolarisation(m_semi_axes[0], m_semi_axes[1], m_semi_axes[2]); }

private:
  std::vector<value_type> const &m_semi_axes;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DEPOLARISATION_ELLIPSOID_H
