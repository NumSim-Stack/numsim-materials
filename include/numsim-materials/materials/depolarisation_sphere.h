#ifndef NUMSIM_MATERIALS_DEPOLARISATION_SPHERE_H
#define NUMSIM_MATERIALS_DEPOLARISATION_SPHERE_H

#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/materials/detail/depolarisation_tensor_base.h"

namespace numsim::materials {

/// Depolarisation tensor of a sphere, I / 3; output "tensor".
template <typename Traits>
class depolarisation_sphere final : public detail::depolarisation_tensor_base<depolarisation_sphere<Traits>, Traits> {
public:
  using base = detail::depolarisation_tensor_base<depolarisation_sphere<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using tensor2 = typename base::tensor2;

  template <typename... Args>
  explicit depolarisation_sphere(Args &&...args) : base(std::forward<Args>(args)...) {
    base::publish();
  }

  static input_parameter_controller parameters() { return base::parameters(); }

  [[nodiscard]] tensor2 shape() const noexcept { return eshelby::depolarisation(value_type{1}, value_type{1}, value_type{1}); }
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DEPOLARISATION_SPHERE_H
