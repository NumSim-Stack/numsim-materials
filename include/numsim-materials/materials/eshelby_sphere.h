#ifndef NUMSIM_MATERIALS_ESHELBY_SPHERE_H
#define NUMSIM_MATERIALS_ESHELBY_SPHERE_H

#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/materials/detail/eshelby_tensor_base.h"

namespace numsim::materials {

/// Eshelby tensor of a sphere in the isotropic "matrix"; output "tensor".
template <typename Traits>
class eshelby_sphere final : public detail::eshelby_tensor_base<eshelby_sphere<Traits>, Traits> {
public:
  using base = detail::eshelby_tensor_base<eshelby_sphere<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using tensor4 = typename base::tensor4;

  template <typename... Args>
  explicit eshelby_sphere(Args &&...args) : base(std::forward<Args>(args)...) {
    base::register_output();
  }

  static input_parameter_controller parameters() { return base::parameters(); }

  [[nodiscard]] tensor4 shape(value_type nu) const { return eshelby::sphere(nu); }
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_ESHELBY_SPHERE_H
