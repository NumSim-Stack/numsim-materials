#ifndef NUMSIM_MATERIALS_DETAIL_DEPOLARISATION_TENSOR_BASE_H
#define NUMSIM_MATERIALS_DETAIL_DEPOLARISATION_TENSOR_BASE_H

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials::detail {

/// Common part of the depolarisation materials: the output "tensor" is pure geometry in an isotropic
/// matrix, so a node is a producer without inputs; Derived provides shape(). Composites consume "tensor".
template <typename Derived, typename Traits>
class depolarisation_tensor_base : public material_base<Derived, Traits> {
public:
  using base = material_base<Derived, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  static_assert(Dim == 3, "depolarisation tensors are three-dimensional");
  using tensor2 = tmech::tensor<value_type, 3, 2>;

  template <typename... Args>
  explicit depolarisation_tensor_base(Args &&...args)
      : base(std::forward<Args>(args)...), m_N(base::template add_output<tensor2>("tensor")) {
    base::constant_output("tensor");
  }

  static input_parameter_controller parameters() { return base::parameters(); }

protected:
  /// Derived calls this once its geometry parameters are read.
  void publish() { m_N = static_cast<Derived const &>(*this).shape(); }

private:
  tensor2 &m_N;
};

} // namespace numsim::materials::detail

#endif // NUMSIM_MATERIALS_DETAIL_DEPOLARISATION_TENSOR_BASE_H
