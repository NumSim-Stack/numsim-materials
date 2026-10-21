#ifndef NUMSIM_MATERIALS_DETAIL_ESHELBY_TENSOR_BASE_H
#define NUMSIM_MATERIALS_DETAIL_ESHELBY_TENSOR_BASE_H

#include <stdexcept>
#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/materials/detail/direction.h"

namespace numsim::materials::detail {

/// Common part of the Eshelby materials: a consumer of Poisson's ratio ("nu_source", "nu_property"),
/// producer of the output "tensor"; Derived provides shape(nu). Composites consume "tensor" like any
/// other property; the engine re-evaluates it when nu changes.
template <typename Derived, typename Traits>
class eshelby_tensor_base : public material_base<Derived, Traits> {
public:
  using base = material_base<Derived, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  static_assert(Dim == 3, "eshelby tensors are three-dimensional");
  using tensor4 = tmech::tensor<value_type, 3, 4>;

  template <typename... Args>
  explicit eshelby_tensor_base(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_nu(base::template add_input<value_type>(base::template get_parameter<std::string>("nu_source"),
                                                  base::template get_parameter<std::string>("nu_property"), EdgeKind::Global)) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "nu_source");
    optional<std::string>(para, "nu_property", std::string{"value"});
    return para;
  }

  void compute() { *m_S = static_cast<Derived const &>(*this).shape(m_nu.get()); }

protected:
  /// Called by the shape's own constructor: binding the update needs the most-derived object.
  void register_output() {
    m_S = &base::template add_output<tensor4>("tensor", &eshelby_tensor_base::compute);
    base::constant_output(*m_S);
  }

private:
  tensor4 *m_S{nullptr};
  input_property<value_type, property_traits> const &m_nu;
};

} // namespace numsim::materials::detail

#endif // NUMSIM_MATERIALS_DETAIL_ESHELBY_TENSOR_BASE_H
