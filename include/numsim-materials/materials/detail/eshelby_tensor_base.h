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
/// other property; the engine re-evaluates it when nu changes. "nu_edge" = "local" reads nu without
/// ordering (the self-consistent cycle: nu of the effective medium), starting from "nu_initial".
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
        m_local(parse_edge(base::template get_parameter<std::string>("nu_edge"))),
        m_nu_initial(base::template get_parameter<value_type>("nu_initial")),
        m_nu(base::template add_input<value_type>(base::template get_parameter<std::string>("nu_source"),
                                                  base::template get_parameter<std::string>("nu_property"),
                                                  m_local ? EdgeKind::Local : EdgeKind::Global)) {
    if (m_local && !(m_nu_initial > -1 && m_nu_initial < value_type{0.5}))
      throw std::invalid_argument("eshelby node '" + base::name() + "': a local nu edge needs nu_initial in (-1, 0.5)");
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "nu_source");
    optional<std::string>(para, "nu_property", std::string{"value"});
    optional<std::string>(para, "nu_edge", std::string{"global"});
    optional<value_type>(para, "nu_initial", value_type{2}); // invalid on purpose: required with a local edge
    return para;
  }

  /// Over a local edge the first update has no produced nu yet and takes nu_initial.
  void compute() {
    *m_S = static_cast<Derived const &>(*this).shape(m_local && m_first ? m_nu_initial : m_nu.get());
    m_first = false;
  }

protected:
  /// Called by the shape's own constructor: binding the update needs the most-derived object.
  void register_output() { m_S = &base::template add_output<tensor4>("tensor", &eshelby_tensor_base::compute); }

private:
  static bool parse_edge(std::string const &e) {
    if (e == "global")
      return false;
    if (e == "local")
      return true;
    throw std::invalid_argument("eshelby node: unknown nu_edge '" + e + "' (global, local)");
  }

  tensor4 *m_S{nullptr};
  bool const m_local;        ///< nu from the previous pass (self-consistent: nu of the effective medium)
  value_type const m_nu_initial;
  bool m_first{true};
  input_property<value_type, property_traits> const &m_nu;
};

} // namespace numsim::materials::detail

#endif // NUMSIM_MATERIALS_DETAIL_ESHELBY_TENSOR_BASE_H
