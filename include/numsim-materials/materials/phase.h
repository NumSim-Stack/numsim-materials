#ifndef NUMSIM_MATERIALS_PHASE_H
#define NUMSIM_MATERIALS_PHASE_H

#include <stdexcept>
#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

/// A phase of a composite as produced properties: a tangent, a volume fraction and, for the mean-field
/// schemes, the Eshelby tensor of its shape (elasticity) or its depolarisation tensor (conductivity).
/// Outputs "tangent", "volume_fraction" and, when wired, "eshelby" and "depolarisation". A phase holds no
/// material: a composite consumes these properties and never reaches into another node.
///
/// Parameters: "tangent_source", "fraction_source", "eshelby_source", "depolarisation_source" and their
/// "_property" names (defaults "tangent", "value", "tensor", "tensor").
template <typename Traits>
class phase final : public material_base<phase<Traits>, Traits> {
public:
  using base = material_base<phase<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit phase(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_tangent(base::template add_output<tensor4>("tangent", &phase::compute)),
        m_fraction(base::template add_output<value_type>("volume_fraction")),
        m_tangent_source(base::template add_input<tensor4>(base::template get_parameter<std::string>("tangent_source"),
                                                           base::template get_parameter<std::string>("tangent_property"), EdgeKind::Global)),
        m_fraction_source(base::template add_input<value_type>(base::template get_parameter<std::string>("fraction_source"),
                                                               base::template get_parameter<std::string>("fraction_property"), EdgeKind::Global)) {
    auto const &eshelby{base::template get_parameter<std::string>("eshelby_source")};
    auto const &depolarisation{base::template get_parameter<std::string>("depolarisation_source")};
    if constexpr (Dim == 3) {
      if (!eshelby.empty()) {
        m_eshelby = &base::template add_output<tensor4>("eshelby");
        m_eshelby_source = &base::template add_input<tensor4>(eshelby, base::template get_parameter<std::string>("eshelby_property"), EdgeKind::Global);
      }
      if (!depolarisation.empty()) {
        m_depolarisation = &base::template add_output<tensor2>("depolarisation");
        m_depolarisation_source = &base::template add_input<tensor2>(depolarisation, base::template get_parameter<std::string>("depolarisation_property"),
                                                                     EdgeKind::Global);
      }
    } else if (!eshelby.empty() || !depolarisation.empty()) {
      throw std::invalid_argument("phase '" + base::name() + "': Eshelby and depolarisation tensors are three-dimensional");
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "tangent_source");
    optional<std::string>(para, "tangent_property", std::string{"tangent"});
    required<std::string>(para, "fraction_source");
    optional<std::string>(para, "fraction_property", std::string{"value"});
    optional<std::string>(para, "eshelby_source", std::string{});
    optional<std::string>(para, "eshelby_property", std::string{"tensor"});
    optional<std::string>(para, "depolarisation_source", std::string{});
    optional<std::string>(para, "depolarisation_property", std::string{"tensor"});
    return para;
  }

  /// One callback for every output.
  void compute() {
    m_tangent = m_tangent_source.get();
    m_fraction = m_fraction_source.get();
    if (!(m_fraction >= 0))
      throw std::invalid_argument("phase '" + base::name() + "': the volume fraction must be non-negative");
    if (m_eshelby)
      *m_eshelby = m_eshelby_source->get();
    if (m_depolarisation)
      *m_depolarisation = m_depolarisation_source->get();
  }

private:
  tensor4 &m_tangent;
  value_type &m_fraction;
  input_property<tensor4, property_traits> const &m_tangent_source;
  input_property<value_type, property_traits> const &m_fraction_source;
  tensor4 *m_eshelby{nullptr}; ///< null without an eshelby_source
  input_property<tensor4, property_traits> const *m_eshelby_source{nullptr};
  tensor2 *m_depolarisation{nullptr}; ///< null without a depolarisation_source
  input_property<tensor2, property_traits> const *m_depolarisation_source{nullptr};
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_PHASE_H
