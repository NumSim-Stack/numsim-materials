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
/// "_property" names (defaults "tangent", "value", "tensor", "tensor"); "tangent_edge" = "local" reads the
/// tangent without ordering (a nonlinear phase whose strain a mean_field_strain produces from this very
/// tangent: the value of the previous pass), starting from "initial_tangent_source" / "_property".
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
        m_tangent{&base::template add_output<tensor4>("tangent"),
                  input<tensor4>("tangent", parse_edge(base::template get_parameter<std::string>("tangent_edge")))},
        m_initial_tangent(input<tensor4>("initial_tangent")),
        m_fraction{&base::template add_output<value_type>("volume_fraction"), input<value_type>("fraction")},
        m_eshelby(optional_relay<tensor4>("eshelby")),
        m_depolarisation(optional_relay<tensor2>("depolarisation")) {
    if (!m_tangent.source || !m_fraction.source)
      throw std::invalid_argument("phase '" + base::name() + "': needs a tangent_source and a fraction_source");
    if ((m_tangent.source->edge_kind() == EdgeKind::Local) != (m_initial_tangent != nullptr))
      throw std::invalid_argument("phase '" + base::name() + "': initial_tangent_source goes with tangent_edge = local, and only then");
    if (Dim != 3 && (m_eshelby.output || m_depolarisation.output))
      throw std::invalid_argument("phase '" + base::name() + "': Eshelby and depolarisation tensors are three-dimensional");
    // one callback per relay: a varying fraction leaves the tensors constant, a local tangent the fraction
    base::add_callback(&phase::copy_tangent, *m_tangent.output).reads(m_tangent.source, m_initial_tangent);
    base::add_callback(&phase::copy_fraction, *m_fraction.output).reads(m_fraction.source);
    if (m_eshelby.output)
      base::add_callback(&phase::copy_eshelby, *m_eshelby.output).reads(m_eshelby.source);
    if (m_depolarisation.output)
      base::add_callback(&phase::copy_depolarisation, *m_depolarisation.output).reads(m_depolarisation.source);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "tangent_source");
    optional<std::string>(para, "tangent_property", std::string{"tangent"});
    optional<std::string>(para, "tangent_edge", std::string{"global"});
    optional<std::string>(para, "initial_tangent_source", std::string{});
    optional<std::string>(para, "initial_tangent_property", std::string{"tangent"});
    required<std::string>(para, "fraction_source");
    optional<std::string>(para, "fraction_property", std::string{"value"});
    optional<std::string>(para, "eshelby_source", std::string{});
    optional<std::string>(para, "eshelby_property", std::string{"tensor"});
    optional<std::string>(para, "depolarisation_source", std::string{});
    optional<std::string>(para, "depolarisation_property", std::string{"tensor"});
    return para;
  }

  /// Over a local edge the first update has no produced tangent yet.
  void copy_tangent() {
    *m_tangent.output = m_initial_tangent && m_first ? m_initial_tangent->get() : m_tangent.source->get();
    m_first = false;
  }

  void copy_fraction() {
    m_fraction.copy();
    if (!(*m_fraction.output >= 0))
      throw std::invalid_argument("phase '" + base::name() + "': the volume fraction must be non-negative");
  }

  void copy_eshelby() { m_eshelby.copy(); }
  void copy_depolarisation() { m_depolarisation.copy(); }

private:
  /// An output republishing an input; both null when the input is not wired.
  template <typename T>
  struct relay {
    T *output{nullptr};
    input_property<T, property_traits> const *source{nullptr};
    void copy() const { *output = source->get(); }
  };

  /// The input named by the parameters "<key>_source" and "<key>_property"; null without a source.
  template <typename T>
  input_property<T, property_traits> const *input(std::string const &key, EdgeKind edge = EdgeKind::Global) {
    auto const &source{base::template get_parameter<std::string>(key + "_source")};
    if (source.empty())
      return nullptr;
    return &base::template add_input<T>(source, base::template get_parameter<std::string>(key + "_property"), edge);
  }

  /// The output "<key>" republishing input(key), when that is wired.
  template <typename T>
  relay<T> optional_relay(std::string const &key) {
    auto const *source{input<T>(key)};
    return {source ? &base::template add_output<T>(key) : nullptr, source};
  }

  static EdgeKind parse_edge(std::string const &e) {
    if (e == "global")
      return EdgeKind::Global;
    if (e == "local")
      return EdgeKind::Local;
    throw std::invalid_argument("phase: unknown tangent_edge '" + e + "' (global, local)");
  }

  relay<tensor4> m_tangent;
  input_property<tensor4, property_traits> const *m_initial_tangent; ///< with a local edge only
  relay<value_type> m_fraction;
  relay<tensor4> m_eshelby;
  relay<tensor2> m_depolarisation;
  bool m_first{true};
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_PHASE_H
