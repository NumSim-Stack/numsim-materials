#ifndef NUMSIM_MATERIALS_RULE_OF_MIXTURE_H
#define NUMSIM_MATERIALS_RULE_OF_MIXTURE_H

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/tensor_identities.h"

namespace numsim::materials {

/// Effective stiffness of phases (phase nodes producing "tangent" and "volume_fraction"): Voigt (equal
/// strain), Reuss (equal stress) or Hill (their mean). A tangent producer: outputs "tangent" and the strain
/// concentration tensors "concentration_<phase>" with sum_i c_i A_i = Isym; the stress is a linear_stress
/// on the tangent.
///
/// Parameters: "phases" (phase names), "kernel" (voigt | reuss | hill).
template <typename Traits>
class rule_of_mixture final : public material_base<rule_of_mixture<Traits>, Traits> {
public:
  using base = material_base<rule_of_mixture<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  enum class kernel_kind { voigt, reuss, hill };

  template <typename... Args>
  explicit rule_of_mixture(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_tangent(base::template add_output<tensor4>("tangent", &rule_of_mixture::compute)),
        m_phases(base::template get_parameter<std::vector<std::string>>("phases")),
        m_kernel(parse_kernel(base::template get_parameter<std::string>("kernel"))) {
    if (m_phases.empty())
      throw std::invalid_argument("rule_of_mixture '" + base::name() + "': needs at least one phase");
    for (auto const &phase : m_phases) {
      m_tangents.push_back(&base::template add_input<tensor4>(phase, "tangent", EdgeKind::Global));
      m_fractions.push_back(&base::template add_input<value_type>(phase, "volume_fraction", EdgeKind::Global));
      m_concentrations.push_back(&base::template add_output<tensor4>("concentration_" + phase));
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::vector<std::string>>(para, "phases");
    optional<std::string>(para, "kernel", std::string{"voigt"});
    return para;
  }

  /// One callback for every output.
  void compute() {
    auto const &Is{sym_identity<value_type, Dim>()};
    tensor4 voigt, compliance;
    value_type total{0};
    for (std::size_t i{0}; i < m_phases.size(); ++i) {
      auto const c{m_fractions[i]->get()};
      total += c;
      voigt += c * m_tangents[i]->get();
      if (m_kernel != kernel_kind::voigt)
        compliance += c * tmech::inv(m_tangents[i]->get());
    }
    if (std::abs(total - 1) > value_type{1e-10})
      throw std::invalid_argument("rule_of_mixture '" + base::name() + "': the volume fractions must sum to one");
    switch (m_kernel) {
    case kernel_kind::voigt:
      m_tangent = voigt;
      for (auto *A : m_concentrations)
        *A = Is;
      break;
    case kernel_kind::reuss:
      m_tangent = tmech::inv(compliance);
      for (std::size_t i{0}; i < m_phases.size(); ++i)
        *m_concentrations[i] = tmech::dcontract(tmech::inv(m_tangents[i]->get()), m_tangent);
      break;
    case kernel_kind::hill: {
      tensor4 const reuss{tmech::inv(compliance)};
      m_tangent = (voigt + reuss) / 2;
      for (std::size_t i{0}; i < m_phases.size(); ++i)
        *m_concentrations[i] = (Is + tmech::dcontract(tmech::inv(m_tangents[i]->get()), reuss)) / 2;
      break;
    }
    }
  }

private:
  static kernel_kind parse_kernel(std::string const &k) {
    if (k == "voigt")
      return kernel_kind::voigt;
    if (k == "reuss")
      return kernel_kind::reuss;
    if (k == "hill")
      return kernel_kind::hill;
    throw std::invalid_argument("rule_of_mixture: unknown kernel '" + k + "' (voigt, reuss, hill)");
  }

  tensor4 &m_tangent;
  std::vector<std::string> const &m_phases;
  kernel_kind m_kernel;
  std::vector<input_property<tensor4, property_traits> const *> m_tangents;
  std::vector<input_property<value_type, property_traits> const *> m_fractions;
  std::vector<tensor4 *> m_concentrations;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_RULE_OF_MIXTURE_H
