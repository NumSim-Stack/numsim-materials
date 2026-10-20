#ifndef NUMSIM_MATERIALS_MEAN_FIELD_STRAIN_H
#define NUMSIM_MATERIALS_MEAN_FIELD_STRAIN_H

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/object_spec.h"
#include "numsim-materials/mean_field/kernel_registry.h"

namespace numsim::materials {

/// Strain localisation of a mean-field scheme: from the composite strain increment and the phases' tangents,
/// fractions and (dilute, Mori-Tanaka) Eshelby tensors, its kernel object gives the concentrations, and every
/// phase gets "strain_<phase>" = strain_n + A_r : d_strain (history) and "concentration_<phase>". The phase
/// materials consume those strains; a composite averages what they produce. A nonlinear phase closes a cycle
/// (its tangent depends on its strain): its phase node reads the tangent over a local edge, so one update()
/// is one pass, repeated until the composite's residual says the passes agree.
///
/// Parameters: "strain_source" (history "strain"), "phases" (phase nodes, the matrix first), "kernel"
/// {"type": voigt | reuss | hill | dilute | mori_tanaka | self_consistent, ...} (the global composite's).
template <typename Traits>
class mean_field_strain final : public material_base<mean_field_strain<Traits>, Traits> {
public:
  using base = material_base<mean_field_strain<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  static_assert(Dim == 3, "mean_field_strain: the Eshelby tensors are three-dimensional");
  using tensor2 = tmech::tensor<value_type, 3, 2>;
  using tensor4 = tmech::tensor<value_type, 3, 4>;
  using spec = object_spec<typename base::parameter_handler>;

  template <typename... Args>
  explicit mean_field_strain(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_kernel(mean_field::kernel_registry<Traits>::instance().make(base::template get_parameter<spec>("kernel"))),
        m_strain(base::template add_input_history<tensor2>(
            connection_source{base::template get_parameter<std::string>("strain_source"), "strain"}, EdgeKind::Global)) {
    auto const &names{base::template get_parameter<std::vector<std::string>>("phases")};
    if (names.empty())
      throw std::invalid_argument("mean_field_strain '" + base::name() + "': needs at least one phase");
    m_kernel->bind(names);
    auto const needs{m_kernel->needs()};
    for (std::size_t r{0}; r < names.size(); ++r) {
      m_C.push_back(needs.tangents ? &base::template add_input<tensor4>(names[r], "tangent", EdgeKind::Global) : nullptr);
      m_c.push_back(&base::template add_input<value_type>(names[r], "volume_fraction", EdgeKind::Global));
      m_S.push_back(needs.eshelby && r > 0 ? &base::template add_input<tensor4>(names[r], "eshelby", EdgeKind::Global) : nullptr);
      m_A.push_back(&base::template add_output<tensor4>("concentration_" + names[r]));
      m_eps.push_back(&base::template add_history_output<tensor2>("strain_" + names[r], r == 0 ? &mean_field_strain::compute : nullptr));
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "strain_source");
    required<std::vector<std::string>>(para, "phases");
    required<spec>(para, "kernel");
    return para;
  }

  /// One callback for every output: the concentrations of the tangents as they stand, then the strains.
  void compute() {
    std::vector<mean_field::phase_view<value_type>> phases;
    value_type total{0};
    for (std::size_t r{0}; r < m_c.size(); ++r) {
      phases.push_back({m_c[r]->get(), m_C[r] ? &m_C[r]->get() : nullptr, m_S[r] ? &m_S[r]->get() : nullptr});
      total += phases.back().fraction;
    }
    if (std::abs(total - 1) > value_type{1e-10})
      throw std::invalid_argument("mean_field_strain '" + base::name() + "': the volume fractions must sum to one");
    std::vector<tensor4> A(phases.size());
    m_kernel->concentrations(phases, A);
    tensor2 const d_eps{m_strain.new_value() - m_strain.old_value()};
    for (std::size_t r{0}; r < phases.size(); ++r) {
      *m_A[r] = A[r];
      m_eps[r]->new_value() = m_eps[r]->old_value() + tmech::dcontract(A[r], d_eps);
    }
  }

private:
  std::unique_ptr<mean_field::kernel<Traits>> m_kernel;
  input_history<tensor2, property_traits> const &m_strain;
  std::vector<input_property<tensor4, property_traits> const *> m_C, m_S;
  std::vector<input_property<value_type, property_traits> const *> m_c;
  std::vector<tensor4 *> m_A;
  std::vector<history_property<tensor2> *> m_eps;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_MEAN_FIELD_STRAIN_H
