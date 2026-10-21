#ifndef NUMSIM_MATERIALS_MEAN_FIELD_COMPOSITE_GLOBAL_H
#define NUMSIM_MATERIALS_MEAN_FIELD_COMPOSITE_GLOBAL_H

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/object_spec.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/mean_field/kernel_registry.h"

namespace numsim::materials {

/// Effective tangent of phases (phase nodes: "tangent", "volume_fraction", "eshelby" where the kernel reads
/// it), homogenised once per update by a kernel object: tangent = sum_r c_r C_r : A_r. A tangent producer:
/// outputs "tangent" and "concentration_<phase>"; the stress is a linear_stress on the tangent.
///
/// Parameters: "phases" (the matrix first: dilute and Mori-Tanaka embed in it), "kernel" {"type": voigt |
/// reuss | hill | dilute | mori_tanaka | self_consistent, ...}.
template <typename Traits>
class mean_field_composite_global final : public material_base<mean_field_composite_global<Traits>, Traits> {
public:
  using base = material_base<mean_field_composite_global<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  static_assert(Dim == 3, "mean_field_composite_global: the Eshelby tensors are three-dimensional");
  using tensor4 = tmech::tensor<value_type, 3, 4>;
  using spec = object_spec<typename base::parameter_handler>;

  template <typename... Args>
  explicit mean_field_composite_global(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_tangent(base::template add_output<tensor4>("tangent")),
        m_kernel(mean_field::kernel_registry<Traits>::instance().make(base::template get_parameter<spec>("kernel"))) {
    auto const &names{base::template get_parameter<std::vector<std::string>>("phases")};
    if (names.empty())
      throw std::invalid_argument("mean_field_composite_global '" + base::name() + "': needs at least one phase");
    m_kernel->bind(names);
    bool const eshelby{m_kernel->needs().eshelby};
    for (std::size_t r{0}; r < names.size(); ++r) {
      m_C.push_back(&base::template add_input<tensor4>(names[r], "tangent", EdgeKind::Global));
      m_c.push_back(&base::template add_input<value_type>(names[r], "volume_fraction", EdgeKind::Global));
      m_S.push_back(eshelby && r > 0 ? &base::template add_input<tensor4>(names[r], "eshelby", EdgeKind::Global) : nullptr);
      m_A.push_back(&base::template add_output<tensor4>("concentration_" + names[r]));
    }
    // constant phases: the kernel runs once
    base::add_callback(&mean_field_composite_global::compute, m_tangent, m_A).constant();
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::vector<std::string>>(para, "phases");
    required<spec>(para, "kernel");
    return para;
  }

  void compute() {
    std::vector<mean_field::phase_view<value_type>> phases;
    value_type total{0};
    for (std::size_t r{0}; r < m_C.size(); ++r) {
      phases.push_back({m_c[r]->get(), &m_C[r]->get(), m_S[r] ? &m_S[r]->get() : nullptr});
      total += phases.back().fraction;
    }
    if (std::abs(total - 1) > value_type{1e-10})
      throw std::invalid_argument("mean_field_composite_global '" + base::name() + "': the volume fractions must sum to one");
    std::vector<tensor4> A(phases.size());
    m_kernel->concentrations(phases, A);
    m_tangent = tensor4{};
    for (std::size_t r{0}; r < phases.size(); ++r) {
      *m_A[r] = A[r];
      m_tangent += phases[r].fraction * tmech::dcontract(*phases[r].tangent, A[r]);
    }
  }

private:
  tensor4 &m_tangent;
  std::unique_ptr<mean_field::kernel<Traits>> m_kernel;
  std::vector<input_property<tensor4, property_traits> const *> m_C, m_S;
  std::vector<input_property<value_type, property_traits> const *> m_c;
  std::vector<tensor4 *> m_A;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_MEAN_FIELD_COMPOSITE_GLOBAL_H
