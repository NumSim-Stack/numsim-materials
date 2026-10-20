#ifndef NUMSIM_MATERIALS_MEAN_FIELD_STRESS_H
#define NUMSIM_MATERIALS_MEAN_FIELD_STRESS_H

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

/// The composite stress and tangent of phases evaluated at their local strains (mean_field_strain):
/// "stress" = sum_r c_r sigma_r, "tangent" = sum_r c_r C_r : A_r with the materials' current stresses and
/// tangents and the concentrations of the strain node. "residual" is the largest relative difference
/// between a phase node's tangent (the one the concentrations were formed with) and its material's current
/// tangent: zero for linear phases, the fixed-point error of a nonlinear pass otherwise.
///
/// Parameters: "concentration_source" (the mean_field_strain), "matrix" and "phases" (phase nodes),
/// "matrix_stress_source" and "stress_sources" (the materials, parallel to "phases"), optional
/// "matrix_tangent_source" and "tangent_sources" (default: the stress sources), "stress_property",
/// "tangent_property" and, when the matrix material names them differently, "matrix_stress_property",
/// "matrix_tangent_property".
template <typename Traits>
class mean_field_stress final : public material_base<mean_field_stress<Traits>, Traits> {
public:
  using base = material_base<mean_field_stress<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit mean_field_stress(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_stress(base::template add_output<tensor2>("stress", &mean_field_stress::compute)),
        m_tangent(base::template add_output<tensor4>("tangent")),
        m_residual(base::template add_output<value_type>("residual")) {
    auto const &concentrations{base::template get_parameter<std::string>("concentration_source")};
    auto const &stress_property{base::template get_parameter<std::string>("stress_property")};
    auto const &tangent_property{base::template get_parameter<std::string>("tangent_property")};
    std::vector<std::string> phases{base::template get_parameter<std::vector<std::string>>("phases")};
    std::vector<std::string> stresses{base::template get_parameter<std::vector<std::string>>("stress_sources")};
    std::vector<std::string> tangents{base::template get_parameter<std::vector<std::string>>("tangent_sources")};
    if (tangents.empty())
      tangents = stresses;
    if (phases.empty() || stresses.size() != phases.size() || tangents.size() != phases.size())
      throw std::invalid_argument("mean_field_stress '" + base::name() + "': phases, stress_sources and tangent_sources must be parallel lists");
    phases.insert(phases.begin(), base::template get_parameter<std::string>("matrix"));
    stresses.insert(stresses.begin(), base::template get_parameter<std::string>("matrix_stress_source"));
    auto const &matrix_tangent{base::template get_parameter<std::string>("matrix_tangent_source")};
    tangents.insert(tangents.begin(), matrix_tangent.empty() ? stresses.front() : matrix_tangent);
    auto const &matrix_stress_property{base::template get_parameter<std::string>("matrix_stress_property")};
    auto const &matrix_tangent_property{base::template get_parameter<std::string>("matrix_tangent_property")};
    for (std::size_t i{0}; i < phases.size(); ++i) {
      bool const matrix{i == 0};
      m_c.push_back(&base::template add_input<value_type>(phases[i], "volume_fraction", EdgeKind::Global));
      m_C_used.push_back(&base::template add_input<tensor4>(phases[i], "tangent", EdgeKind::Global));
      m_A.push_back(&base::template add_input<tensor4>(concentrations, "concentration_" + phases[i], EdgeKind::Global));
      m_sigma.push_back(&base::template add_input<tensor2>(
          stresses[i], matrix && !matrix_stress_property.empty() ? matrix_stress_property : stress_property, EdgeKind::Global));
      m_C.push_back(&base::template add_input<tensor4>(
          tangents[i], matrix && !matrix_tangent_property.empty() ? matrix_tangent_property : tangent_property, EdgeKind::Global));
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "concentration_source");
    required<std::string>(para, "matrix");
    required<std::vector<std::string>>(para, "phases");
    required<std::string>(para, "matrix_stress_source");
    required<std::vector<std::string>>(para, "stress_sources");
    optional<std::string>(para, "matrix_tangent_source", std::string{});
    optional<std::vector<std::string>>(para, "tangent_sources", std::vector<std::string>{});
    optional<std::string>(para, "stress_property", std::string{"stress"});
    optional<std::string>(para, "tangent_property", std::string{"tangent"});
    optional<std::string>(para, "matrix_stress_property", std::string{});
    optional<std::string>(para, "matrix_tangent_property", std::string{});
    return para;
  }

  /// One callback for every output.
  void compute() {
    m_stress = tensor2{};
    m_tangent = tensor4{};
    m_residual = 0;
    for (std::size_t i{0}; i < m_c.size(); ++i) {
      auto const c{m_c[i]->get()};
      auto const &C{m_C[i]->get()};
      m_stress += c * m_sigma[i]->get();
      m_tangent += c * tmech::dcontract(C, m_A[i]->get());
      m_residual = std::max(m_residual, tmech::norm(tensor4{C - m_C_used[i]->get()}) / tmech::norm(C));
    }
  }

private:
  tensor2 &m_stress;
  tensor4 &m_tangent;
  value_type &m_residual;
  std::vector<input_property<value_type, property_traits> const *> m_c;
  std::vector<input_property<tensor4, property_traits> const *> m_C_used; ///< the phase nodes' tangents
  std::vector<input_property<tensor4, property_traits> const *> m_A;
  std::vector<input_property<tensor2, property_traits> const *> m_sigma;
  std::vector<input_property<tensor4, property_traits> const *> m_C; ///< the materials' tangents
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_MEAN_FIELD_STRESS_H
