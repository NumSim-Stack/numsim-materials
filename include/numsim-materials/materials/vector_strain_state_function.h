#ifndef NUMSIM_MATERIALS_VECTOR_STRAIN_STATE_FUNCTION_H
#define NUMSIM_MATERIALS_VECTOR_STRAIN_STATE_FUNCTION_H

#include <cmath>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Vector-based equivalent strain: eps_eq = |m^T * eps * n|
///
/// Uses direction vectors m and n to project the strain tensor
/// onto a scalar measure. Useful for directional damage models.
///
///   eps_eq = |dot(m, eps * n)|
///   d(eps_eq)/d(eps) = sym(m ⊗ n) * sign(dot(m, eps * n))
template <typename Traits>
class vector_strain_state_function final
    : public material_base<vector_strain_state_function<Traits>, Traits> {
public:
  using base = material_base<vector_strain_state_function<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  static constexpr auto Dim = base::Dim;
  using tensor1 = tmech::tensor<value_type, Dim, 1>;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;

  template <typename... Args>
  explicit vector_strain_state_function(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_eps_eq(base::template add_output<value_type>(
            "equivalent_strain", &vector_strain_state_function::compute)),
        m_deps_eq(base::template add_output<tensor2>("d_equivalent_strain")),
        m_strain_source(base::template get_parameter<std::string>("strain_source")),
        m_strain(base::template add_input<tensor2>(
            m_strain_source, "strain", EdgeKind::Global))
  {
    // Initialize direction vectors from parameters
    const auto& m_indices = base::template get_parameter<std::vector<std::size_t>>("m_direction");
    const auto& n_indices = base::template get_parameter<std::vector<std::size_t>>("n_direction");
    for (std::size_t i = 0; i < std::min(m_indices.size(), static_cast<std::size_t>(Dim)); ++i)
      m_m(i) = (i < m_indices.size()) ? value_type{1} : value_type{0};
    for (std::size_t i = 0; i < std::min(n_indices.size(), static_cast<std::size_t>(Dim)); ++i)
      m_n(i) = (i < n_indices.size()) ? value_type{1} : value_type{0};
    // Simple: use unit vectors along specified axes
    m_m = tensor1{};
    m_n = tensor1{};
    if (!m_indices.empty()) m_m(m_indices[0]) = value_type{1};
    if (!n_indices.empty()) m_n(n_indices[0]) = value_type{1};
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("strain_source").template add<is_required>();
    para.template insert<std::vector<std::size_t>>("m_direction").template add<is_required>();
    para.template insert<std::vector<std::size_t>>("n_direction").template add<is_required>();
    return para;
  }

  void compute() {
    const auto& eps = m_strain.get();
    const auto raw = tmech::dot(m_m, eps * m_n);
    m_eps_eq = std::abs(raw);

    const auto sgn = (raw >= value_type{0}) ? value_type{1} : value_type{-1};
    m_deps_eq = sgn * (tmech::otimes(m_m, m_n) + tmech::otimes(m_n, m_m)) * value_type{0.5};
  }

private:
  value_type& m_eps_eq;
  tensor2& m_deps_eq;
  const std::string& m_strain_source;
  const input_property<tensor2, property_traits>& m_strain;
  tensor1 m_m;
  tensor1 m_n;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_VECTOR_STRAIN_STATE_FUNCTION_H
