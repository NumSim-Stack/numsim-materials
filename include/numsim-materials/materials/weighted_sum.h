#ifndef WEIGHTED_SUM_H
#define WEIGHTED_SUM_H

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Generic N-term weighted sum of tensor-valued materials.
///
///   stress  = Σ wᵢ * σᵢ
///   tangent = Σ wᵢ * Cᵢ
///
/// Each term is a (weight_material, constituent_material) pair.
/// The weight material must produce a scalar property "value".
/// The constituent material must produce "stress" (rank-2) and "tangent" (rank-4).
///
/// Parameters:
///   "name":  material name
///   "terms": vector<pair<string,string>> — (weight_name, constituent_name) pairs
template <typename Traits>
class weighted_sum final
    : public material_base<weighted_sum<Traits>, Traits> {
public:
  using base = material_base<weighted_sum<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;
  using terms_type = std::vector<std::pair<std::string, std::string>>;

  template <typename... Args>
  explicit weighted_sum(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_stress(base::template add_output<tensor2>("stress", &weighted_sum::update_stress)),
        m_tangent(base::template add_output<tensor4>("tangent", &weighted_sum::update_tangent)),
        m_terms_param(base::template get_parameter<terms_type>("terms")),
        m_weight_property(base::template get_parameter<std::string>("weight_property")),
        m_stress_property(base::template get_parameter<std::string>("stress_property")),
        m_tangent_property(base::template get_parameter<std::string>("tangent_property"))
  {
    // Dynamically create inputs for each term
    for (const auto& [weight_name, mat_name] : m_terms_param) {
      auto& w = base::template add_input<value_type>(weight_name, m_weight_property, EdgeKind::Global);
      auto& s = base::template add_input<tensor2>(mat_name, m_stress_property, EdgeKind::Global);
      auto& c = base::template add_input<tensor4>(mat_name, m_tangent_property, EdgeKind::Global);
      m_terms.push_back({&w, &s, &c});
    }

  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<terms_type>("terms").template add<is_required>();
    para.template insert<std::string>("weight_property")
        .template add<set_default>(std::string{"value"});
    para.template insert<std::string>("stress_property")
        .template add<set_default>(std::string{"stress"});
    para.template insert<std::string>("tangent_property")
        .template add<set_default>(std::string{"tangent"});
    return para;
  }

  void update() override {
    update_stress();
    update_tangent();
  }

  void update_stress() noexcept {
    m_stress = tensor2{};
    for (const auto& [w, sig, _] : m_terms) {
      m_stress += w->get() * sig->get();
    }
  }

  void update_tangent() noexcept {
    m_tangent = tensor4{};
    for (const auto& [w, _, C] : m_terms) {
      m_tangent += w->get() * C->get();
    }
  }

private:
  struct term {
    const input_property<value_type, property_traits>* weight;
    const input_property<tensor2, property_traits>* stress;
    const input_property<tensor4, property_traits>* tangent;
  };

  tensor2& m_stress;
  tensor4& m_tangent;
  const terms_type& m_terms_param;
  const std::string& m_weight_property;
  const std::string& m_stress_property;
  const std::string& m_tangent_property;
  std::vector<term> m_terms;
};

} // namespace numsim::materials

#endif // WEIGHTED_SUM_H
