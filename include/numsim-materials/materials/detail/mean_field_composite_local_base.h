#ifndef NUMSIM_MATERIALS_DETAIL_MEAN_FIELD_COMPOSITE_LOCAL_BASE_H
#define NUMSIM_MATERIALS_DETAIL_MEAN_FIELD_COMPOSITE_LOCAL_BASE_H

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/tensor_identities.h"

namespace numsim::materials {

/// Per phase the "material::property" sources of a local composite, by role.
using phase_sources = std::vector<std::map<std::string, std::string>>;

namespace detail {

/// The average of phases evaluated at their local strains: per phase a strain concentration, a stress, a
/// tangent and a volume fraction, from whichever nodes produce them; stress = sum_r c_r sigma_r and tangent =
/// sum_r c_r C_r : A_r. The concentrations must satisfy sum_r c_r A_r = I (checked per update).
template <typename Derived, typename Traits>
class mean_field_composite_local_base : public material_base<Derived, Traits> {
public:
  using base = material_base<Derived, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit mean_field_composite_local_base(Args &&...args)
      : base(std::forward<Args>(args)...) {
    auto const &phases{base::template get_parameter<phase_sources>("phases")};
    if (phases.empty())
      throw std::invalid_argument(where() + "needs at least one phase");
    for (auto const &phase : phases) {
      for (auto const &[role, source] : phase)
        if (std::ranges::find(Derived::roles(), role) == Derived::roles().end())
          throw std::invalid_argument(where() + "a phase has no role '" + role + "'");
      m_A.push_back(&input<tensor4>(phase, "concentration"));
      m_sigma.push_back(&input<tensor2>(phase, "stress"));
      m_C.push_back(&input<tensor4>(phase, "tangent"));
      m_c.push_back(&input<value_type>(phase, "fraction"));
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<phase_sources>(para, "phases");
    return para;
  }

protected:
  /// Called by the node's own constructor: binding the update needs the most-derived object.
  void register_outputs() {
    m_stress = &base::template add_output<tensor2>("stress", &Derived::compute);
    m_tangent = &base::template add_output<tensor4>("tangent");
  }

  /// The sums; the derived node adds what it reports on top.
  void average() {
    *m_stress = tensor2{};
    *m_tangent = tensor4{};
    tensor4 partition;
    for (std::size_t r{0}; r < m_c.size(); ++r) {
      auto const c{m_c[r]->get()};
      *m_stress += c * m_sigma[r]->get();
      *m_tangent += c * tmech::dcontract(m_C[r]->get(), m_A[r]->get());
      partition += c * m_A[r]->get();
    }
    auto const &Is{sym_identity<value_type, Dim>()};
    if (tmech::norm(tensor4{partition - Is}) > value_type{1e-8} * tmech::norm(Is))
      throw std::invalid_argument(where() + "the concentrations weighted by the fractions do not sum to the identity");
  }

  template <typename T>
  input_property<T, property_traits> const &input(std::map<std::string, std::string> const &phase, std::string const &role) {
    auto const it{phase.find(role)};
    if (it == phase.end())
      throw std::invalid_argument(where() + "a phase names no '" + role + "'");
    auto const src{connection_source::parse(it->second)};
    return base::template add_input<T>(src.material, src.property, EdgeKind::Global);
  }

  std::string where() const { return std::string{Derived::type_name()} + " '" + base::name() + "': "; }

  tensor2 *m_stress{nullptr};
  tensor4 *m_tangent{nullptr};
  std::vector<input_property<tensor4, property_traits> const *> m_A, m_C;
  std::vector<input_property<tensor2, property_traits> const *> m_sigma;
  std::vector<input_property<value_type, property_traits> const *> m_c;
};

} // namespace detail

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DETAIL_MEAN_FIELD_COMPOSITE_LOCAL_BASE_H
