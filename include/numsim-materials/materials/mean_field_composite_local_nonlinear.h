#ifndef NUMSIM_MATERIALS_MEAN_FIELD_COMPOSITE_LOCAL_NONLINEAR_H
#define NUMSIM_MATERIALS_MEAN_FIELD_COMPOSITE_LOCAL_NONLINEAR_H

#include <algorithm>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/materials/detail/mean_field_composite_local_base.h"

namespace numsim::materials {

/// A composite of nonlinear phases at their local strains: the localisation used the tangents of the
/// previous pass ("used_tangent", the phase node's on a local edge), so "residual" = max_r ||C_r - C_used,r||
/// / ||C_r|| says when the passes agree. Outputs "stress", "tangent", "residual".
///
/// Parameters: "phases" [{"concentration", "stress", "tangent", "fraction", "used_tangent" (the phases whose
/// tangent changes)}: "material::property"].
template <typename Traits>
class mean_field_composite_local_nonlinear final
    : public detail::mean_field_composite_local_base<mean_field_composite_local_nonlinear<Traits>, Traits> {
public:
  using base = detail::mean_field_composite_local_base<mean_field_composite_local_nonlinear<Traits>, Traits>;
  using typename base::value_type;
  using typename base::tensor4;

  template <typename... Args>
  explicit mean_field_composite_local_nonlinear(Args &&...args)
      : base(std::forward<Args>(args)...), m_residual(base::template add_output<value_type>("residual")) {
    auto const &phases{base::template get_parameter<phase_sources>("phases")};
    for (std::size_t r{0}; r < phases.size(); ++r)
      if (phases[r].contains("used_tangent"))
        m_used.push_back({r, &base::template input<tensor4>(phases[r], "used_tangent")});
    if (m_used.empty())
      throw std::invalid_argument(base::where() + "no phase names the 'used_tangent' its strain was localised with");
    base::register_outputs(m_residual).reads(base::m_A, base::m_sigma, base::m_C, base::m_c, m_used | std::views::values);
  }

  static constexpr char const *type_name() { return "mean_field_composite_local_nonlinear"; }
  static std::vector<std::string> const &roles() {
    static std::vector<std::string> const r{"concentration", "stress", "tangent", "fraction", "used_tangent"};
    return r;
  }

  void compute() {
    base::average();
    m_residual = 0;
    for (auto const &[r, used] : m_used) {
      auto const &C{base::m_C[r]->get()};
      m_residual = std::max(m_residual, tmech::norm(tensor4{C - used->get()}) / tmech::norm(C));
    }
  }

private:
  value_type &m_residual;
  std::vector<std::pair<std::size_t, input_property<tensor4, property_traits> const *>> m_used;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_MEAN_FIELD_COMPOSITE_LOCAL_NONLINEAR_H
