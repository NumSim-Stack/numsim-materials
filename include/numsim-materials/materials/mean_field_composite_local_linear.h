#ifndef NUMSIM_MATERIALS_MEAN_FIELD_COMPOSITE_LOCAL_LINEAR_H
#define NUMSIM_MATERIALS_MEAN_FIELD_COMPOSITE_LOCAL_LINEAR_H

#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/materials/detail/mean_field_composite_local_base.h"

namespace numsim::materials {

/// A composite of phases at their local strains with a linear localisation (concentrations from fixed
/// tangents): one pass per update, the tangent exact. Outputs "stress", "tangent".
///
/// Parameters: "phases" [{"concentration", "stress", "tangent", "fraction"}: "material::property"].
template <typename Traits>
class mean_field_composite_local_linear final
    : public detail::mean_field_composite_local_base<mean_field_composite_local_linear<Traits>, Traits> {
public:
  using base = detail::mean_field_composite_local_base<mean_field_composite_local_linear<Traits>, Traits>;

  template <typename... Args>
  explicit mean_field_composite_local_linear(Args &&...args) : base(std::forward<Args>(args)...) {
    base::register_outputs().reads(base::m_A, base::m_sigma, base::m_C, base::m_c);
  }

  static constexpr char const *type_name() { return "mean_field_composite_local_linear"; }
  static std::vector<std::string> const &roles() {
    static std::vector<std::string> const r{"concentration", "stress", "tangent", "fraction"};
    return r;
  }

  void compute() { base::average(); }
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_MEAN_FIELD_COMPOSITE_LOCAL_LINEAR_H
