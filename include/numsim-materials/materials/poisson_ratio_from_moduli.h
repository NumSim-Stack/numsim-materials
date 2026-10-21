#ifndef NUMSIM_MATERIALS_POISSON_RATIO_FROM_MODULI_H
#define NUMSIM_MATERIALS_POISSON_RATIO_FROM_MODULI_H

#include <stdexcept>
#include <string>

#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

/// Poisson's ratio (3K - 2G) / (2 (3K + G)) of the parameters "K" and "G" as the constant output
/// "value": computed once at construction, no update callback (like constant_scalar). A ratio that has
/// to follow a tangent comes from isotropic_moduli instead.
template <typename Traits>
class poisson_ratio_from_moduli final : public material_base<poisson_ratio_from_moduli<Traits>, Traits> {
public:
  using base = material_base<poisson_ratio_from_moduli<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit poisson_ratio_from_moduli(Args &&...args)
      : base(std::forward<Args>(args)...), m_nu(base::template add_output<value_type>("value")) {
    auto const K{base::template get_parameter<value_type>("K")};
    auto const G{base::template get_parameter<value_type>("G")};
    if (!(K > 0 && G > 0))
      throw std::invalid_argument("poisson_ratio_from_moduli '" + base::name() + "': K and G must be positive");
    m_nu = poisson_ratio(K, G);
    base::constant_output(m_nu);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<value_type>(para, "K");
    required<value_type>(para, "G");
    return para;
  }

private:
  value_type &m_nu;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_POISSON_RATIO_FROM_MODULI_H
