#ifndef NUMSIM_MATERIALS_SCALAR_SPLIT_H
#define NUMSIM_MATERIALS_SCALAR_SPLIT_H

#include <stdexcept>
#include <string>
#include <vector>

#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

/// Splits a produced scalar by fixed weights: outputs "value_<i>" = weights[i] * source (e.g. a fibre
/// fraction over the orientation nodes).
///
/// Parameters: "source", "property" (default "value"), "weights".
template <typename Traits>
class scalar_split final : public material_base<scalar_split<Traits>, Traits> {
public:
  using base = material_base<scalar_split<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit scalar_split(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_weights(base::template get_parameter<std::vector<value_type>>("weights")),
        m_source(base::template add_input<value_type>(base::template get_parameter<std::string>("source"),
                                                      base::template get_parameter<std::string>("property"), EdgeKind::Global)) {
    if (m_weights.empty())
      throw std::invalid_argument("scalar_split '" + base::name() + "': weights must not be empty");
    for (std::size_t i{0}; i < m_weights.size(); ++i)
      m_parts.push_back(&base::template add_output<value_type>("value_" + std::to_string(i), i == 0 ? &scalar_split::compute : nullptr));
    for (std::size_t i{0}; i < m_weights.size(); ++i)
      base::constant_output("value_" + std::to_string(i), {&m_source});
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "source");
    optional<std::string>(para, "property", std::string{"value"});
    required<std::vector<value_type>>(para, "weights");
    return para;
  }

  void compute() {
    for (std::size_t i{0}; i < m_parts.size(); ++i)
      *m_parts[i] = m_weights[i] * m_source.get();
  }

private:
  std::vector<value_type> const &m_weights;
  input_property<value_type, property_traits> const &m_source;
  std::vector<value_type *> m_parts;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_SCALAR_SPLIT_H
