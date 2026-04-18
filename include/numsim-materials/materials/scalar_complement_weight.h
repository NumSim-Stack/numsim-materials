#ifndef NUMSIM_MATERIALS_SCALAR_COMPLEMENT_WEIGHT_H
#define NUMSIM_MATERIALS_SCALAR_COMPLEMENT_WEIGHT_H

#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

/// Reads a history property's new_value and produces 1 - value.
template <typename Traits>
class scalar_complement_weight final
    : public material_base<scalar_complement_weight<Traits>, Traits> {
public:
  using base = material_base<scalar_complement_weight<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit scalar_complement_weight(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_value(base::template add_output<value_type>("value", &scalar_complement_weight::update)),
        m_input(base::template add_input_history<value_type>(
            connection_source::parse(base::template get_parameter<std::string>("source")),
            EdgeKind::Global))
  {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("source").template add<is_required>();
    return para;
  }

  void update() override {
    m_value = value_type{1} - m_input.new_value();
  }

private:
  value_type& m_value;
  const input_history<value_type, property_traits>& m_input;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_SCALAR_COMPLEMENT_WEIGHT_H
