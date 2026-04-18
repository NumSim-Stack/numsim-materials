#ifndef SCALAR_STEPPER_H
#define SCALAR_STEPPER_H

#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/traits.h"

namespace numsim::materials {

template <typename Traits>
class scalar_stepper final : public material_base<scalar_stepper<Traits>, Traits> {
public:
  using base = material_base<scalar_stepper<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit scalar_stepper(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_his(base::template add_history_output<value_type>("state", &scalar_stepper::update)),
        m_inc(base::template get_parameter<value_type>("increment"))
  {}

  static input_parameter_controller parameters(){
    input_parameter_controller param{base::parameters()};
    param.template insert<value_type>("increment").template add<is_required>();
    return param;
  }

  void update() override {
    m_his.new_value() += m_inc;
  }

private:
  history_property<value_type> &m_his;
  const value_type &m_inc;
};

}

#endif // SCALAR_STEPPER_H
