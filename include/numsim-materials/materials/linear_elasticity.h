#ifndef LINEAR_ELASTICITY_H
#define LINEAR_ELASTICITY_H

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

template <typename Traits>
class linear_elasticity
    : public material_base<linear_elasticity<Traits>, Traits> {
public:
  using base = material_base<linear_elasticity<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;


  /**
   * @brief Constructs a linear elasticity instance with the given arguments.
   *
   * This constructor initializes reaction parameters, registers properties
   * and functions, and queries required external properties and functions.
   *
   * @tparam Args Variadic template for forwarding constructor arguments.
   * @param args Arguments to initialize the base class.
   */
  template <typename... Args>
  explicit linear_elasticity(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_sig(base::template add_output<tensor2>("stress", &linear_elasticity::update_stress)),
        m_C(base::template add_output<tensor4>("tangent")),
        m_eps_name(base::template get_parameter<std::string>("strain_producer_name")),
        m_K(base::template get_parameter<value_type>("K")),
        m_G(base::template get_parameter<value_type>("G")),
        m_eps(base::template add_input<tensor2>(m_eps_name, "strain", EdgeKind::Global))
  {
    update_tangent();
  }

  /**
   * @brief Defines the parameters required by this class.
   *
   * This function specifies the expected parameters, marking some as required
   * and providing default values for others.
   *
   * @return An input parameter controller defining the parameter structure.
   */
  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<value_type>("K").template add<is_required>();
    para.template insert<value_type>("G").template add<is_required>();
    para.template insert<std::string>("strain_producer_name")
        .template add<is_required>();
    return para;
  }

  void update_stress(){
    m_sig = tmech::dcontract(m_C, m_eps.get());
  }

  void update_tangent(){
    const auto I{tmech::eye<value_type, Dim, 2>()};
    const auto IIsym{(tmech::otimesu(I,I) + (tmech::otimesl(I,I)))*0.5};
    const auto IIvol{tmech::otimes(I, I)/Dim};
    const auto IIdev{IIsym - IIvol};
    m_C = 3*m_K*IIvol + 2*m_G*IIdev;
  }

private:
  /// Produced properties
  tensor2 &m_sig;
  tensor4 &m_C;

  /// Parameters
  const value_type &m_K;
  const value_type &m_G;
  const std::string &m_eps_name;

  /// Consumed properties
  const input_property<tensor2, property_traits> &m_eps;
};

}
#endif // LINEAR_ELASTICITY_H
