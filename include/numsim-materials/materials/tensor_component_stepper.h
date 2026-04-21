#ifndef TENSOR_COMPONENT_STEPPER_H
#define TENSOR_COMPONENT_STEPPER_H

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

template <std::size_t Rank, typename Traits>
class tensor_component_stepper
    : public material_base<tensor_component_stepper<Rank, Traits>, Traits> {
public:
  using base = material_base<tensor_component_stepper<Rank, Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor = tmech::tensor<value_type, Dim, Rank>;
  using indices = std::vector<std::size_t>;

  static_assert(Rank == 1 || Rank == 2, "tensor_component_stepper::update only for rank==1,2");

  /**
   * @brief Constructs a tensor component stepper instance with the given arguments.
   *
   * This constructor initializes reaction parameters, registers properties
   * and functions, and queries required external properties and functions.
   *
   * @tparam Args Variadic template for forwarding constructor arguments.
   * @param args Arguments to initialize the base class.
   */
  template <typename... Args>
  explicit tensor_component_stepper(Args &&...args)
      : base(std::forward<Args>(args)...),
        // produced properties
        m_tensor(base::template add_output<tensor>("strain", &tensor_component_stepper::update)),
        m_inc(base::template get_parameter<value_type>("increment")),
        m_indices(base::template get_parameter<indices>("indices"))
  {}

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
    para.template insert<value_type>("increment").template add<is_required>();
    para.template insert<indices>("indices").template add<is_required>();
    return para;
  }

  /// Accumulates the increment into the specified tensor component.
  /// NOTE: This mutates via +=, not assignment. Calling update() multiple
  /// times per timestep (e.g., via update_property) will double-increment.
  /// The numerical_diff_checker handles this by excluding the stepper's
  /// callback during perturbation.
  void update() override {
    if constexpr (Rank == 1){
      m_tensor(m_indices[0]) += m_inc;
    }

    if constexpr (Rank == 2){
      m_tensor(m_indices[0], m_indices[1]) += m_inc;
    }
  }

private:
  /// Produced properties
  tensor &m_tensor;

  /// Parameters
  const value_type &m_inc;
  const std::vector<std::size_t> &m_indices;
};

template<typename Traits>
using tensor_component_stepper_rank1 = tensor_component_stepper<1, Traits>;

template<typename Traits>
using tensor_component_stepper_rank2 = tensor_component_stepper<2, Traits>;

}

#endif // TENSOR_COMPONENT_STEPPER_H
