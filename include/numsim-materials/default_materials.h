#ifndef NUMSIM_MATERIALS_DEFAULT_MATERIALS_H
#define NUMSIM_MATERIALS_DEFAULT_MATERIALS_H

#include <numsim-core/object_registry.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/materials/scalar_stepper.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/autocatalytic_reaction.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/materials/scalar_identity_weight.h"
#include "numsim-materials/materials/scalar_complement_weight.h"
#include "numsim-materials/materials/weighted_sum.h"
#include "numsim-materials/postprocessing/property_console_logger.h"

namespace numsim::materials {

/// Material factory alias: maps string keys to material constructors.
template<typename Traits>
using material_factory = numsim_core::object_registry<
    material_interface<Traits>,
    typename Traits::ParameterHandler,
    typename Traits::PropertyHandler,
    typename Traits::MaterialHandler>;

/// Default 3D trait policy (alias for material_policy_default).
using material_policy_3d = material_policy_default;

/// Default 2D trait policy.
struct material_policy_2d {
  using PropertyHandler = material_policy_default::PropertyHandler;
  using ParameterHandler = material_policy_default::ParameterHandler;
  using MaterialHandler = material_policy_default::MaterialHandler;
  using InputParameterController = material_policy_default::InputParameterController;
  using value_type = double;
  static constexpr std::size_t Dim = 2;
};

/// Register all built-in types for a given Traits policy.
template<typename Traits>
void register_default_materials() {
  auto& factory = material_factory<Traits>::instance();
  factory.template register_type<scalar_stepper<Traits>>("scalar_stepper");
  factory.template register_type<linear_elasticity<Traits>>("linear_elasticity");
  factory.template register_type<autocatalytic_reaction<Traits>>("autocatalytic_reaction");
  factory.template register_type<backward_euler<Traits>>("backward_euler");
  factory.template register_type<tensor_component_stepper<2, Traits>>("tensor_stepper_rank2");
  factory.template register_type<tensor_component_stepper<1, Traits>>("tensor_stepper_rank1");
  factory.template register_type<scalar_identity_weight<Traits>>("scalar_identity_weight");
  factory.template register_type<scalar_complement_weight<Traits>>("scalar_complement_weight");
  factory.template register_type<weighted_sum<Traits>>("weighted_sum");
  factory.template register_type<property_console_logger<Traits>>("property_console_logger");
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DEFAULT_MATERIALS_H
