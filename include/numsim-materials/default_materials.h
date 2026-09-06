#ifndef DEFAULT_MATERIALS_H
#define DEFAULT_MATERIALS_H

#include <numsim-core/object_registry.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/solvers/local_newton.h"
#include "numsim-materials/solvers/vector_newton.h"
#include "numsim-materials/materials/scalar_stepper.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/props_scalar.h"
#include "numsim-materials/materials/j2_plasticity.h"
#include "numsim-materials/materials/j2_rk_plasticity.h"
#include "numsim-materials/materials/drucker_prager_plasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/exponential_isotropic_hardening.h"
#include "numsim-materials/materials/linear_damage_law.h"
#include "numsim-materials/materials/curing_rate.h"
#include "numsim-materials/materials/strain_energy_state_function.h"
#include "numsim-materials/materials/vector_strain_state_function.h"
#include "numsim-materials/materials/isotropic_tangent.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_stress.h"
#include "numsim-materials/materials/autocatalytic_reaction.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/materials/scalar_identity_weight.h"
#include "numsim-materials/materials/scalar_complement_weight.h"
#include "numsim-materials/materials/weighted_sum.h"
#include "numsim-materials/materials/von_mises_state_function.h"
#include "numsim-materials/materials/strain_threshold_yield.h"
#include "numsim-materials/materials/exponential_damage_law.h"
#include "numsim-materials/materials/isotropic_damage.h"
#include "numsim-materials/postprocessing/property_console_logger.h"

namespace numsim::materials {

/// Material factory alias: maps string keys to material constructors and schemas.
template<typename Traits>
using material_factory = numsim_core::object_registry<
    material_interface<Traits>,
    typename Traits::InputParameterController,
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

// --- Registration macros ---

// --- Registration macros ---

#define NUMSIM_CONCAT_IMPL(X, Y) X##Y
#define NUMSIM_CONCAT(X, Y) NUMSIM_CONCAT_IMPL(X, Y)

/// Register a material class using static initialization.
/// Place in a .cpp file alongside the explicit template instantiation.
///
/// Usage (in my_material.cpp):
///   template class my_material<material_policy_3d>;
///   NUMSIM_REGISTER_MATERIAL(material_factory<material_policy_3d>,
///                             my_material<material_policy_3d>, "my_material")
#define NUMSIM_REGISTER_MATERIAL(factory_type, classname, key) \
  namespace { \
    static const bool NUMSIM_CONCAT(numsim_reg_, __COUNTER__) = \
      factory_type::instance().template register_type<classname>(key); \
  }

/// Register all built-in types for a given Traits policy.
/// Call once at startup before using create_from_json().
template<typename Traits>
void register_default_materials() {
  auto& factory = material_factory<Traits>::instance();
  factory.template register_type<scalar_stepper<Traits>>("scalar_stepper");
  factory.template register_type<linear_elasticity<Traits>>("linear_elasticity");
  factory.template register_type<constant_scalar<Traits>>("constant_scalar");
  factory.template register_type<props_scalar<Traits>>("props_scalar");

  // Plasticity. Each model is its own class -- j2_plasticity,
  // j2_rk_plasticity and drucker_prager_plasticity -- rather than a template
  // over the yield-function type, so the registrable names are the classes.
  factory.template register_type<j2_plasticity<Traits>>("j2_plasticity");
  factory.template register_type<j2_rk_plasticity<Traits>>("j2_rk_plasticity");

  // Drucker-Prager was held back while its cone parameters arrived inside a
  // C++ "yield_function" object the JSON reader could not convert: a document
  // naming it would have got a DEFAULT-constructed one -- eta = beta = k = 0 --
  // which builds, runs, never yields, and is indistinguishable from
  // elasticity. #43 made eta, beta and K_bulk plain required scalars, so a
  // deck now either supplies them or gets a named missing-parameter error.
  // That was the stated condition for registering it (closes #33).
  factory.template register_type<drucker_prager_plasticity<Traits>>(
      "drucker_prager_plasticity");

  factory.template register_type<linear_isotropic_hardening<Traits>>(
      "linear_isotropic_hardening");
  factory.template register_type<exponential_isotropic_hardening<Traits>>(
      "exponential_isotropic_hardening");
  factory.template register_type<linear_damage_law<Traits>>("linear_damage_law");
  factory.template register_type<curing_rate<Traits>>("curing_rate");
  factory.template register_type<strain_energy_state_function<Traits>>(
      "strain_energy_state_function");
  factory.template register_type<vector_strain_state_function<Traits>>(
      "vector_strain_state_function");
  factory.template register_type<isotropic_tangent<Traits>>("isotropic_tangent");
  factory.template register_type<linear_stress<Traits>>("linear_stress");
  factory.template register_type<autocatalytic_reaction<Traits>>("autocatalytic_reaction");
  factory.template register_type<backward_euler<Traits>>("backward_euler");
  // local_newton is the solver every return map names through "solver_source".
  // Registering the plasticity classes without it leaves them unreachable from
  // a document anyway: the deck can name j2_plasticity but not the solver it
  // requires.
  factory.template register_type<local_newton<Traits>>("local_newton");
  factory.template register_type<tensor_component_stepper<1, Traits>>("tensor_component_stepper_rank1");
  factory.template register_type<tensor_component_stepper<2, Traits>>("tensor_component_stepper_rank2");
  factory.template register_type<scalar_identity_weight<Traits>>("scalar_identity_weight");
  factory.template register_type<scalar_complement_weight<Traits>>("scalar_complement_weight");
  factory.template register_type<weighted_sum<Traits>>("weighted_sum");
  factory.template register_type<property_console_logger<Traits>>("property_console_logger");
  factory.template register_type<von_mises_state_function<Traits>>("von_mises_state_function");
  factory.template register_type<strain_threshold_yield<Traits>>("strain_threshold_yield");
  factory.template register_type<exponential_damage_law<Traits>>("exponential_damage_law");
  factory.template register_type<isotropic_damage<Traits>>("isotropic_damage");
  // A single registered type covers every unknown combination — the layout is
  // dispatched from the "unknowns" parameter, not baked into the template.
  factory.template register_type<vector_newton<Traits>>("vector_newton");
}

} // namespace numsim::materials

#endif // DEFAULT_MATERIALS_H
