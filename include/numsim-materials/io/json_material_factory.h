#ifndef NUMSIM_MATERIALS_JSON_MATERIAL_FACTORY_H
#define NUMSIM_MATERIALS_JSON_MATERIAL_FACTORY_H

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/object_store.h"
#include "numsim-materials/io/json_parameter_converter.h"

namespace numsim::materials {

/// Create a material from a JSON object using the factory registry.
///
/// Reads "type" from JSON, looks up the schema from the factory,
/// converts JSON parameters via the visitor pattern, and creates
/// the material in the given context.
///
/// Usage:
///   register_default_materials<policy>();
///   for (auto& mat : json["materials"])
///     create_from_json(ctx, mat);
///   ctx.finalize();
template<typename Traits, typename JsonType>
material_interface<Traits>& create_from_json(
    material_context<Traits>& ctx,
    const JsonType& json)
{
  using input_controller = typename Traits::InputParameterController;
  using parameter_handler = typename Traits::ParameterHandler;
  using adapter = json_adapter<JsonType>;
  using factory_type = typename object_store<Traits>::factory_type;

  auto type_name = adapter::template get<std::string>(adapter::at(json, "type"));
  auto& factory = factory_type::instance();

  auto schema = factory.schema(type_name);

  parameter_handler params;
  json_to_parameters(json, schema, params);

  return ctx.create(type_name, params);
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_JSON_MATERIAL_FACTORY_H
