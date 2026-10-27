#ifndef NUMSIM_MATERIALS_IO_OBJECT_READERS_H
#define NUMSIM_MATERIALS_IO_OBJECT_READERS_H

#include <any>
#include <map>
#include <stdexcept>
#include <string>

#include "numsim-materials/core/object_spec.h"
#include "numsim-materials/io/json_parameter_converter.h"
#include "numsim-materials/mean_field/kernel_registry.h"
#include "numsim-materials/mean_field/shapes.h"

namespace numsim::materials {

namespace detail {

/// {"type": ..., ...} into an object_spec, its parameters read against the schema of that type.
template <typename ParameterHandler, typename JsonType, typename Registry>
object_spec<ParameterHandler> read_object(JsonType const &j, std::string const &key, Registry const &types,
                                          json_reader_registry<JsonType> const &readers) {
  using adapter = json_adapter<JsonType>;
  if (!adapter::is_object(j) || !adapter::contains(j, "type"))
    throw std::invalid_argument("'" + key + "' is an object {\"type\": ..., ...}");
  object_spec<ParameterHandler> spec{adapter::template get<std::string>(adapter::at(j, "type")), {}};
  json_to_parameters(j, types.schema(spec.type), spec.parameters, readers);
  return spec;
}

} // namespace detail

/// Readers for the configured objects of a material's parameters: "kernel" (a mean-field kernel) and
/// "shapes" ({phase: shape}). The readers keep a reference to `readers` for the nested parameters.
template <typename Traits, typename JsonType>
void add_object_readers(json_reader_registry<JsonType> &readers) {
  using ph = typename Traits::ParameterHandler;
  using spec = object_spec<ph>;
  readers.template add_for_key<spec>("kernel", [&readers](JsonType const &j, std::string const &key) -> std::any {
    return detail::read_object<ph>(j, key, mean_field::kernel_registry<Traits>::instance(), readers);
  });
  readers.template add_for_key<std::map<std::string, spec>>("shapes", [&readers](JsonType const &j, std::string const &key) -> std::any {
    using adapter = json_adapter<JsonType>;
    if (!adapter::is_object(j))
      throw std::invalid_argument("'" + key + "' is an object {phase: {\"type\": ..., ...}}");
    std::map<std::string, spec> shapes;
    adapter::for_each_key(j, [&](std::string const &phase) {
      shapes.emplace(phase, detail::read_object<ph>(adapter::at(j, phase), key + "." + phase, mean_field::shape_registry<Traits>(), readers));
    });
    return shapes;
  });
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_IO_OBJECT_READERS_H
