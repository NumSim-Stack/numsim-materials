#ifndef JSON_MATERIAL_FACTORY_H
#define JSON_MATERIAL_FACTORY_H

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/object_store.h"
#include "numsim-materials/io/json_parameter_converter.h"

namespace numsim::materials {

namespace detail {

/// Edit distance, capped: we only care whether two names are CLOSE, and a
/// full matrix for every registered type on every failure is wasted work.
inline std::size_t edit_distance(std::string_view a, std::string_view b,
                                 std::size_t cap) {
  if (a.size() > b.size()) std::swap(a, b);
  if (b.size() - a.size() > cap) return cap + 1;

  std::vector<std::size_t> prev(a.size() + 1), curr(a.size() + 1);
  for (std::size_t i = 0; i <= a.size(); ++i) prev[i] = i;

  for (std::size_t j = 1; j <= b.size(); ++j) {
    curr[0] = j;
    for (std::size_t i = 1; i <= a.size(); ++i)
      curr[i] = std::min({prev[i] + 1, curr[i - 1] + 1,
                          prev[i - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
    prev = curr;
  }
  return prev[a.size()];
}

/// "Did you mean 'x'?", or the full list when nothing is close.
///
/// A misspelt type used to fail with the name echoed back and nothing else --
/// no candidates, no list -- which leaves a reader guessing at spelling for a
/// registry they cannot see from the document.
inline std::string suggest_type(const std::vector<std::string>& known,
                                const std::string& given) {
  if (known.empty())
    return "No material types are registered -- did you call "
           "register_default_materials<policy>()?";

  // A third of the name may differ before a suggestion stops being useful.
  const std::size_t cap = std::max<std::size_t>(2, given.size() / 3);
  const std::string* best = nullptr;
  std::size_t best_d = cap + 1;
  for (const auto& k : known)
    if (auto d = edit_distance(k, given, cap); d < best_d) {
      best_d = d;
      best = &k;
    }

  if (best) return "Did you mean '" + *best + "'?";

  std::string all = "Registered types are:";
  auto sorted = known;
  std::sort(sorted.begin(), sorted.end());
  for (const auto& k : sorted) all += " " + k;
  return all + ".";
}

} // namespace detail

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
  using parameter_handler = typename Traits::ParameterHandler;
  using adapter = json_adapter<JsonType>;
  using factory_type = typename object_store<Traits>::factory_type;

  auto& factory = factory_type::instance();

  // What this entry calls itself, if anything, so every error below can say
  // WHICH entry in the document it is about. A deck with twenty materials
  // otherwise reports a failure with nothing to locate it by.
  const std::string where =
      adapter::contains(json, "name")
          ? " (material \"" +
                adapter::template get<std::string>(adapter::at(json, "name")) +
                "\")"
          : " (entry has no \"name\" either)";

  // Reading "type" straight out used to surface nlohmann's own
  // "key 'type' not found" exception, from a library the user never named,
  // with no indication of which entry was at fault.
  if (!adapter::contains(json, "type"))
    throw std::runtime_error(
        "create_from_json: no \"type\" key" + where +
        ". Every material entry needs a \"type\" naming a registered "
        "material.");

  auto type_name = adapter::template get<std::string>(adapter::at(json, "type"));

  // An unknown type is almost always a typo, and the registry knows the whole
  // list -- it just never offered it. Reporting the candidates turns a dead
  // end into a one-line fix.
  if (!factory.contains(type_name))
    throw std::runtime_error("create_from_json: unknown material type '" +
                             type_name + "'" + where + ". " +
                             detail::suggest_type(factory.registered_types(),
                                                  type_name));

  auto schema = factory.schema(type_name);

  parameter_handler params;
  json_to_parameters(json, schema, params);

  return ctx.create(type_name, params);
}

} // namespace numsim::materials

#endif // JSON_MATERIAL_FACTORY_H
