#ifndef JSON_MATERIAL_FACTORY_H
#define JSON_MATERIAL_FACTORY_H

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/object_store.h"
#include "numsim-materials/io/json_parameter_converter.h"
#include "numsim-materials/io/object_readers.h"

namespace numsim::materials {

namespace detail {

/// Edit distance, stopping with cap + 1 as soon as a row cannot get below the cap.
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
    if (*std::min_element(curr.begin(), curr.end()) > cap) return cap + 1;
    prev = curr;
  }
  return prev[a.size()];
}

/// "Did you mean 'x'?" for the closest registered type, or the full list when nothing is close.
inline std::string suggest_type(const std::vector<std::string>& known,
                                const std::string& given) {
  if (known.empty())
    return "No material types are registered -- did you call "
           "register_default_materials<policy>()?";

  const std::size_t cap = std::max<std::size_t>(2, given.size() / 3); // a third of the name may differ
  const auto distance = [&](const std::string& k) { return edit_distance(k, given, cap); };
  if (const auto best = std::ranges::min_element(known, {}, distance); distance(*best) <= cap)
    return "Did you mean '" + *best + "'?";

  auto sorted = known;
  std::ranges::sort(sorted);
  return std::accumulate(sorted.begin(), sorted.end(), std::string{"Registered types are:"},
                         [](std::string all, const std::string& k) { return std::move(all) + " " + k; }) + ".";
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

  // which entry of the document an error is about: every message below carries it
  std::string where{" (entry has no \"name\" either)"};
  if (adapter::contains(json, "name")) {
    const JsonType name_value = adapter::at(json, "name"); // by value: GCC 14 flags the reference as dangling
    where = adapter::is_string(name_value)
                ? " (material \"" + adapter::template get<std::string>(name_value) + "\")"
                : " (entry whose \"name\" is not a string)";
  }
  const auto located = [&where](const std::exception& e) {
    std::string msg{e.what()};
    return msg.find(where) == std::string::npos ? msg + where : msg;
  };

  if (!adapter::contains(json, "type"))
    throw std::runtime_error(
        "create_from_json: no \"type\" key" + where +
        ". Every material entry needs a \"type\" naming a registered "
        "material.");

  auto type_name = adapter::template get<std::string>(adapter::at(json, "type"));

  if (!factory.contains(type_name)) // almost always a typo: offer the closest registered name
    throw std::runtime_error("create_from_json: unknown material type '" +
                             type_name + "'" + where + ". " +
                             detail::suggest_type(factory.registered_types(),
                                                  type_name));

  auto schema = factory.schema(type_name);

  parameter_handler params;
  try { // reader failures and the material's own checks name the entry too
    auto readers = make_default_json_registry<JsonType>();
    add_object_readers<Traits>(readers); // "kernel", "shapes": objects with their own schema
    json_to_parameters(json, schema, params, readers);
    return ctx.create(type_name, params);
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(located(e));
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(located(e));
  }
}

} // namespace numsim::materials

#endif // JSON_MATERIAL_FACTORY_H
