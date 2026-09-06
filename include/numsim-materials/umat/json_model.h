#ifndef JSON_MODEL_H
#define JSON_MODEL_H

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <span>
#include <string>
#include <typeindex>
#include <vector>

#include <nlohmann/json.hpp>
#include "numsim-materials/default_materials.h"
#include "numsim-materials/io/json_material_factory.h"
#include "numsim-materials/core/input_types.h"
#include "numsim-materials/umat/errors.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/umat_interface.h"

/// Define a UMAT model from JSON rather than compiled C++, so a new material is
/// a config edit and not a rebuild of the shared library.
///
/// io/json_material_factory's document, plus an optional "constants" array
/// binding the deck's *USER MATERIAL constants to named parameters:
///
///     {
///       "materials": [
///         {"type": "external_strain_source", "name": "strain_in"},
///         {"type": "constant_scalar", "name": "K", "value": 0},
///         {"type": "constant_scalar", "name": "G", "value": 0},
///         {"type": "isotropic_tangent", "name": "stiffness",
///          "K_source": "K", "G_source": "G"},
///         {"type": "linear_stress", "name": "elastic",
///          "tangent_source": "stiffness", "strain_source": "strain_in"}
///       ],
///       "constants": ["K::value", "G::value"]
///     }
///
/// PROPS[i] replaces the PARAMETER named by constants[i], in the library's own
/// qualified-name syntax ("time::state") and parsed by the same
/// connection_source::parse. Values in the document are placeholders for
/// anything listed there.
///
/// Named "constants", not "props": a PROPERTY here is a graph node, and it is
/// what the deck calls them (*USER MATERIAL, CONSTANTS=).
namespace numsim::materials::umat {

/// Host-driven source materials. Kept out of register_default_materials() so
/// the core defaults carry no dependency on the UMAT layer.
template <typename Traits>
void register_umat_materials() {
  auto& factory = material_factory<Traits>::instance();
  factory.template register_type<external_strain_source<Traits>>(
      "external_strain_source");
  factory.template register_type<external_scalar_source<Traits>>(
      "external_scalar_source");
}

/// The materials a document may name, registered once per Traits.
///
/// Runs at REGISTRATION time too: checking targets against a material's
/// declared parameters needs a populated factory. Idempotent.
template <typename Traits>
void ensure_materials_registered() {
  static std::once_flag once;
  std::call_once(once, [] {
    register_default_materials<Traits>();
    register_umat_materials<Traits>();
  });
}

/// The parameter a binding will actually write.
///
/// Its own function because it is the single place that has to stay in step
/// with the substitution loop below — validating one name and writing another
/// is how a target ends up half-checked.
inline std::string bound_parameter(const nlohmann::json& /*material*/,
                                   const connection_source& binding) {
  return binding.property;
}

/// Reject a target naming a parameter the material does not declare.
///
/// nlohmann::json CREATES a missing key rather than failing, so a misspelled
/// parameter is written where nothing reads it while the real one keeps its
/// placeholder — a wrong-but-plausible modulus behind a stderr warning.
template <typename Traits>
void require_declared_parameter(const nlohmann::json& material,
                                const connection_source& binding,
                                const std::string& target) {
  if (!material.contains("type") || !material["type"].is_string())
    throw fatal_error("json_model: material '" + binding.material +
                      "' has no \"type\", so \"" + target +
                      "\" cannot be checked against its parameters");

  const auto type = material["type"].get<std::string>();
  auto& factory = object_store<Traits>::factory_type::instance();
  // An unknown type is caught at build time; not failing here keeps a document
  // free to name a material the caller registers later.
  if (!factory.contains(type)) return;

  // Declared AND numeric. Every material declares "name", and most declare
  // *_source strings, so checking mere existence accepts targets that can only
  // fail later — with a JSON type error rather than anything about decks.
  const auto wanted = bound_parameter(material, binding);
  const auto schema = factory.schema(type);
  std::vector<std::string> numeric;
  bool declared = false, wanted_is_numeric = false;
  for (const auto& [key, param] : schema) {
    const auto tid = param->type_id();
    const bool is_num = tid == std::type_index(typeid(double)) ||
                        tid == std::type_index(typeid(float)) ||
                        tid == std::type_index(typeid(int)) ||
                        tid == std::type_index(typeid(std::size_t));
    if (is_num) numeric.push_back(key);
    if (key == wanted) {
      declared = true;
      wanted_is_numeric = is_num;
    }
  }
  if (declared && wanted_is_numeric) return;

  std::sort(numeric.begin(), numeric.end());
  std::string known;
  for (const auto& key : numeric) known += (known.empty() ? "" : ", ") + key;
  throw fatal_error(
      "json_model: constants entry '" + target + "' names " +
      (declared
           ? "'" + wanted + "', which " + type + " does not take as a number"
           : "parameter '" + wanted + "', which " + type + " does not declare") +
      " — a host constant can bind: " + (known.empty() ? "(nothing)" : known));
}

/// Parse a "material::parameter" target with the library's existing splitter,
/// so this does not invent a second syntax for a qualified name.
inline connection_source parse_constant_target(const std::string& target) {
  try {
    auto src = connection_source::parse(target);
    if (src.material.empty() || src.property.empty()) throw std::invalid_argument("");
    return src;
  } catch (const std::invalid_argument&) {
    throw fatal_error(
        "json_model: constants entry '" + target +
        "' must be written \"material::parameter\"");
  }
}

/// Build a registry builder from a JSON document.
///
/// Parsing happens once, here; the builder only substitutes and creates. A
/// malformed document is a setup fault, so it raises fatal_error rather than
/// asking for a smaller increment.
template <typename Traits>
typename umat_registry<Traits>::builder make_json_builder(
    const std::string& document) {
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(document);
  } catch (const std::exception& e) {
    throw fatal_error(std::string("json_model: cannot parse the model "
                                  "document: ") +
                      e.what());
  }
  if (!parsed.contains("materials") || !parsed["materials"].is_array())
    throw fatal_error("json_model: the document needs a \"materials\" array");

  // An unrecognised key is a setup fault: a document still spelling the array
  // "props" would be accepted with every constant unbound, leaving the
  // placeholders as the moduli. Same check json_to_parameters does per
  // material, one level up.
  for (const auto& [key, value] : parsed.items()) {
    if (key == "materials" || key == "constants") continue;
    throw fatal_error(
        "json_model: unrecognised top-level key \"" + key +
        "\"; the document takes \"materials\" and \"constants\"" +
        (key == "props" ? " (the binding array is named \"constants\", since "
                          "\"property\" already means a graph node here)"
                        : ""));
  }

  // Validated at registration rather than mid-analysis, and BOTH halves of the
  // target — a check stopping at the material name reads as though the whole
  // thing were verified.
  std::vector<connection_source> bindings;
  if (parsed.contains("constants")) {
    if (!parsed["constants"].is_array())
      throw fatal_error("json_model: \"constants\" must be an array of "
                        "\"material::parameter\" strings");
    ensure_materials_registered<Traits>();
    std::vector<std::string> seen;
    for (const auto& entry : parsed["constants"]) {
      if (!entry.is_string())
        throw fatal_error(
            "json_model: every \"constants\" entry must be a string");
      const auto target = entry.get<std::string>();

      // One constant per target: a repeat overwrites, dropping the earlier
      // constant and leaving whatever it should have bound at its placeholder.
      if (std::find(seen.begin(), seen.end(), target) != seen.end())
        throw fatal_error("json_model: constants entry '" + target +
                          "' appears twice; each host constant binds one "
                          "target, and a repeat silently drops the earlier one");
      seen.push_back(target);

      auto binding = parse_constant_target(target);
      const nlohmann::json* owner = nullptr;
      for (const auto& m : parsed["materials"])
        if (m.contains("name") &&
            m["name"].get<std::string>() == binding.material)
          owner = &m;
      if (!owner)
        throw fatal_error("json_model: constants entry targets material '" +
                          binding.material +
                          "', which the document does not define");
      require_declared_parameter<Traits>(*owner, binding, target);
      bindings.push_back(std::move(binding));
    }
  }

  return [parsed, bindings](material_context<Traits>& ctx,
                            std::span<const double> props) {
    ensure_materials_registered<Traits>();

    if (props.size() < bindings.size())
      throw fatal_error(
          "json_model: the document binds " + std::to_string(bindings.size()) +
          " material constants but the deck supplied " +
          std::to_string(props.size()) +
          " — check *USER MATERIAL, CONSTANTS=");

    // Into a copy, so the registered document stays a template.
    nlohmann::json doc = parsed;
    for (std::size_t i = 0; i < bindings.size(); ++i)
      for (auto& material : doc["materials"])
        if (material.contains("name") &&
            material["name"].get<std::string>() == bindings[i].material)
          material[bindings[i].property] = props[i];

    for (const auto& material : doc["materials"])
      create_from_json<Traits>(ctx, material);
    ctx.finalize();
  };
}

/// Register a model defined by a JSON document.
template <typename Traits>
void register_json_model(
    std::string cmname, const std::string& document,
    typename umat_registry<Traits>::config cfg,
    typename plane_stress_evaluator<Traits>::options ps_opts = {}) {
  umat_registry<Traits>::instance().register_model(
      std::move(cmname), make_json_builder<Traits>(document), std::move(cfg),
      ps_opts);
}

}  // namespace numsim::materials::umat

#endif  // JSON_MODEL_H
