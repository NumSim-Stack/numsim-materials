#ifndef NUMSIM_MATERIALS_UMAT_JSON_MODEL_H
#define NUMSIM_MATERIALS_UMAT_JSON_MODEL_H

#include <algorithm>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "numsim-materials/default_materials.h"
#include "numsim-materials/io/json_material_factory.h"
#include "numsim-materials/core/input_types.h"
#include "numsim-materials/umat/errors.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/umat_interface.h"

/// Define a UMAT model from JSON rather than from compiled C++.
///
/// A builder written as a lambda means rebuilding the shared library for every
/// new material. This turns a document into the same builder the registry
/// takes, so a new material is a config edit.
///
/// The document is io/json_material_factory's, plus an optional "constants"
/// array binding the deck's *USER MATERIAL constants to named parameters.
/// Spelled "constants", not "props": a PROPERTY here is a graph node, and it is
/// what the deck calls them (*USER MATERIAL, CONSTANTS=).
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
/// PROPS[i] replaces the parameter named by constants[i], written
/// "material::parameter" — the library's existing qualified-name syntax
/// ("time::state"), parsed by connection_source::parse. The right-hand side is
/// a PARAMETER, not a property.
///
/// Values in the document are placeholders for anything listed there. Paired
/// with constant_scalar, a deck constant enters as a graph property, so
/// consumers are ordered after it — see materials/isotropic_tangent.h.
///
/// An entry naming a props_scalar binds the SLOT instead: that material reads
/// the number from the host on every call rather than having it baked in. Swap
/// the type and nothing else in the document changes.
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

/// What a binding writes into the document. One function for both the
/// validation and the substitution below — checking one parameter name and
/// writing another is how a target ends up half-verified.
struct constant_binding_target {
  std::string parameter;
  /// props_scalar is told its SLOT and reads the number itself each call;
  /// everything else has the number written in now.
  bool writes_slot{false};
};

inline constant_binding_target bound_parameter(
    const nlohmann::json& material, const connection_source& binding) {
  if (material.value("type", std::string{}) == "props_scalar") {
    // The target names the property the constant arrives on — "value", as for
    // constant_scalar — so swapping the type does not force "constants" to be
    // rewritten. The parameter written ("index") is therefore not what the
    // document says.
    if (binding.property != "value")
      throw fatal_error(
          "json_model: a props_scalar target names the property it publishes, "
          "which is \"value\" — got \"" + binding.property +
          "\"; the slot comes from the entry's position in \"constants\"");
    return {"index", true};
  }
  return {binding.property, false};
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

  const auto wanted = bound_parameter(material, binding).parameter;
  std::vector<std::string> declared;
  for (const auto& [key, unused] : factory.schema(type)) declared.push_back(key);
  if (std::find(declared.begin(), declared.end(), wanted) != declared.end())
    return;

  std::string known;
  std::sort(declared.begin(), declared.end());
  for (const auto& key : declared) known += (known.empty() ? "" : ", ") + key;
  throw fatal_error("json_model: constants entry '" + target +
                    "' names parameter '" + wanted + "', which " + type +
                    " does not declare — it takes: " + known);
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
      for (auto& material : doc["materials"]) {
        if (!material.contains("name") ||
            material["name"].get<std::string>() != bindings[i].material)
          continue;
        // Resolved by the function that validated the target, so the two
        // cannot drift.
        const auto target = bound_parameter(material, bindings[i]);
        material[target.parameter] =
            target.writes_slot ? nlohmann::json(i) : nlohmann::json(props[i]);
      }

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

#endif  // NUMSIM_MATERIALS_UMAT_JSON_MODEL_H
