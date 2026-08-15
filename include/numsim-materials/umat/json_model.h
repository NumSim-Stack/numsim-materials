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
#include "numsim-materials/umat/errors.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/umat_interface.h"

/// Define a UMAT model from JSON rather than from compiled C++.
///
/// A builder written as a lambda forces a rebuild of the shared library for
/// every new material, which defeats the point of the deck driving the model.
/// This turns a JSON document into the same builder the registry already takes,
/// so a new material means editing a config file.
///
/// The document is the one io/json_material_factory already understands, plus
/// an optional "props" array binding the deck's *USER MATERIAL constants to
/// named parameters:
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
///       "props": ["K.value", "G.value"]
///     }
///
/// PROPS[i] replaces the parameter named by props[i], written "material.param",
/// before that material is created. Values in the document are therefore
/// placeholders for anything listed there. Pairing this with constant_scalar
/// means a deck constant enters as a graph property, so consumers are ordered
/// after it and follow it — see materials/isotropic_tangent.h.
namespace numsim::materials::umat {

/// Register the host-driven source materials with the runtime factory.
///
/// Kept here rather than in register_default_materials() so the core defaults
/// stay free of any dependency on the UMAT layer; these materials only mean
/// something when a host is driving the graph.
template <typename Traits>
void register_umat_materials() {
  auto& factory = material_factory<Traits>::instance();
  factory.template register_type<external_strain_source<Traits>>(
      "external_strain_source");
  factory.template register_type<external_scalar_source<Traits>>(
      "external_scalar_source");
}

/// Split "material.parameter". Both halves must be non-empty.
inline std::pair<std::string, std::string> split_props_target(
    const std::string& target) {
  const auto dot = target.find('.');
  if (dot == std::string::npos || dot == 0 || dot + 1 == target.size())
    throw fatal_error(
        "json_model: props entry '" + target +
        "' must be written \"material.parameter\"");
  return {target.substr(0, dot), target.substr(dot + 1)};
}

/// Build a registry builder from a JSON document.
///
/// Parsing happens once, here; the returned builder only substitutes PROPS and
/// creates. Any error in the document surfaces on the first UMAT call for the
/// material, as a fatal_error — a malformed config is a setup fault, not
/// something a smaller increment fixes.
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

  // Validate the props bindings now rather than on first use, so a typo is
  // reported when the model is registered rather than mid-analysis.
  std::vector<std::pair<std::string, std::string>> bindings;
  if (parsed.contains("props")) {
    if (!parsed["props"].is_array())
      throw fatal_error("json_model: \"props\" must be an array of "
                        "\"material.parameter\" strings");
    for (const auto& entry : parsed["props"]) {
      if (!entry.is_string())
        throw fatal_error("json_model: every \"props\" entry must be a string");
      auto binding = split_props_target(entry.get<std::string>());
      const bool known = std::any_of(
          parsed["materials"].begin(), parsed["materials"].end(),
          [&](const nlohmann::json& m) {
            return m.contains("name") &&
                   m["name"].get<std::string>() == binding.first;
          });
      if (!known)
        throw fatal_error("json_model: props entry targets material '" +
                          binding.first +
                          "', which the document does not define");
      bindings.push_back(std::move(binding));
    }
  }

  return [parsed, bindings](material_context<Traits>& ctx,
                            std::span<const double> props) {
    static std::once_flag once;
    std::call_once(once, [] {
      register_default_materials<Traits>();
      register_umat_materials<Traits>();
    });

    if (props.size() < bindings.size())
      throw fatal_error(
          "json_model: the document binds " + std::to_string(bindings.size()) +
          " material constants but the deck supplied " +
          std::to_string(props.size()) +
          " — check *USER MATERIAL, CONSTANTS=");

    // Substitute into a copy, so the registered document stays a template and
    // a second thread building the same model is unaffected.
    nlohmann::json doc = parsed;
    for (std::size_t i = 0; i < bindings.size(); ++i)
      for (auto& material : doc["materials"])
        if (material.contains("name") &&
            material["name"].get<std::string>() == bindings[i].first)
          material[bindings[i].second] = props[i];

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
