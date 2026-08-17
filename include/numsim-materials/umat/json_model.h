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
/// A builder written as a lambda forces a rebuild of the shared library for
/// every new material, which defeats the point of the deck driving the model.
/// This turns a JSON document into the same builder the registry already takes,
/// so a new material means editing a config file.
///
/// The document is the one io/json_material_factory already understands, plus
/// an optional "constants" array binding the deck's *USER MATERIAL constants to
/// named parameters. It is spelled "constants" rather than "props" because in
/// this library a PROPERTY is a graph node — reusing that word for the deck's
/// numbers would name two unrelated things the same. "constants" is also what
/// the deck itself calls them (*USER MATERIAL, CONSTANTS=).
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
/// "material::parameter" — the same qualified-name syntax the rest of the
/// library uses for wiring ("time::state"), parsed by the same
/// connection_source::parse. Note the right-hand side is a PARAMETER here, not
/// a property.
///
/// Values written in the document are placeholders for anything listed there.
/// Pairing this with constant_scalar means a deck constant enters as a graph
/// property, so consumers are ordered after it and follow it — see
/// materials/isotropic_tangent.h.
///
/// An entry naming a props_scalar binds the SLOT rather than the value: that
/// material is told which constant it owns and reads the number from the host
/// on every call, instead of having it baked into the graph. The document is
/// otherwise unchanged — swap constant_scalar for props_scalar and nothing
/// downstream notices — which is what a host with genuinely varying constants
/// needs, CalculiX interpolating them by temperature being the case in hand.
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

/// Register the materials a document may name, once per Traits.
///
/// Hoisted out of the builder so it also runs at REGISTRATION time: the
/// binding targets are checked against each material's declared parameters,
/// and that needs a populated factory. Registering types is idempotent.
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
/// Checking only that the MATERIAL exists leaves the other half of the target
/// unvalidated, and nlohmann::json CREATES a missing key rather than failing —
/// so a misspelled parameter is written to a key nothing reads while the real
/// one keeps the document's placeholder. The result is a wrong-but-plausible
/// modulus behind a stderr warning, in a job that reports no error at all.
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
  // A type the factory does not know is caught when the graph is built. Not
  // failing here keeps a document free to name a material the caller registers
  // after this one.
  if (!factory.contains(type)) return;

  const auto wanted = bound_parameter(material, binding);
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

  // An unrecognised top-level key is a setup fault, not something to ignore.
  // A document still spelling the binding array "props" would otherwise be
  // accepted with every constant silently unbound, leaving the placeholders in
  // the document as the material's moduli. json_to_parameters already warns
  // about unknown keys per material; this is the same check one level up.
  for (const auto& [key, value] : parsed.items()) {
    if (key == "materials" || key == "constants") continue;
    throw fatal_error(
        "json_model: unrecognised top-level key \"" + key +
        "\"; the document takes \"materials\" and \"constants\"" +
        (key == "props" ? " (the binding array is named \"constants\", since "
                          "\"property\" already means a graph node here)"
                        : ""));
  }

  // Validate the bindings now rather than on first use, so a typo is reported
  // when the model is registered rather than mid-analysis. BOTH halves of the
  // target: a check that stops at the material name is the more dangerous kind,
  // because it reads as though the whole thing were verified.
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

      // One host constant per target. Repeating one makes the later slot
      // overwrite the earlier, so an earlier constant is dropped and whatever
      // it should have bound keeps its placeholder — silently.
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

    // Substitute into a copy, so the registered document stays a template and
    // a second thread building the same model is unaffected.
    nlohmann::json doc = parsed;
    for (std::size_t i = 0; i < bindings.size(); ++i)
      for (auto& material : doc["materials"]) {
        if (!material.contains("name") ||
            material["name"].get<std::string>() != bindings[i].material)
          continue;
        // Same binding, two binding TIMES. A props_scalar is told which slot it
        // owns and reads the number on every call; anything else has the number
        // copied in now and keeps it for the life of the graph. The position in
        // "constants" is the slot either way, so the document says the same
        // thing and only the material type decides when it is read.
        if (material.value("type", std::string{}) == "props_scalar")
          material["index"] = i;
        else
          material[bindings[i].property] = props[i];
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
