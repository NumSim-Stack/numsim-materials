#ifndef NUMSIM_MATERIALS_ORIENTATION_PHASES_H
#define NUMSIM_MATERIALS_ORIENTATION_PHASES_H

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/lebedev.h"
#include "numsim-materials/core/orientation.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/io/json_substitute.h"

namespace numsim::materials {

/// The deck entry "orientation_phases" as plain entries: fibres at their local strains, one fibre phase
/// per orientation node (half sphere), each with its own copy of the "per_direction" sub-deck ("{q}" -> q).
///
/// Emits "<name>_fractions" (scalar_split of the fibre fraction by the node weights); per direction
/// "<name>_S_<q>" (the kernel's "eshelby" template along n_q, dilute and mori_tanaka), "<name>_<q>" (phase
/// from the "phase" template) and the sub-deck; then "<name>_strain" (mean_field_strain, the matrix first)
/// and "<name>" (mean_field_composite_local_linear, or _nonlinear with "localisation": "nonlinear"). A phase
/// material reads "strain_<name>_<q>" of "<name>_strain".
///
/// Keys: "name", "kernel" {"type", "eshelby" (dilute, mori_tanaka) or "shape" and optional "matrix_shape"
/// (self_consistent), the kernel's own data}, "strain_source", "matrix" (its phase node), "matrix_stress",
/// "fraction_source", "orientation" [a11, a22, a33, a12, a13, a23], "per_direction", "stress" ("{q}"
/// template); optional "precision" (11), "phase", "fraction_property", "matrix_tangent" and "tangent"
/// (default the phase nodes' tangents), "localisation" (linear | nonlinear: the fibres localise with their
/// current tangents, read by the "phase" template over a local edge), "matrix_used_tangent", "tolerance" and
/// "max_passes" (nonlinear: the composite's fixed point).
/// Sources are "material::property".
template <typename JsonType>
std::vector<JsonType> expand_orientation_phases(JsonType const &e) {
  auto const require = [&](char const *key) -> JsonType const & {
    if (!e.contains(key))
      throw std::invalid_argument(std::string{"orientation_phases: \""} + key + "\" is missing");
    return e.at(key);
  };
  auto const name{require("name").template get<std::string>()};
  auto const matrix{require("matrix").template get<std::string>()};
  JsonType const &kernel_entry{require("kernel")};
  if (!kernel_entry.is_object() || !kernel_entry.contains("type"))
    throw std::invalid_argument("orientation_phases: kernel is an object {\"type\": voigt | reuss | hill | dilute | mori_tanaka | self_consistent, ...}");
  auto const kernel_type{kernel_entry.at("type").template get<std::string>()};
  bool const eshelby{kernel_type == "dilute" || kernel_type == "mori_tanaka"}, self_consistent{kernel_type == "self_consistent"};
  if (eshelby && !kernel_entry.contains("eshelby"))
    throw std::invalid_argument("orientation_phases: kernel " + kernel_type + " needs \"eshelby\", the shape template");
  if (self_consistent && !kernel_entry.contains("shape"))
    throw std::invalid_argument("orientation_phases: kernel self_consistent needs \"shape\", the fibre shape template");
  auto const a{require("orientation").template get<std::vector<double>>()};
  if (a.size() != 6)
    throw std::invalid_argument("orientation_phases: orientation takes [a11, a22, a33, a12, a13, a23]");
  int const precision{e.contains("precision") ? e.at("precision").template get<int>() : 11};
  if (precision < 0)
    throw std::invalid_argument("orientation_phases: precision must be a Lebedev degree");
  (void)lebedev::rule(static_cast<std::size_t>(precision)); // an aligned distribution never asks for it
  auto const localisation{e.contains("localisation") ? e.at("localisation").template get<std::string>() : std::string{"linear"}};
  if (localisation != "linear" && localisation != "nonlinear")
    throw std::invalid_argument("orientation_phases: unknown localisation '" + localisation + "' (linear, nonlinear)");
  bool const nonlinear{localisation == "nonlinear"};
  JsonType const phase_template = e.contains("phase") ? e.at("phase") : JsonType::object();
  if (nonlinear && phase_template.value("tangent_edge", std::string{}) != "local")
    throw std::invalid_argument("orientation_phases: a nonlinear localisation reads the fibre tangents over a local edge: the \"phase\" "
                                "template needs \"tangent_edge\": \"local\" and an \"initial_tangent_source\"");
  for (char const *key : {"matrix_used_tangent", "tolerance", "max_passes"})
    if (!nonlinear && e.contains(key))
      throw std::invalid_argument(std::string{"orientation_phases: "} + key + " goes with \"localisation\": \"nonlinear\"");
  JsonType const &deck{require("per_direction")};
  if (!deck.is_array() || deck.empty())
    throw std::invalid_argument("orientation_phases: per_direction lists the entries of one direction");
  auto const nodes{orientation::half_nodes(orientation::fit_acg(symmetric_tensor<double>(std::span<double const, 6>{a.data(), 6})),
                                           static_cast<std::size_t>(precision))};

  // the kernel as mean_field_strain reads it: the templates leave, self-consistent gets a shape per phase
  JsonType kernel = kernel_entry;
  for (char const *key : {"eshelby", "shape", "matrix_shape"})
    kernel.erase(key);
  if (self_consistent)
    kernel["shapes"][matrix] = kernel_entry.contains("matrix_shape") ? kernel_entry.at("matrix_shape") : JsonType{{"type", "sphere"}};

  auto const strain_name{name + "_strain"};
  auto const local = [&](std::string const &phase_name, std::string stress, std::string tangent) {
    return JsonType{{"concentration", strain_name + "::concentration_" + phase_name}, {"stress", std::move(stress)},
                    {"tangent", std::move(tangent)}, {"fraction", phase_name + "::volume_fraction"}};
  };
  JsonType phases = JsonType::array({matrix}), weights = JsonType::array();
  JsonType locals = JsonType::array({local(matrix, require("matrix_stress").template get<std::string>(),
                                           e.contains("matrix_tangent") ? e.at("matrix_tangent").template get<std::string>() : matrix + "::tangent")});
  if (e.contains("matrix_used_tangent"))
    locals[0]["used_tangent"] = e.at("matrix_used_tangent");
  std::vector<JsonType> out(1); // the fractions, once the weights are known
  for (std::size_t q{0}; q < nodes.size(); ++q) {
    auto const s{std::to_string(q)}, phase_name{name + "_" + s};
    auto const &[n, w]{nodes[q]};
    JsonType const direction = JsonType::array({n(0), n(1), n(2)});
    JsonType ph = substitute(phase_template, "{q}", s);
    ph["type"] = "phase";
    ph["name"] = phase_name;
    ph["fraction_source"] = name + "_fractions";
    ph["fraction_property"] = "value_" + s;
    if (eshelby) {
      JsonType shape = kernel_entry.at("eshelby");
      shape["name"] = name + "_S_" + s;
      shape["direction"] = direction;
      ph["eshelby_source"] = shape["name"];
      out.push_back(std::move(shape));
    }
    if (self_consistent) {
      kernel["shapes"][phase_name] = kernel_entry.at("shape");
      kernel["shapes"][phase_name]["direction"] = direction;
    }
    out.push_back(std::move(ph));
    for (auto const &entry : deck)
      out.push_back(substitute(entry, "{q}", s));
    weights.push_back(w);
    phases.push_back(phase_name);
    locals.push_back(local(phase_name, substitute(require("stress"), "{q}", s).template get<std::string>(),
                           e.contains("tangent") ? substitute(e.at("tangent"), "{q}", s).template get<std::string>() : phase_name + "::tangent"));
    if (nonlinear)
      locals.back()["used_tangent"] = phase_name + "::tangent";
  }
  out.front() = {{"type", "scalar_split"}, {"name", name + "_fractions"}, {"source", require("fraction_source")}, {"weights", weights}};
  if (e.contains("fraction_property"))
    out.front()["property"] = e.at("fraction_property");
  out.push_back({{"type", "mean_field_strain"}, {"name", strain_name}, {"strain_source", require("strain_source")}, {"phases", phases}, {"kernel", kernel}});
  out.push_back({{"type", nonlinear ? "mean_field_composite_local_nonlinear" : "mean_field_composite_local_linear"}, {"name", name}, {"phases", locals}});
  for (char const *key : {"tolerance", "max_passes"})
    if (e.contains(key))
      out.back()[key] = e.at(key);
  return out;
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_ORIENTATION_PHASES_H
