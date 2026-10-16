#ifndef NUMSIM_MATERIALS_REPORT_REPORT_JSON_H
#define NUMSIM_MATERIALS_REPORT_REPORT_JSON_H

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/report/load_path.h"
#include "numsim-materials/umat/load_control.h"
#include "numsim-materials/umat/material_point_evaluator.h"

/// Parsing of report documents into value types; nothing here runs a material.
namespace numsim::materials::report {

using umat::load_control;

template <typename T>
struct load_case {
  using tensor2 = tmech::tensor<T, 3, 2>;
  std::string name;
  load_control<T> control;
  std::vector<tensor2> points; ///< strain states the path interpolates (already scaled)
  tensor2 stress;              ///< stress target per unit load factor
  load_path<T> path;
  int tangent_every{1};        ///< finite-difference tangent check every n-th step; 0 = never

  /// Prescribed strain at a step: linear between the points of its segment.
  [[nodiscard]] tensor2 strain_at(load_step<T> const &s) const {
    return tensor2{points[s.segment] + (points[s.segment + 1] - points[s.segment]) * s.fraction};
  }
};

template <typename T>
tmech::tensor<T, 3, 2> parse_symmetric(nlohmann::json const &j) {
  auto const v = j.get<std::vector<T>>();
  if (v.size() != 6)
    throw std::invalid_argument("expected six components [11, 22, 33, 12, 13, 23]");
  return symmetric_tensor<T>(std::span<T const, 6>{v.data(), 6});
}

template <typename T>
load_control<T> parse_control(nlohmann::json const &c) {
  auto const kind = c.value("control", std::string{"strain"});
  if (kind == "strain")
    return load_control<T>::strain();
  if (kind == "uniaxial_stress")
    return load_control<T>::uniaxial_stress();
  if (kind == "mixed") {
    auto const names = c.at("stress_components").get<std::vector<std::string>>();
    std::vector<std::string_view> const views(names.begin(), names.end());
    return load_control<T>::mixed(std::span<std::string_view const>{views});
  }
  throw std::invalid_argument("unknown control '" + kind + "' (strain, uniaxial_stress, mixed)");
}

/// Proportional: "direction" times the factors of "path". Non-proportional: "points", strain states
/// in order (the load factor then counts segments). Both are scaled by "scale".
template <typename T>
load_case<T> parse_case(nlohmann::json const &c) {
  using tensor2 = tmech::tensor<T, 3, 2>;
  auto const scale{c.value("scale", T{1})};
  std::vector<tensor2> points;
  std::vector<T> factors;
  if (c.contains("points") == c.contains("direction"))
    throw std::invalid_argument("case '" + c.value("name", std::string{}) + "': give either direction + path or points");
  if (c.contains("points")) {
    if (c.contains("path"))
      throw std::invalid_argument("case '" + c.value("name", std::string{}) + "': points and path exclude each other");
    for (auto const &pt : c["points"]) {
      points.push_back(tensor2{scale * parse_symmetric<T>(pt)});
      factors.push_back(T(factors.size()));
    }
  } else {
    auto const direction{parse_symmetric<T>(c.at("direction"))};
    factors = c.at("path").get<std::vector<T>>();
    for (auto const f : factors)
      points.push_back(tensor2{scale * f * direction});
  }
  return {c.at("name").get<std::string>(), parse_control<T>(c), std::move(points),
          c.contains("stress") ? parse_symmetric<T>(c["stress"]) : tensor2{},
          load_path<T>{std::move(factors), c.value("steps", 50), c.contains("points") ? T{1} : scale, c.value("dt", T{1})},
          c.value("tangent_every", 1)};
}

/// A case run again with several step counts (document key "studies").
struct refinement_study {
  std::string name;
  nlohmann::json case_spec; ///< the case as written, "steps" to be replaced
  std::vector<int> steps;
};

/// `spec` with its step count replaced.
inline nlohmann::json with_steps(nlohmann::json spec, int steps) {
  spec["steps"] = steps;
  return spec;
}

inline std::vector<refinement_study> parse_studies(nlohmann::json const &doc) {
  std::vector<refinement_study> out;
  for (auto const &s : doc.value("studies", nlohmann::json::array())) {
    auto const case_name = s.at("case").get<std::string>();
    auto const &cases{doc.at("cases")};
    auto const it{std::find_if(cases.begin(), cases.end(), [&](auto const &c) { return c.at("name") == case_name; })};
    if (it == cases.end())
      throw std::invalid_argument("study '" + s.at("name").get<std::string>() + "': no case '" + case_name + "'");
    out.push_back({s.at("name").get<std::string>(), *it, s.at("steps").get<std::vector<int>>()});
  }
  return out;
}

/// "point": the evaluator configuration (strain_source, stress_source, optional tangent_source, time_source).
template <typename Traits>
typename umat::material_point_evaluator<Traits>::config parse_point(nlohmann::json const &doc) {
  auto const &p{doc.at("point")};
  typename umat::material_point_evaluator<Traits>::config cfg;
  cfg.strain_source = p.at("strain_source").template get<std::string>();
  cfg.stress_source = p.at("stress_source").template get<std::string>();
  if (p.contains("tangent_source"))
    cfg.tangent_source = p["tangent_source"].template get<std::string>();
  cfg.time_source = p.value("time_source", std::string{});
  return cfg;
}

/// Report columns as (column, "material::property") pairs: the point's strain and stress, then
/// "record" (nlohmann::json orders those by column name).
inline std::vector<std::pair<std::string, std::string>> parse_columns(nlohmann::json const &doc) {
  auto const &p{doc.at("point")};
  std::vector<std::pair<std::string, std::string>> out{{"strain", p.at("strain_source").get<std::string>() + "::strain"},
                                                       {"stress", p.at("stress_source").get<std::string>() + "::stress"}};
  auto const record = doc.value("record", nlohmann::json::object());
  for (auto const &[name, source] : record.items())
    out.emplace_back(name, source.get<std::string>());
  return out;
}

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_REPORT_JSON_H
