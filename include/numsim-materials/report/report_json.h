#ifndef NUMSIM_MATERIALS_REPORT_REPORT_JSON_H
#define NUMSIM_MATERIALS_REPORT_REPORT_JSON_H

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/report/load_path.h"
#include "numsim-materials/umat/load_control.h"

/// Parsing of report documents into value types; nothing here runs a material.
namespace numsim::materials::report {

using umat::load_control;

template <typename T>
struct load_case {
  using tensor2 = tmech::tensor<T, 3, 2>;
  std::string name;
  load_control<T> control;                     ///< small strain
  umat::deformation_control<T> finite_control; ///< the same components, for F / P
  std::vector<tensor2> points; ///< strain states (or F - I) the path interpolates (already scaled)
  tensor2 stress;              ///< stress target per unit load factor
  load_path<T> path;
  int tangent_every{1};        ///< finite-difference tangent check every n-th step; 0 = never
  bool finite_strain{false};   ///< which of the two controls was parsed

  /// Prescribed strain at a step: linear between the points of its segment.
  [[nodiscard]] tensor2 strain_at(load_step<T> const &s) const {
    return tensor2{points[s.segment] + (points[s.segment + 1] - points[s.segment]) * s.fraction};
  }
};

/// Six components give a symmetric tensor, nine (the six, then 21, 31, 32) a general one.
template <typename T>
tmech::tensor<T, 3, 2> parse_tensor(nlohmann::json const &j) {
  auto const v = j.get<std::vector<T>>();
  if (v.size() == 6)
    return symmetric_tensor<T>(std::span<T const, 6>{v.data(), 6});
  if (v.size() == 9)
    return general_tensor<T>(std::span<T const, 9>{v.data(), 9});
  throw std::invalid_argument("expected six components [11, 22, 33, 12, 13, 23] or nine (then 21, 31, 32)");
}

template <typename T>
tmech::tensor<T, 3, 2> parse_symmetric(nlohmann::json const &j) {
  return parse_tensor<T>(j);
}

/// A point with a deformation_gradient_source is driven by F and reports P.
inline bool is_finite_strain(nlohmann::json const &doc) {
  return doc.at("point").contains("deformation_gradient_source");
}

namespace detail {
/// `v` as an int of at least `min`; fractions and values beyond int throw.
inline int as_count(nlohmann::json const &v, std::string const &what, int min) {
  if (!v.is_number_integer() || v.get<std::int64_t>() < min || v.get<std::int64_t>() > std::numeric_limits<int>::max())
    throw std::invalid_argument(what + ": expected an integer >= " + std::to_string(min) + ", got " + v.dump());
  return v.get<int>();
}

inline int count(nlohmann::json const &c, std::string const &key, int fallback, int min) {
  return c.contains(key) ? as_count(c[key], "case '" + c.value("name", std::string{}) + "': " + key, min) : fallback;
}

template <typename Control>
Control parse_control(nlohmann::json const &c, Control all_prescribed) {
  auto const kind = c.value("control", std::string{"strain"});
  if (kind == "strain")
    return all_prescribed;
  if (kind == "uniaxial_stress")
    return Control::uniaxial_stress();
  if (kind == "mixed") {
    auto const names = c.at("stress_components").get<std::vector<std::string>>();
    std::vector<std::string_view> const views(names.begin(), names.end());
    return Control::mixed(std::span<std::string_view const>{views});
  }
  throw std::invalid_argument("unknown control '" + kind + "' (strain, uniaxial_stress, mixed)");
}
} // namespace detail

template <typename T>
load_control<T> parse_control(nlohmann::json const &c) {
  return detail::parse_control(c, load_control<T>::strain());
}

/// The same "control" for finite strain; "mixed" may name 21, 31, 32 as well.
template <typename T>
umat::deformation_control<T> parse_deformation_control(nlohmann::json const &c) {
  return detail::parse_control(c, umat::deformation_control<T>::deformation());
}

/// Proportional: "direction" times the factors of "path". Non-proportional: "points", strain states
/// in order (the load factor then counts segments). Both are scaled by "scale". Only the control of
/// the point's kinematics is parsed (is_finite_strain()); the other stays all prescribed.
template <typename T>
load_case<T> parse_case(nlohmann::json const &c, bool finite_strain = false) {
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
  return {c.at("name").get<std::string>(), finite_strain ? load_control<T>::strain() : parse_control<T>(c),
          finite_strain ? parse_deformation_control<T>(c) : umat::deformation_control<T>::deformation(), std::move(points),
          c.contains("stress") ? parse_symmetric<T>(c["stress"]) : tensor2{},
          load_path<T>{std::move(factors), detail::count(c, "steps", 50, 1), c.contains("points") ? T{1} : scale,
                       c.value("dt", T{1})},
          detail::count(c, "tangent_every", 1, 0), finite_strain};
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
    std::vector<int> steps;
    for (auto const &n : s.at("steps"))
      steps.push_back(detail::as_count(n, "study '" + s.at("name").get<std::string>() + "': steps", 1));
    out.push_back({s.at("name").get<std::string>(), *it, std::move(steps)});
  }
  return out;
}

/// The file stems the report writes, cases first, then <study>_<steps>: no path, all distinct.
inline std::vector<std::string> output_stems(nlohmann::json const &doc) {
  std::vector<std::string> out;
  auto const add = [&](std::string stem) {
    bool const plain{!stem.empty() && stem != "." && stem != ".." && std::ranges::none_of(stem, [](unsigned char ch) {
      return ch == '/' || ch == '\\' || std::iscntrl(ch);
    })};
    if (!plain)
      throw std::invalid_argument("report: '" + stem + "' is no file name in the output directory");
    if (std::ranges::contains(out, stem))
      throw std::invalid_argument("report: two outputs are named '" + stem + "'");
    out.push_back(std::move(stem));
  };
  for (auto const &c : doc.at("cases"))
    add(c.at("name").get<std::string>());
  for (auto const &study : parse_studies(doc))
    for (auto const n : study.steps)
      add(study.name + "_" + std::to_string(n));
  return out;
}

/// "point": the parameters of the material-point driver (strain_source or deformation_gradient_source,
/// stress_source, optional tangent_source and time_source, and optionally tolerance, max_iter, fd_step).
template <typename Traits>
typename Traits::ParameterHandler parse_point(nlohmann::json const &doc) {
  using T = typename Traits::value_type;
  typename Traits::ParameterHandler out;
  for (auto const &[key, value] : doc.at("point").items()) {
    if (key == "strain_source" || key == "deformation_gradient_source" || key == "stress_source" ||
        key == "tangent_source" || key == "time_source")
      out.template insert<std::string>(key, value.template get<std::string>());
    else if (key == "tolerance" || key == "fd_step")
      out.template insert<T>(key, value.template get<T>());
    else if (key == "max_iter")
      out.template insert<int>(key, detail::as_count(value, "report: \"point\": max_iter", 0));
    else
      throw std::invalid_argument("report: \"point\" has no parameter '" + key + "'");
  }
  return out;
}

/// A report column wired to a graph property.
struct column_spec {
  std::string name;
  std::string source; ///< "material::property[i,j]"
  bool full{false};   ///< rank-2: nine components instead of the symmetric six
};

/// The point's kinematics and stress first (strain/stress, or F/P with nine components), then
/// "record" (nlohmann::json orders those by column name; a value may be {"source", "full"}).
inline std::vector<column_spec> parse_columns(nlohmann::json const &doc) {
  auto const &p{doc.at("point")};
  auto const stress{p.at("stress_source").get<std::string>() + "::stress"};
  std::vector<column_spec> out;
  if (is_finite_strain(doc))
    out = {{"F", p.at("deformation_gradient_source").get<std::string>() + "::deformation_gradient", true}, {"P", stress, true}};
  else
    out = {{"strain", p.at("strain_source").get<std::string>() + "::strain"}, {"stress", stress}};
  auto const record = doc.value("record", nlohmann::json::object());
  for (auto const &[name, value] : record.items())
    if (value.is_string())
      out.push_back({name, value.get<std::string>()});
    else
      out.push_back({name, value.at("source").get<std::string>(), value.value("full", false)});
  return out;
}

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_REPORT_JSON_H
