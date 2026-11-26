#ifndef NUMSIM_MATERIALS_REPORT_REPORT_JSON_H
#define NUMSIM_MATERIALS_REPORT_REPORT_JSON_H

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/driver/load_control.h"
#include "numsim-materials/report/load_path.h"

/// Parsing of report documents into value types; nothing here runs a material.
namespace numsim::materials::report {

template <typename T>
struct load_case {
  using tensor2 = tmech::tensor<T, 3, 2>;
  std::string name;
  load_control<T> control;                     ///< small strain
  deformation_control<T> finite_control; ///< the same components, for F / P
  std::vector<tensor2> points; ///< strain states (or F - I) the path interpolates (already scaled)
  tensor2 stress;              ///< stress target per unit load factor
  load_path<T> path;
  int tangent_every{1};        ///< finite-difference tangent check every n-th step; 0 = never
  bool finite_strain{false};   ///< which of the two controls was parsed
  std::optional<T> fd_step;    ///< perturbation of the tangent check; unset = the driver's default
  int max_cutbacks{0};         ///< halvings of a failing step before the case stops

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

/// Six components only, a small-strain tensor.
template <typename T>
tmech::tensor<T, 3, 2> parse_symmetric(nlohmann::json const &j) {
  auto const v = j.get<std::vector<T>>();
  if (v.size() != 6)
    throw std::invalid_argument("expected six components [11, 22, 33, 12, 13, 23], got " + std::to_string(v.size()));
  return symmetric_tensor<T>(std::span<T const, 6>{v.data(), 6});
}

namespace detail {
/// Runs `f`; an nlohmann error or an invalid_argument it throws gets `where` in front.
template <typename F>
decltype(auto) in_context(std::string const &where, F &&f) {
  try {
    return f();
  } catch (nlohmann::json::exception const &e) {
    throw std::invalid_argument(where + ": " + e.what());
  } catch (std::invalid_argument const &e) {
    throw std::invalid_argument(where + ": " + e.what());
  }
}

/// j[key] as V; errors name the key.
template <typename V>
V get(nlohmann::json const &j, std::string const &key) {
  return in_context("\"" + key + "\"", [&] { return j.at(key).template get<V>(); });
}

/// The same, `fallback` without the key.
template <typename V>
V get(nlohmann::json const &j, std::string const &key, V fallback) {
  return j.is_object() && j.contains(key) ? get<V>(j, key) : fallback;
}

/// "case 'name'" for messages.
inline std::string case_label(nlohmann::json const &c) {
  if (c.is_object() && c.contains("name") && c["name"].is_string())
    return "case '" + c["name"].get<std::string>() + "'";
  return "a case without a \"name\"";
}
} // namespace detail

/// A point with a deformation_gradient_source is driven by F and reports P.
inline bool is_finite_strain(nlohmann::json const &doc) {
  return detail::in_context("report: \"point\"", [&] { return doc.at("point").contains("deformation_gradient_source"); });
}

namespace detail {
/// `v` as an int of at least `min`; fractions and values beyond int throw.
inline int as_count(nlohmann::json const &v, std::string const &what, int min) {
  if (!v.is_number_integer() || v.get<std::int64_t>() < min || v.get<std::int64_t>() > std::numeric_limits<int>::max())
    throw std::invalid_argument(what + ": expected an integer >= " + std::to_string(min) + ", got " + v.dump());
  return v.get<int>();
}

inline int count(nlohmann::json const &c, std::string const &key, int fallback, int min) {
  return c.contains(key) ? as_count(c[key], "\"" + key + "\"", min) : fallback;
}

/// "max_cutbacks": 0 (the default) to 20, a million substeps.
inline int max_cutbacks(nlohmann::json const &c) {
  auto const n{count(c, "max_cutbacks", 0, 0)};
  if (n > 20)
    throw std::invalid_argument("\"max_cutbacks\": at most 20, got " + std::to_string(n));
  return n;
}

template <typename Control>
Control parse_control(nlohmann::json const &c, Control all_prescribed) {
  auto const kind{get<std::string>(c, "control", "strain")};
  if (kind == "strain")
    return all_prescribed;
  if (kind == "uniaxial_stress")
    return Control::uniaxial_stress();
  if (kind == "mixed") {
    auto const names{get<std::vector<std::string>>(c, "stress_components")};
    std::vector<std::string_view> const views(names.begin(), names.end());
    return in_context("\"stress_components\"", [&] { return Control::mixed(std::span<std::string_view const>{views}); });
  }
  throw std::invalid_argument("\"control\": unknown '" + kind + "' (strain, uniaxial_stress, mixed)");
}

/// `j` as a tensor: nine components only at finite strain.
template <typename T>
tmech::tensor<T, 3, 2> tensor_of(nlohmann::json const &j, bool finite_strain) {
  return finite_strain ? parse_tensor<T>(j) : parse_symmetric<T>(j);
}

/// c[key] as a tensor; errors name the key.
template <typename T>
tmech::tensor<T, 3, 2> tensor_at(nlohmann::json const &c, std::string const &key, bool finite_strain) {
  return in_context("\"" + key + "\"", [&] { return tensor_of<T>(c.at(key), finite_strain); });
}

template <typename T>
load_case<T> parse_case(nlohmann::json const &c, bool finite_strain) {
  using tensor2 = tmech::tensor<T, 3, 2>;
  if (!c.is_object())
    throw std::invalid_argument("expected an object, got " + c.dump());
  auto const name{get<std::string>(c, "name")};
  auto const scale{get<T>(c, "scale", T{1})};
  std::vector<tensor2> points;
  std::vector<T> factors;
  if (c.contains("points") == c.contains("direction"))
    throw std::invalid_argument("give either direction + path or points");
  if (c.contains("points")) {
    if (c.contains("path"))
      throw std::invalid_argument("points and path exclude each other");
    auto const &list{c["points"]};
    if (!list.is_array())
      throw std::invalid_argument("\"points\": expected an array of tensors, got " + list.dump());
    for (std::size_t i{0}; i < list.size(); ++i) {
      points.push_back(tensor2{scale * in_context("\"points\"[" + std::to_string(i) + "]", [&] { return tensor_of<T>(list[i], finite_strain); })});
      factors.push_back(T(factors.size()));
    }
    if (!points.empty() && tmech::norm(points.front()) != T{0})
      throw std::invalid_argument("the first point must be zero (step 0 is the unloaded state)");
  } else {
    auto const direction{tensor_at<T>(c, "direction", finite_strain)};
    factors = get<std::vector<T>>(c, "path");
    if (!factors.empty() && factors.front() != T{0})
      throw std::invalid_argument("the path must start at 0 (step 0 is the unloaded state)");
    for (auto const f : factors)
      points.push_back(tensor2{scale * f * direction});
  }
  return {name, finite_strain ? load_control<T>::strain() : parse_control(c, load_control<T>::strain()),
          finite_strain ? parse_control(c, deformation_control<T>::deformation()) : deformation_control<T>::deformation(),
          std::move(points), c.contains("stress") ? tensor_at<T>(c, "stress", finite_strain) : tensor2{},
          load_path<T>{std::move(factors), count(c, "steps", 50, 1), c.contains("points") ? T{1} : scale, get<T>(c, "dt", T{1})},
          count(c, "tangent_every", 1, 0), finite_strain,
          c.contains("fd_step") ? std::optional<T>{get<T>(c, "fd_step")} : std::nullopt, max_cutbacks(c)};
}
} // namespace detail

template <typename T>
load_control<T> parse_control(nlohmann::json const &c) {
  return detail::parse_control(c, load_control<T>::strain());
}

/// The same "control" for finite strain; "mixed" may name 21, 31, 32 as well.
template <typename T>
deformation_control<T> parse_deformation_control(nlohmann::json const &c) {
  return detail::parse_control(c, deformation_control<T>::deformation());
}

/// Proportional: "direction" times the factors of "path". Non-proportional: "points", strain states
/// in order (the load factor then counts segments). Both are scaled by "scale". Only the control of
/// the point's kinematics is parsed (is_finite_strain()); the other stays all prescribed. Tensors have
/// six components, at finite strain also nine. Errors are
/// invalid_arguments naming the case and the key.
template <typename T>
load_case<T> parse_case(nlohmann::json const &c, bool finite_strain = false) {
  return detail::in_context(detail::case_label(c), [&] { return detail::parse_case<T>(c, finite_strain); });
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
  std::vector<std::string> names;
  detail::in_context("report: \"cases\"", [&] {
    for (auto const &c : doc.at("cases"))
      names.push_back(detail::get<std::string>(c, "name"));
  });
  for (auto &name : names)
    add(std::move(name));
  for (auto const &study : parse_studies(doc))
    for (auto const n : study.steps)
      add(study.name + "_" + std::to_string(n));
  return out;
}

/// "point": the parameters of the material-point driver (strain_source or deformation_gradient_source,
/// stress_source, optional tangent_source, stress_property, tangent_property ("damaged_stress") and time_source,
/// and optionally tolerance, max_iter, fd_step, and the deprecated fixed point residual_source, residual_tolerance,
/// max_passes: a composite declaring converges() takes its tolerance and max_passes as parameters).
template <typename Traits>
typename Traits::ParameterHandler parse_point(nlohmann::json const &doc) {
  using T = typename Traits::value_type;
  return detail::in_context("report: \"point\"", [&] {
    auto const &point{doc.at("point")};
    if (!point.is_object())
      throw std::invalid_argument("expected an object, got " + point.dump());
    typename Traits::ParameterHandler out;
    for (auto const &[key, value] : point.items()) {
      if (key == "strain_source" || key == "deformation_gradient_source" || key == "stress_source" ||
          key == "tangent_source" || key == "stress_property" || key == "tangent_property" || key == "time_source" ||
          key == "residual_source")
        out.template insert<std::string>(key, detail::get<std::string>(point, key));
      else if (key == "tolerance" || key == "fd_step" || key == "residual_tolerance")
        out.template insert<T>(key, detail::get<T>(point, key));
      else if (key == "max_iter")
        out.template insert<int>(key, detail::as_count(value, "\"max_iter\"", 0));
      else if (key == "max_passes")
        out.template insert<int>(key, detail::as_count(value, "\"max_passes\"", 1));
      else
        throw std::invalid_argument("no parameter '" + key + "'");
    }
    return out;
  });
}

/// A report column wired to a graph property.
struct column_spec {
  std::string name;
  std::string source; ///< "material::property[i,j]"
  bool full{false};   ///< rank-2: nine components instead of the symmetric six
};

/// The point's kinematics and stress first (strain/stress, or F/P with nine components), then
/// "record" (nlohmann::json orders those by column name; a value may be {"source", "full"}). Names
/// are CSV fields: no comma, quote or control character.
inline std::vector<column_spec> parse_columns(nlohmann::json const &doc) {
  std::vector<column_spec> out;
  detail::in_context("report: \"point\"", [&] {
    auto const &p{doc.at("point")};
    auto const stress{detail::get<std::string>(p, "stress_source") + "::" +
                      detail::get<std::string>(p, "stress_property", "stress")};
    if (is_finite_strain(doc))
      out = {{"F", detail::get<std::string>(p, "deformation_gradient_source") + "::deformation_gradient", true},
             {"P", stress, true}};
    else
      out = {{"strain", detail::get<std::string>(p, "strain_source") + "::strain"}, {"stress", stress}};
  });
  detail::in_context("report: \"record\"", [&] {
    auto const record = doc.value("record", nlohmann::json::object());
    if (!record.is_object())
      throw std::invalid_argument("expected an object {\"column\": \"material::property\"}, got " + record.dump());
    for (auto const &[name, value] : record.items())
      detail::in_context("\"" + name + "\"", [&] {
        if (name.empty() || std::ranges::any_of(name, [](unsigned char ch) { return ch == ',' || ch == '"' || std::iscntrl(ch); }))
          throw std::invalid_argument("a column name is not empty and has no comma, quote or control character");
        if (value.is_string())
          out.push_back({name, value.get<std::string>()});
        else
          out.push_back({name, detail::get<std::string>(value, "source"), detail::get<bool>(value, "full", false)});
      });
  });
  return out;
}

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_REPORT_JSON_H
