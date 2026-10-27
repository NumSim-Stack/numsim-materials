#ifndef NUMSIM_MATERIALS_REPORT_REPORT_SPEC_H
#define NUMSIM_MATERIALS_REPORT_REPORT_SPEC_H

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

/// Report documents: "include" files are merged first (arrays appended, named entries replaced
/// by name, local keys win), then every string "$name" is replaced by parameters.name.
namespace numsim::materials::report {

namespace detail {

/// Appends `value` to the array `target`; an object with a "name" replaces an entry of the same name.
inline void merge_array(nlohmann::json &target, nlohmann::json const &value) {
  for (auto const &entry : value) {
    auto const it{entry.is_object() && entry.contains("name")
                      ? std::find_if(target.begin(), target.end(),
                                     [&](auto const &e) { return e.is_object() && e.contains("name") && e.at("name") == entry.at("name"); })
                      : target.end()};
    if (it == target.end())
      target.push_back(entry);
    else
      *it = entry;
  }
}

inline void merge_into(nlohmann::json &base, nlohmann::json const &local) {
  for (auto const &[key, value] : local.items()) {
    if (base.contains(key) && base[key].is_array() && value.is_array())
      merge_array(base[key], value);
    else if (base.contains(key) && base[key].is_object() && value.is_object())
      merge_into(base[key], value);
    else
      base[key] = value;
  }
}

/// Replaces "$name" strings; `usage[name]` collects the paths where each parameter went.
inline void substitute(nlohmann::json &j, nlohmann::json const &parameters, std::string const &where,
                       nlohmann::json &usage) {
  if (j.is_string()) {
    auto const s = j.get<std::string>();
    if (s.size() > 1 && s.front() == '$') {
      auto const key{s.substr(1)};
      if (!parameters.contains(key))
        throw std::invalid_argument(where + ": unknown parameter '" + s + "'");
      j = parameters[key];
      usage[key].push_back(where);
    }
  } else if (j.is_array()) {
    for (std::size_t i{0}; i < j.size(); ++i)
      substitute(j[i], parameters, where + "[" + std::to_string(i) + "]", usage);
  } else if (j.is_object()) {
    for (auto &[key, value] : j.items())
      substitute(value, parameters, where + "." + key, usage);
  }
}

inline nlohmann::json load_included(nlohmann::json const &spec, std::filesystem::path const &dir, int depth = 0) {
  if (depth > 16)
    throw std::invalid_argument("include: nesting deeper than 16 levels (a cycle?) at " + dir.string());
  auto merged = nlohmann::json::object();
  for (auto const &file : spec.value("include", nlohmann::json::array())) {
    auto const path{dir / file.get<std::string>()};
    std::ifstream in{path};
    if (!in)
      throw std::invalid_argument("include: cannot open " + path.string());
    auto const included = nlohmann::json::parse(in);
    merge_into(merged, load_included(included, path.parent_path(), depth + 1));
  }
  auto local = spec;
  local.erase("include");
  merge_into(merged, local);
  return merged;
}

} // namespace detail

/// The report document with includes merged, parameters substituted and "parameter_usage" added.
inline nlohmann::json resolve_report(nlohmann::json const &spec, std::filesystem::path const &dir) {
  auto out = detail::load_included(spec, dir);
  auto const parameters = out.value("parameters", nlohmann::json::object());
  auto usage = nlohmann::json::object();
  for (auto const &[key, value] : parameters.items())
    usage[key] = nlohmann::json::array();
  for (auto &[key, value] : out.items())
    if (key != "parameters")
      detail::substitute(value, parameters, key, usage);
  out["parameter_usage"] = usage;
  return out;
}

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_REPORT_SPEC_H
