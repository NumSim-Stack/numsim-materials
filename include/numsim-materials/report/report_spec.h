#ifndef NUMSIM_MATERIALS_REPORT_REPORT_SPEC_H
#define NUMSIM_MATERIALS_REPORT_REPORT_SPEC_H

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

/// Report documents: "include" files merged first (arrays appended, named entries replaced, local keys win),
/// then every string "$name" (an identifier) replaced by parameters.name.
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

/// "$name", name an ASCII identifier; anything else ("$\sigma_{11}$") is text.
inline bool is_parameter_reference(std::string const &s) {
  auto const letter = [](char ch) { return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '_'; };
  return s.size() > 1 && s.front() == '$' && letter(s[1]) &&
         std::all_of(s.begin() + 2, s.end(), [&](char ch) { return letter(ch) || (ch >= '0' && ch <= '9'); });
}

/// Replaces "$name" strings; `usage[name]` collects the paths where each parameter went.
inline void substitute(nlohmann::json &j, nlohmann::json const &parameters, std::string const &where,
                       nlohmann::json &usage) {
  if (j.is_string()) {
    auto const s = j.get<std::string>();
    if (is_parameter_reference(s)) {
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

/// The JSON document in `file`; a syntax error names the file.
inline nlohmann::json parse_file(std::filesystem::path const &file) {
  std::ifstream in{file};
  if (!in)
    throw std::invalid_argument("cannot open " + file.string());
  try {
    return nlohmann::json::parse(in);
  } catch (nlohmann::json::parse_error const &e) {
    throw std::invalid_argument(file.string() + ": " + e.what());
  }
}

/// The files being included (a cycle) and those already merged (a diamond merges a file once).
struct include_state {
  std::vector<std::filesystem::path> open, merged;
};

/// `spec` with its includes merged; `where` names it in messages. Include paths are relative to `dir`.
inline nlohmann::json load_included(nlohmann::json const &spec, std::filesystem::path const &dir, std::string const &where,
                                    include_state &state) {
  if (!spec.is_object())
    throw std::invalid_argument(where + ": a report document is a JSON object");
  auto const files = spec.value("include", nlohmann::json::array());
  if (!files.is_array() || !std::all_of(files.begin(), files.end(), [](auto const &f) { return f.is_string(); }))
    throw std::invalid_argument(where + ": \"include\" is an array of file names");
  auto merged = nlohmann::json::object();
  for (auto const &file : files) {
    std::filesystem::path const name{file.get<std::string>()};
    if (name.is_absolute())
      throw std::invalid_argument(where + ": include '" + name.string() + "' is absolute; includes are relative to the including file");
    auto const path{dir / name};
    if (!std::filesystem::is_regular_file(path))
      throw std::invalid_argument(where + ": cannot open include " + path.string());
    auto const canonical{std::filesystem::canonical(path)};
    if (std::ranges::contains(state.open, canonical)) {
      std::string chain;
      for (auto const &f : state.open)
        chain += f.filename().string() + " -> ";
      throw std::invalid_argument("include cycle: " + chain + canonical.filename().string() + " (" + path.string() + ")");
    }
    if (std::ranges::contains(state.merged, canonical))
      continue;
    state.open.push_back(canonical);
    merge_into(merged, load_included(parse_file(path), path.parent_path(), path.string(), state));
    state.open.pop_back();
    state.merged.push_back(canonical);
  }
  auto local = spec;
  local.erase("include");
  merge_into(merged, local);
  return merged;
}

} // namespace detail

/// The report document with includes merged, parameters substituted and "parameter_usage" added. An include is
/// a file relative to the including one, holding an object; a file included twice is merged once.
inline nlohmann::json resolve_report(nlohmann::json const &spec, std::filesystem::path const &dir) {
  detail::include_state state;
  auto out = detail::load_included(spec, dir, "report", state);
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
