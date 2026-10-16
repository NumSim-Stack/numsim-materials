#ifndef NUMSIM_MATERIALS_REPORT_REPORT_JSON_H
#define NUMSIM_MATERIALS_REPORT_REPORT_JSON_H

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "numsim-materials/report/load_control.h"
#include "numsim-materials/report/load_path.h"
#include "numsim-materials/report/symmetric_basis.h"

/// Parsing of report documents into value types; nothing here runs a material.
namespace numsim::materials::report {

template <typename T>
struct load_case {
  std::string name;
  load_control<T> control;
  tmech::tensor<T, 3, 2> direction; ///< strain per unit load
  tmech::tensor<T, 3, 2> stress;    ///< stress target per unit load
  load_path<T> path;
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

template <typename T>
load_case<T> parse_case(nlohmann::json const &c) {
  return {c.at("name").get<std::string>(), parse_control<T>(c), parse_symmetric<T>(c.at("direction")),
          c.contains("stress") ? parse_symmetric<T>(c["stress"]) : tmech::tensor<T, 3, 2>{},
          load_path<T>{c.at("path").get<std::vector<T>>(), c.value("steps", 50), c.value("scale", T{1}), c.value("dt", T{1})}};
}

/// "record" as (column, "material::property") pairs, sorted by column name.
inline std::vector<std::pair<std::string, std::string>> parse_record(nlohmann::json const &doc) {
  std::vector<std::pair<std::string, std::string>> out;
  auto const record = doc.value("record", nlohmann::json::object());
  for (auto const &[name, source] : record.items())
    out.emplace_back(name, source.get<std::string>());
  return out;
}

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_REPORT_JSON_H
