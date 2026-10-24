#ifndef NUMSIM_MATERIALS_REPORT_GRAPH_DESCRIPTION_H
#define NUMSIM_MATERIALS_REPORT_GRAPH_DESCRIPTION_H

#include <string>

#include <nlohmann/json.hpp>
#include "numsim-materials/core/graph_description.h"
#include "numsim-materials/report/case_runner.h"

namespace numsim::materials::report {

/// describe_graph of a report's model as JSON, with each material's type and parameters as used.
template <typename Traits>
nlohmann::json describe_graph(nlohmann::json const &doc) {
  using nlohmann::json;
  case_model<Traits> model(doc);
  auto &factory{object_store<Traits>::factory_type::instance()};
  auto out = json::array();
  for (auto const &d : materials::describe_graph(model.context())) {
    auto const *spec{[&]() -> json const * {
      for (auto const &m : doc.at("model").at("materials"))
        if (m.at("name") == d.name)
          return &m;
      return nullptr;
    }()};
    std::string const type = spec != nullptr ? std::string(spec->at("type")) : std::string{};
    auto entry = json::object();
    entry["name"] = d.name;
    entry["type"] = type;
    entry["parameters"] = json::array();
    if (spec != nullptr && factory.contains(type))
      for (auto const &[key, param] : factory.schema(type)) {
        auto row = json::object();
        row["name"] = key;
        row["value"] = spec->contains(key) ? (*spec)[key] : json{};
        row["description"] = param->description();
        entry["parameters"].push_back(row);
      }
    entry["properties"] = json::array();
    for (auto const &p : d.produced) {
      auto prop = json::object();
      prop["name"] = p.name;
      prop["history"] = p.history;
      entry["properties"].push_back(prop);
    }
    entry["inputs"] = json::array();
    for (auto const &i : d.inputs) {
      auto in = json::object();
      in["source"] = i.source;
      in["kind"] = i.local ? "local" : "global";
      entry["inputs"].push_back(in);
    }
    out.push_back(entry);
  }
  return out;
}

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_GRAPH_DESCRIPTION_H
