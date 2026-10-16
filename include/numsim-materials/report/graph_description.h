#ifndef NUMSIM_MATERIALS_REPORT_GRAPH_DESCRIPTION_H
#define NUMSIM_MATERIALS_REPORT_GRAPH_DESCRIPTION_H

#include <algorithm>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "numsim-materials/core/history_property.h"
#include "numsim-materials/report/case_runner.h"

namespace numsim::materials::report {

namespace detail {
template <typename Traits>
bool is_history(property_base const *p) {
  using T = typename Traits::value_type;
  auto const holds = [p]<typename V>() { return dynamic_cast<history_property<V, property_traits> const *>(p) != nullptr; };
  return holds.template operator()<T>() || holds.template operator()<tmech::tensor<T, 3, 2>>() ||
         holds.template operator()<tmech::tensor<T, 3, 4>>();
}
} // namespace detail

/// Materials of a report's model with their parameters as used, produced properties and consumed inputs.
template <typename Traits>
nlohmann::json describe_graph(nlohmann::json const &doc) {
  using nlohmann::json;
  material_point<Traits> point(doc);
  auto &factory{object_store<Traits>::factory_type::instance()};
  auto out = json::array();
  for (auto const &m : doc.at("model").at("materials")) {
    auto const name = m.at("name").get<std::string>();
    auto const type = m.at("type").get<std::string>();
    auto entry = json::object();
    entry["name"] = name;
    entry["type"] = type;
    entry["parameters"] = json::array();
    entry["properties"] = json::array();
    entry["inputs"] = json::array();
    if (factory.contains(type))
      for (auto const &[key, param] : factory.schema(type)) {
        auto row = json::object();
        row["name"] = key;
        row["value"] = m.contains(key) ? m[key] : json{};
        row["description"] = param->description();
        entry["parameters"].push_back(row);
      }
    if (auto const *mat{point.context().find(name)}) {
      auto const &registry{mat->get_property_registry().produced_properties()};
      std::vector<property_base const *> produced(registry.begin(), registry.end());
      std::ranges::sort(produced, {}, [](auto const *p) { return p->traits().id.name; }); // the registry orders by address
      for (auto const *p : produced) {
        auto prop = json::object();
        prop["name"] = p->traits().id.name;
        prop["history"] = detail::is_history<Traits>(p);
        entry["properties"].push_back(prop);
      }
      for (auto const &input : mat->typed_inputs()) {
        auto in = json::object();
        in["source"] = input->source_owner() + "::" + input->source_name();
        in["kind"] = input->edge_kind() == EdgeKind::Global ? "global" : "local";
        entry["inputs"].push_back(in);
      }
    }
    out.push_back(entry);
  }
  return out;
}

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_GRAPH_DESCRIPTION_H
