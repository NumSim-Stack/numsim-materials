#ifndef NUMSIM_MATERIALS_CORE_GRAPH_DESCRIPTION_H
#define NUMSIM_MATERIALS_CORE_GRAPH_DESCRIPTION_H

#include <algorithm>
#include <string>
#include <vector>

#include "numsim-materials/core/material_context.h"

namespace numsim::materials {

struct produced_property {
  std::string name;
  bool history{false};
};

struct consumed_property {
  std::string source; ///< "material::property"
  bool local{false};  ///< EdgeKind::Local
};

struct material_description {
  std::string name;
  std::vector<produced_property> produced; ///< sorted by name
  std::vector<consumed_property> inputs;
};

/// Materials in creation order with what they produce and consume.
template <typename Traits>
std::vector<material_description> describe_graph(material_context<Traits> const &ctx) {
  std::vector<material_description> out;
  for (auto const *mat : ctx.materials()) {
    material_description d{std::string{mat->name()}, {}, {}};
    for (auto const *p : mat->get_property_registry().produced_properties())
      d.produced.push_back({p->traits().id.name, p->is_history()});
    std::ranges::sort(d.produced, {}, &produced_property::name); // the registry orders by address
    for (auto const &input : mat->typed_inputs())
      d.inputs.push_back({input->source_owner() + "::" + input->source_name(), input->edge_kind() == EdgeKind::Local});
    out.push_back(std::move(d));
  }
  return out;
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_GRAPH_DESCRIPTION_H
