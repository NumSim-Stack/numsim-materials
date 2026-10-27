#ifndef NUMSIM_MATERIALS_JSON_EXPANSIONS_H
#define NUMSIM_MATERIALS_JSON_EXPANSIONS_H

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "numsim-materials/io/json_substitute.h"
#include "numsim-materials/io/orientation_phases.h"

namespace numsim::materials {

/// Deck entry types that stand for several plain entries: create_from_json expands them before creating
/// anything, so the graph holds only ordinary materials. One generated entry carries the entry's name.
template <typename JsonType>
class json_expansion_registry {
public:
  using expand_fn = std::function<std::vector<JsonType>(JsonType const &)>;

  static json_expansion_registry &instance() {
    static json_expansion_registry registry;
    return registry;
  }

  json_expansion_registry &add(std::string type, expand_fn fn) {
    m_expansions.insert_or_assign(std::move(type), std::move(fn));
    return *this;
  }

  [[nodiscard]] bool contains(std::string const &type) const { return m_expansions.contains(type); }

  [[nodiscard]] std::vector<JsonType> expand(std::string const &type, JsonType const &entry) const {
    auto const it{m_expansions.find(type)};
    if (it == m_expansions.end())
      throw std::invalid_argument("json_expansion_registry: no expansion '" + type + "'");
    return it->second(entry);
  }

  [[nodiscard]] std::vector<std::string> registered_types() const {
    std::vector<std::string> types;
    for (auto const &[type, fn] : m_expansions)
      types.push_back(type);
    return types;
  }

private:
  json_expansion_registry() { add("orientation_phases", &expand_orientation_phases<JsonType>); }

  std::map<std::string, expand_fn> m_expansions;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_JSON_EXPANSIONS_H
