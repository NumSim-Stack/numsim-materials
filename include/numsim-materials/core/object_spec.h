#ifndef NUMSIM_MATERIALS_CORE_OBJECT_SPEC_H
#define NUMSIM_MATERIALS_CORE_OBJECT_SPEC_H

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace numsim::materials {

/// A configured object inside a parameter (a mean-field kernel, an inclusion shape): its type and its own
/// parameters, as the deck object {"type": ..., ...}.
template <typename ParameterHandler>
struct object_spec {
  std::string type;
  ParameterHandler parameters;
};

/// Types of one family of configured objects: each registered type brings its schema and is built from
/// an object_spec, its parameters checked against that schema like a material's.
template <typename Base, typename Controller, typename ParameterHandler>
class spec_registry {
public:
  explicit spec_registry(std::string family) : m_family(std::move(family)) {}

  template <typename Derived>
  spec_registry &add(std::string type) {
    m_entries.insert_or_assign(std::move(type), entry{&Derived::parameters, [](ParameterHandler p) -> std::unique_ptr<Base> {
                                                        return std::make_unique<Derived>(std::move(p));
                                                      }});
    return *this;
  }

  [[nodiscard]] std::unique_ptr<Base> make(object_spec<ParameterHandler> const &spec) const {
    return find(spec.type).make(spec.parameters);
  }

  [[nodiscard]] Controller schema(std::string const &type) const { return find(type).schema(); }

  [[nodiscard]] std::vector<std::string> registered_types() const {
    std::vector<std::string> types;
    for (auto const &[type, e] : m_entries)
      types.push_back(type);
    return types;
  }

private:
  struct entry {
    Controller (*schema)();
    std::function<std::unique_ptr<Base>(ParameterHandler)> make;
  };

  entry const &find(std::string const &type) const {
    if (auto const it{m_entries.find(type)}; it != m_entries.end())
      return it->second;
    std::string known;
    for (auto const &[t, e] : m_entries)
      known += (known.empty() ? "" : ", ") + t;
    throw std::invalid_argument("unknown " + m_family + " '" + type + "' (" + known + ")");
  }

  std::string m_family;
  std::map<std::string, entry> m_entries;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_OBJECT_SPEC_H
