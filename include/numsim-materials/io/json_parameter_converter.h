#ifndef JSON_PARAMETER_CONVERTER_H
#define JSON_PARAMETER_CONVERTER_H

#include <algorithm>
#include <any>
#include <functional>
#include <iterator>
#include <print>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <map>
#include <variant>
#include <vector>
#include <numsim-core/input_parameter_controller.h>
#include "numsim-materials/core/unknown_spec.h"

namespace numsim::materials {

// --- JSON adapter traits ---
// Specialize for your JSON library. Default works with nlohmann::json.

template<typename JsonType>
struct json_adapter {
  static bool contains(const JsonType& j, const std::string& key) { return j.contains(key); }
  static const JsonType& at(const JsonType& j, const std::string& key) { return j.at(key); }
  static bool is_object(const JsonType& j) { return j.is_object(); }
  static bool is_array(const JsonType& j) { return j.is_array(); }
  static bool is_string(const JsonType& j) { return j.is_string(); }
  static bool is_number(const JsonType& j) { return j.is_number(); }
  template<typename T> static T get(const JsonType& j) { return j.template get<T>(); }

  template<typename Fn>
  static void for_each_key(const JsonType& j, Fn&& fn) {
    for (auto it = j.begin(); it != j.end(); ++it)
      fn(it.key());
  }
};

// --- Reader registry ---
// Maps std::type_index → function that reads a JSON value and returns std::any.
// This is purely a data reader — it doesn't know about parameter_handler.

template<typename JsonType>
class json_reader_registry {
public:
  /// A reader gets the parameter key with the value: readers are found by type, and two parameters of one
  /// type (vector_newton's "zero_blocks", weighted_sum's "terms") must each report under their own name.
  using reader_fn = std::function<std::any(const JsonType&, const std::string&)>;
  using adapter = json_adapter<JsonType>;

  /// Register a type with default JSON conversion (json.get<T>() → std::any).
  template<typename T>
  json_reader_registry& add() {
    m_readers[typeid(T)] = [](const JsonType& j, const std::string&) -> std::any {
      return adapter::template get<T>(j);
    };
    return *this;
  }

  /// Register a type with a custom reader function.
  template<typename T>
  json_reader_registry& add(reader_fn fn) {
    m_readers[typeid(T)] = std::move(fn);
    return *this;
  }

  /// A reader for one parameter, taking precedence over the reader of its type.
  template<typename T>
  json_reader_registry& add_for_key(std::string key, reader_fn fn) {
    m_keyed_readers[{std::type_index(typeid(T)), std::move(key)}] = std::move(fn);
    return *this;
  }

  /// Read a JSON value and return as std::any.
  /// Throws if no reader is registered for the given type.
  std::any read(std::type_index tid, const JsonType& j, const std::string& key) const {
    const reader_fn* fn = nullptr;
    if (auto keyed = m_keyed_readers.find({tid, key}); keyed != m_keyed_readers.end())
      fn = &keyed->second;
    else if (auto it = m_readers.find(tid); it != m_readers.end())
      fn = &it->second;

    if (!fn)
      throw std::runtime_error(
          "json_reader_registry: no reader for parameter '" + key + "'");
    try {
      return (*fn)(j, key);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          "json_reader_registry: failed to read parameter '" + key + "': " + e.what());
    }
  }

private:
  struct keyed_hash {
    std::size_t operator()(const std::pair<std::type_index, std::string>& k) const noexcept {
      return std::hash<std::type_index>{}(k.first) ^
             (std::hash<std::string>{}(k.second) << 1);
    }
  };

  std::unordered_map<std::type_index, reader_fn> m_readers;
  std::unordered_map<std::pair<std::type_index, std::string>, reader_fn, keyed_hash>
      m_keyed_readers;
};

// --- Default registry factory ---

template<typename JsonType>
json_reader_registry<JsonType> make_default_json_registry() {
  using adapter = json_adapter<JsonType>;
  json_reader_registry<JsonType> reg;

  // Scalars
  reg.template add<double>();
  reg.template add<float>();
  reg.template add<int>();
  reg.template add<std::size_t>();
  reg.template add<bool>();
  reg.template add<std::string>();

  // Vectors
  reg.template add<std::vector<double>>();
  reg.template add<std::vector<std::string>>();

  // vector<size_t>: element-wise conversion
  reg.template add<std::vector<std::size_t>>(
      [](const JsonType& j, const std::string&) -> std::any {
        std::vector<std::size_t> result;
        for (const auto& elem : j)
          result.push_back(adapter::template get<std::size_t>(elem));
        return result;
      });

  // Map type
  reg.template add<std::unordered_map<std::string, std::vector<std::string>>>();

  // a list of objects of strings: the per-phase "material::property" sources of a local composite
  reg.template add<std::vector<std::map<std::string, std::string>>>([](const JsonType& j, const std::string& key) -> std::any {
    if (!adapter::is_array(j))
      throw std::invalid_argument("'" + key + "' is a list of objects {role: \"material::property\"}");
    std::vector<std::map<std::string, std::string>> result;
    for (const auto& elem : j) {
      if (!adapter::is_object(elem))
        throw std::invalid_argument("'" + key + "' is a list of objects {role: \"material::property\"}");
      auto& entry = result.emplace_back();
      adapter::for_each_key(elem, [&](const std::string& role) {
        entry.emplace(role, adapter::template get<std::string>(adapter::at(elem, role)));
      });
    }
    return result;
  });

  // scalar_or_property: number or {"class": "...", "property": "..."}
  using sop = std::variant<double, std::pair<std::string, std::string>>;
  auto convert_sop = [](const JsonType& j) -> sop {
    if (adapter::is_object(j))
      return std::pair{adapter::template get<std::string>(adapter::at(j, "class")),
                       adapter::template get<std::string>(adapter::at(j, "property"))};
    return adapter::template get<double>(j);
  };

  reg.template add<sop>(
      [convert_sop](const JsonType& j, const std::string&) -> std::any {
        return convert_sop(j);
      });

  reg.template add<std::vector<sop>>(
      [convert_sop](const JsonType& j, const std::string&) -> std::any {
        std::vector<sop> result;
        for (const auto& elem : j)
          result.push_back(convert_sop(elem));
        return result;
      });

  // unknown_spec list for coupled local systems (vector_newton):
  //   "unknowns": [{"name": "dgamma", "kind": "scalar"},
  //                {"name": "backstress", "kind": "sym_tensor"}]
  // No "dim" — the dimension is fixed by the Traits policy, which keeps the
  // kind set small enough for an exhaustive switch on the solver side.
  reg.template add<std::vector<unknown_spec>>(
      [](const JsonType& j, const std::string&) -> std::any {
        std::vector<unknown_spec> result;
        for (const auto& elem : j) {
          unknown_spec s;
          s.name = adapter::template get<std::string>(adapter::at(elem, "name"));
          const auto kind =
              adapter::template get<std::string>(adapter::at(elem, "kind"));
          if (kind == "scalar")           s.kind = unknown_kind::scalar;
          else if (kind == "sym_tensor")  s.kind = unknown_kind::sym_tensor;
          else
            throw std::runtime_error(
                "unknown_spec: unrecognised kind '" + kind +
                "' (expected \"scalar\" or \"sym_tensor\")");
          result.push_back(std::move(s));
        }
        return result;
      });

  // a vector of string pairs, shared by "zero_blocks" and "terms": the message names the key it was given
  // and says what an entry must be in that parameter's words
  const auto string_pairs = [](const JsonType& j, const std::string& key, const char* entry) -> std::any {
    std::vector<block_ref> result;
    std::ranges::transform(j, std::back_inserter(result), [&](const JsonType& elem) {
      std::vector<std::string> names;
      std::ranges::transform(elem, std::back_inserter(names),
                             [](const JsonType& part) { return adapter::template get<std::string>(part); });
      if (names.size() != 2)
        throw std::runtime_error(key + ": each entry must be " + entry + ", but one has " + std::to_string(names.size()));
      return block_ref{std::move(names[0]), std::move(names[1])};
    });
    return result;
  };
  reg.template add<std::vector<block_ref>>(
      [string_pairs](const JsonType& j, const std::string& key) { return string_pairs(j, key, "a pair of two names"); });
  reg.template add_for_key<std::vector<block_ref>>("zero_blocks", [string_pairs](const JsonType& j, const std::string& key) {
    return string_pairs(j, key, "a [row, column] pair of unknown names");
  });

  return reg;
}

// --- Concrete JSON visitor ---
// Implements parameter_visitor_base: reads JSON values via the reader registry.
// input_parameter<T>::accept() calls contains() + read(), then does the typed
// insertion into parameter_handler — the visitor never touches the handler.

template<typename JsonType, typename KeyType = std::string>
class json_parameter_visitor final
    : public numsim_core::parameter_visitor_base<KeyType> {
public:
  using adapter = json_adapter<JsonType>;
  using registry_type = json_reader_registry<JsonType>;

  json_parameter_visitor(const JsonType& json, const registry_type& registry)
      : m_json(json), m_registry(registry) {}

  bool contains(const KeyType& key) const override {
    return adapter::contains(m_json, key);
  }

  std::any read(const KeyType& key, std::type_index tid) const override {
    return m_registry.read(tid, adapter::at(m_json, key), key);
  }

private:
  const JsonType& m_json;
  const registry_type& m_registry;
};

// --- Top-level conversion function ---

/// Convert a JSON object into a parameter_handler using the schema from
/// an input_parameter_controller and a JSON reader registry.
///
/// Pairs {schema, json} → validated parameter_handler:
///   1. Each input_parameter<T> reads from JSON via the visitor (type-safe via accept())
///   2. Validation checks run (is_required, set_default, check_range, etc.)
///
/// Keys in JSON but not in schema trigger a warning (likely a typo).
template<typename JsonType, typename KeyType, typename ParameterHandler>
void json_to_parameters(
    const JsonType& json,
    const numsim_core::input_parameter_controller<KeyType, ParameterHandler>& schema,
    ParameterHandler& params,
    const json_reader_registry<JsonType>& registry)
{
  using adapter = json_adapter<JsonType>;

  // Warn about JSON keys not in schema
  std::unordered_set<std::string> schema_keys;
  for (const auto& [key, _] : schema)
    schema_keys.insert(key);

  adapter::for_each_key(json, [&](const std::string& key) {
    if (key != "type" && !schema_keys.contains(key))
      std::println(stderr, "  warning: unknown parameter '{}' in JSON (not in schema)", key);
  });

  // Read + insert + validate in one call
  json_parameter_visitor<JsonType, KeyType> visitor(json, registry);
  schema.accept(visitor, params);
}

/// Convenience overload that uses the default reader registry.
template<typename JsonType, typename KeyType, typename ParameterHandler>
void json_to_parameters(
    const JsonType& json,
    const numsim_core::input_parameter_controller<KeyType, ParameterHandler>& schema,
    ParameterHandler& params)
{
  const auto registry = make_default_json_registry<JsonType>();
  json_to_parameters(json, schema, params, registry);
}

} // namespace numsim::materials

#endif // JSON_PARAMETER_CONVERTER_H
