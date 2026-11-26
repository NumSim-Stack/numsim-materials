#ifndef NUMSIM_MATERIALS_PARAMETER_HELPERS_H
#define NUMSIM_MATERIALS_PARAMETER_HELPERS_H

#include <numsim-core/input_parameter_controller.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

namespace numsim::materials {

namespace detail {
/// How a parameter type reads in an error message.
template <typename T>
std::string parameter_type_label() {
  if constexpr (std::is_same_v<T, bool>) return "a boolean";
  else if constexpr (std::is_floating_point_v<T>) return "a real number";
  else if constexpr (std::is_integral_v<T>) return "an integer";
  else if constexpr (std::is_same_v<T, std::string>) return "a string";
  else if constexpr (requires { typename T::value_type; typename T::iterator; }) return "a list";
  else return std::string{"of type "} + typeid(T).name();
}

template <typename... Ts>
std::string parameter_type_label_of(std::type_index type) {
  std::string label{std::string{"of type "} + type.name()};
  (void)((type == typeid(Ts) && (label = parameter_type_label<Ts>(), true)) || ...);
  return label;
}

/// parameter_type_label<T>() of the T behind `type`, for the types a deck holds; other types by name.
inline std::string parameter_type_label(std::type_index type) {
  return parameter_type_label_of<bool, int, long, long long, unsigned, unsigned long, unsigned long long, float, double,
                                 long double, std::string, std::vector<int>, std::vector<std::size_t>,
                                 std::vector<float>, std::vector<double>, std::vector<std::string>>(type);
}

template <typename... Ts>
bool is_one_of(std::type_index type) {
  return ((type == typeid(Ts)) || ...);
}

/// A whole number where a real one is declared: the usual mistake behind a wrong type.
inline bool whole_for_real(std::type_index stored, std::type_index declared) {
  return is_one_of<int, long, long long, unsigned, unsigned long, unsigned long long>(stored) &&
         is_one_of<float, double, long double>(declared);
}
} // namespace detail

/// Declare a required parameter `key` of type T.
template <typename T, typename Controller> auto& required(Controller& para, std::string const& key) {
  return para.template insert<T>(key).template add<numsim_core::is_required>();
}

/// Run every check of `para`. One invalid_argument names every missing required parameter, every parameter stored
/// with another type than declared and every other failed check, each group sorted.
template <typename Controller, typename Handler> void check_parameters(Controller const& para, Handler& handler) {
  std::vector<std::string> missing, wrong_type, invalid;
  bool hint{false};
  for (auto const& [key, check] : para) {
    if (handler.contains(key) && std::type_index{handler.data(key).type()} != check->type_id()) {
      wrong_type.push_back("parameter '" + key + "' is not " + detail::parameter_type_label(check->type_id()) +
                           " as this material reads it");
      hint = hint || detail::whole_for_real(handler.data(key).type(), check->type_id());
      continue;
    }
    try {
      check->check_parameter(handler);
    } catch (std::invalid_argument const& e) {
      if (handler.contains(key))
        invalid.emplace_back(e.what());
      else
        missing.push_back(key);
    }
  }
  if (missing.empty() && wrong_type.empty() && invalid.empty())
    return;
  std::ranges::sort(missing);
  std::ranges::sort(wrong_type);
  std::ranges::sort(invalid);
  std::string msg;
  if (!missing.empty()) {
    msg = "missing " + std::to_string(missing.size()) + " required parameter(s):";
    for (auto const& key : missing)
      msg += " '" + key + "'";
  }
  for (auto const& problem : wrong_type)
    msg += (msg.empty() ? "" : "; ") + problem;
  if (hint)
    msg += " (a whole number written without a decimal point is the usual cause)";
  for (auto const& problem : invalid)
    msg += (msg.empty() ? "" : "; ") + problem;
  throw std::invalid_argument(msg);
}

/// Declare an optional parameter `key` of type T with a default.
template <typename T, typename Controller> auto& optional(Controller& para, std::string const& key, T value) {
  return para.template insert<T>(key).template add<numsim_core::set_default>(std::move(value));
}

}  // namespace numsim::materials

#endif  // NUMSIM_MATERIALS_PARAMETER_HELPERS_H
