#ifndef NUMSIM_MATERIALS_PARAMETER_HELPERS_H
#define NUMSIM_MATERIALS_PARAMETER_HELPERS_H

#include <numsim-core/input_parameter_controller.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace numsim::materials {

/// Declare a required parameter `key` of type T; a value of another type is rejected at construction.
template <typename T, typename Controller> auto& required(Controller& para, std::string const& key) {
  return para.template insert<T>(key).template add<numsim_core::is_required>().template add<numsim_core::check_data_type>();
}

/// Run every check of `para`; all missing required parameters are named, sorted, in one invalid_argument.
template <typename Controller, typename Handler> void check_parameters(Controller const& para, Handler& handler) {
  std::vector<std::string> missing;
  for (auto const& [key, check] : para) {
    try {
      check->check_parameter(handler);
    } catch (std::invalid_argument const&) {
      if (handler.contains(key))
        throw;
      missing.push_back(key);
    }
  }
  if (missing.empty())
    return;
  std::ranges::sort(missing);
  std::string msg{"missing " + std::to_string(missing.size()) + " required parameter(s):"};
  for (auto const& key : missing)
    msg += " '" + key + "'";
  throw std::invalid_argument(msg);
}

/// Declare an optional parameter `key` of type T with a default; a value of another type is rejected at construction.
template <typename T, typename Controller> auto& optional(Controller& para, std::string const& key, T value) {
  return para.template insert<T>(key)
      .template add<numsim_core::set_default>(std::move(value))
      .template add<numsim_core::check_data_type>();
}

}  // namespace numsim::materials

#endif  // NUMSIM_MATERIALS_PARAMETER_HELPERS_H
