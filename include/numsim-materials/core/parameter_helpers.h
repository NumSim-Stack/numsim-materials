#ifndef NUMSIM_MATERIALS_PARAMETER_HELPERS_H
#define NUMSIM_MATERIALS_PARAMETER_HELPERS_H

#include <numsim-core/input_parameter_controller.h>

#include <string>
#include <utility>

namespace numsim::materials {

/// Declare a required parameter `key` of type T.
template <typename T, typename Controller> auto& required(Controller& para, std::string const& key) {
  return para.template insert<T>(key).template add<numsim_core::is_required>();
}

/// Declare an optional parameter `key` of type T with a default.
template <typename T, typename Controller> auto& optional(Controller& para, std::string const& key, T value) {
  return para.template insert<T>(key).template add<numsim_core::set_default>(std::move(value));
}

}  // namespace numsim::materials

#endif  // NUMSIM_MATERIALS_PARAMETER_HELPERS_H
