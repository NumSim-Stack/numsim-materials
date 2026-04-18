#ifndef NUMSIM_MATERIALS_TRAITS_H
#define NUMSIM_MATERIALS_TRAITS_H

#include <any>
#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include <numsim-core/input_parameter_controller.h>
#include <numsim-core/wrapper.h>
#include <numsim-core/query_map.h>
#include <numsim-core/parameter_handler.h>

#include "numsim-materials/core/property_registry.h"

namespace numsim::materials {

using numsim_core::is_required;
using numsim_core::set_default;
using numsim_core::cwrapper;
using numsim_core::wrapper;

using scalar_or_property = std::variant<double, std::pair<std::string, std::string>>;

using input_parameters = std::variant<double, float, int,
                                      std::string,
                                      std::vector<std::string>,
                                      std::vector<double>,
                                      std::vector<std::size_t>,
                                      std::vector<scalar_or_property>,
                                      scalar_or_property,
                                      std::unordered_map<std::string, std::vector<std::string>>,
                                      std::vector<std::pair<std::string, std::string>>>;

struct material_policy_default {
  using PropertyHandler = property_registry;
  using ParameterHandler = numsim_core::parameter_handler<>;
  using MaterialHandler =
      numsim_core::query_map<std::tuple<std::string>, std::unordered_map, std::any>;
  using InputParameterController =
      numsim_core::input_parameter_controller<std::string, ParameterHandler, input_parameters>;
  using value_type = double;
  static constexpr std::size_t Dim = 3;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_TRAITS_H
