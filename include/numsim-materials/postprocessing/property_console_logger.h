#ifndef PROPERTY_CONSOLE_LOGGER_H
#define PROPERTY_CONSOLE_LOGGER_H

#include <iomanip>
#include <iostream>
#include "numsim-materials/core/material_base.h"

namespace numsim::materials {

template <typename Traits>
class property_console_logger
    : public material_base<property_console_logger<Traits>, Traits> {
public:
  using base = material_base<property_console_logger<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using class_to_prop_map =
      std::unordered_map<std::string, std::vector<std::string>>;

  template <typename... Args>
  property_console_logger(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_class_to_property(
            base::template get_parameter<class_to_prop_map>("input")),
        m_order(
            base::template get_parameter<std::vector<std::string>>("order"))
  {
    for (const auto& [class_name, properties] : m_class_to_property) {
      auto& vec{m_inputs[class_name]};
      vec.reserve(properties.size());
      for (const auto& prop_name : properties) {
        auto& input = base::template add_input<value_type>(
            class_name, prop_name, EdgeKind::Global);
        vec.push_back({prop_name, &input});
      }
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<class_to_prop_map>("input")
        .template add<is_required>();
    para.template insert<std::vector<std::string>>("order")
        .template add<set_default>(std::vector<std::string>{});
    return para;
  }

  void log() {
    if (m_order.empty()) {
      for (const auto& [class_name, data] : m_inputs)
        print_data(class_name, data);
      std::cout << std::endl;
    } else {
      for (const auto& next : m_order) {
        auto pos{m_inputs.find(next)};
        if (pos != m_inputs.end())
          print_data(pos->first, pos->second);
      }
      std::cout << std::endl;
    }
  }

protected:
  using input_entry = std::pair<std::string, const input_property<value_type, property_traits>*>;

  void print_data(std::string const& class_name, std::vector<input_entry> const& data) {
    std::cout << "Class: \033[33m" << class_name << "\033[0m" << std::endl;
    std::cout << "+-----------------+--------------------+" << std::endl;
    std::cout << "| Property Name   | Value              |" << std::endl;
    std::cout << "+-----------------+--------------------+" << std::endl;
    for (const auto& [prop_name, input] : data) {
      std::cout << "| \033[32m" << std::setw(15) << std::left << prop_name << "\033[0m"
                << " | \033[31m" << std::setw(18) << std::left;
      if (input->is_wired())
        std::cout << input->get();
      else
        std::cout << "(unwired)";
      std::cout << "\033[0m |" << std::endl;
    }
    std::cout << "+-----------------+--------------------+" << std::endl;
  }

  const class_to_prop_map& m_class_to_property;
  const std::vector<std::string>& m_order;
  std::unordered_map<std::string, std::vector<input_entry>> m_inputs;
};

}

#endif // PROPERTY_CONSOLE_LOGGER_H
