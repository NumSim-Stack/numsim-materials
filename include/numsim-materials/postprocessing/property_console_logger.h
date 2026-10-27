#ifndef PROPERTY_CONSOLE_LOGGER_H
#define PROPERTY_CONSOLE_LOGGER_H

#include <print>
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
        m_tick(base::template add_output<value_type>("tick")),
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
    base::add_callback(&property_console_logger::log, m_tick).reads_all_inputs().stateful(); // prints on every update
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
      std::print("\n");
    } else {
      for (const auto& next : m_order) {
        auto pos{m_inputs.find(next)};
        if (pos != m_inputs.end())
          print_data(pos->first, pos->second);
      }
      std::print("\n");
    }
  }

protected:
  using input_entry = std::pair<std::string, const input_property<value_type, property_traits>*>;

  void print_data(std::string const& class_name, std::vector<input_entry> const& data) {
    constexpr std::string_view rule{"+-----------------+--------------------+"};
    std::println("Class: \033[33m{}\033[0m", class_name);
    std::println("{}\n| Property Name   | Value              |\n{}", rule, rule);
    for (const auto& [prop_name, input] : data)
      std::println("| \033[32m{:<15}\033[0m | \033[31m{:<18}\033[0m |", prop_name,
                   input->is_wired() ? std::format("{:.6g}", input->get()) : std::string{"(unwired)"});
    std::println("{}", rule);
  }

  value_type& m_tick; ///< the output the logging callback is bound to
  const class_to_prop_map& m_class_to_property;
  const std::vector<std::string>& m_order;
  std::unordered_map<std::string, std::vector<input_entry>> m_inputs;
};

}

#endif // PROPERTY_CONSOLE_LOGGER_H
