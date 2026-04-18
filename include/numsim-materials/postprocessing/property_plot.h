#ifndef NUMSIM_MATERIALS_PROPERTY_PLOT_H
#define NUMSIM_MATERIALS_PROPERTY_PLOT_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/postprocessing/plot_backend.h"

namespace numsim::materials {

/// Postprocessor material that plots consumed properties in real time.
///
/// Parameters:
///   "name":    material name
///   "x_source": "material::property" string for x-axis
///   "y_sources": map of { series_name: "material::property" } for y-axes
///   "backend": pointer to a plot_backend (set via set_backend() after construction)
///
/// Usage:
///   auto backend = std::make_shared<qcustomplot_backend>("Curing");
///   parameter.insert<std::string>("name", "plotter");
///   parameter.insert<std::string>("x_source", "time::state");
///   parameter.insert<std::vector<std::string>>("y_sources", {"curing::current_state"});
///   auto& plotter = ctx.create<property_plot<P>>(parameter);
///   plotter.set_backend(backend);
///   ctx.finalize();  // wires inputs
///   // plot updates live during ctx.update()
template <typename Traits>
class property_plot final
    : public material_base<property_plot<Traits>, Traits> {
public:
  using base = material_base<property_plot<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  property_plot(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_x_source_str(base::template get_parameter<std::string>("x_source")),
        m_y_source_strs(base::template get_parameter<std::vector<std::string>>("y_sources"))
  {
    // Register a dummy output so the property engine calls our update()
    base::template add_output<int>("_plot_tick", &property_plot::update);

    // Parse x-axis source
    auto x_src = connection_source::parse(m_x_source_str);
    m_x_input = &base::template add_input<value_type>(
        x_src.material, x_src.property, EdgeKind::Global);

    // Parse y-axis sources
    for (const auto& src_str : m_y_source_strs) {
      auto src = connection_source::parse(src_str);
      auto& input = base::template add_input<value_type>(
          src.material, src.property, EdgeKind::Global);
      m_y_inputs.push_back({src_str, &input});
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("x_source").template add<is_required>();
    para.template insert<std::vector<std::string>>("y_sources").template add<is_required>();
    return para;
  }

  /// Set the plotting backend. Must be called before finalize().
  void set_backend(std::shared_ptr<plot_backend> backend) {
    m_backend = std::move(backend);
  }

  void update() override {
    if (!m_backend) return;

    // Initialize series on first call
    if (!m_initialized) {
      for (const auto& [name, _] : m_y_inputs)
        m_series_ids.push_back(m_backend->add_series(name));
      m_backend->show();
      m_initialized = true;
    }

    // Append data
    double x = m_x_input->get();
    for (std::size_t i = 0; i < m_y_inputs.size(); ++i)
      m_backend->append(m_series_ids[i], x, m_y_inputs[i].second->get());

    m_backend->refresh();
  }

private:
  const std::string& m_x_source_str;
  const std::vector<std::string>& m_y_source_strs;

  const input_property<value_type, property_traits>* m_x_input{nullptr};
  std::vector<std::pair<std::string, const input_property<value_type, property_traits>*>> m_y_inputs;

  std::shared_ptr<plot_backend> m_backend;
  std::vector<int> m_series_ids;
  bool m_initialized{false};
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_PROPERTY_PLOT_H
