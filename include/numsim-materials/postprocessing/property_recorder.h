#ifndef NUMSIM_MATERIALS_PROPERTY_RECORDER_H
#define NUMSIM_MATERIALS_PROPERTY_RECORDER_H

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tmech/tmech.h>

#include "numsim-materials/core/material_base.h"
#include "numsim-materials/io/record_buffer.h"

namespace numsim::materials {

/// Post-processor material that BUFFERS consumed properties to memory over a run
/// and serializes them to a file. The recording analog of `property_plot`: same
/// wiring (named `"material::property"` sources, a dummy output so the engine
/// calls `update()` each step, one `add_input` per source), but instead of
/// pushing to a live plot it accumulates a `record_buffer` the caller writes out
/// with any `output_writer` (CSV, VTK, …) after the run.
///
/// Parameters:
///   "name":           material name
///   "scalar_sources": list of "material::property" scalar sources (optional)
///   "tensor_sources": list of "material::property" rank-2 tensor sources
///                     (optional) — flattened to Dim*Dim component columns
///                     `<name>_ij` (full storage; symmetric-only reduction is a
///                     later option).
///
/// Usage:
///   param.insert<std::string>("name", "recorder");
///   param.insert<std::vector<std::string>>("scalar_sources", {"solver::dgamma"});
///   param.insert<std::vector<std::string>>("tensor_sources", {"J2::stress"});
///   auto& rec = ctx.create<property_recorder<P>>(param);
///   ctx.finalize();
///   for (...) ctx.update();                 // one row buffered per step
///   rec.write(csv_writer{}, "history.csv");  // or vtk_timeseries_writer{}
template <typename Traits>
class property_recorder final
    : public material_base<property_recorder<Traits>, Traits> {
public:
  using base = material_base<property_recorder<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  static constexpr std::size_t Dim = base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;

  template <typename... Args>
  property_recorder(Args&&... args) : base(std::forward<Args>(args)...) {
    // The source lists are only needed to build the inputs + columns here, so
    // bind them as ctor locals (they live in the material's own copied
    // parameter_handler; keeping long-lived reference members buys nothing).
    auto const& scalar_srcs =
        base::template get_parameter<std::vector<std::string>>("scalar_sources");
    auto const& tensor_srcs =
        base::template get_parameter<std::vector<std::string>>("tensor_sources");

    if (scalar_srcs.empty() && tensor_srcs.empty())
      throw std::runtime_error(
          "property_recorder '" + base::name() +
          "': at least one of scalar_sources / tensor_sources must be set.");

    // Drive update() each step via a dummy output. The property engine evaluates
    // EVERY property in topological order (no consumed-only / dead-code pruning),
    // so this consumer-less output's callback fires once per ctx.update() — the
    // invariant the whole recorder (and property_plot) relies on.
    base::template add_output<int>("_record_tick", &property_recorder::update);

    // Scalar sources → one column each.
    for (auto const& s : scalar_srcs) {
      auto src = connection_source::parse(s);
      m_scalar_inputs.push_back(
          &base::template add_input<value_type>(src.material, src.property,
                                                EdgeKind::Global));
      m_buffer.declare_column(sanitize(s));
    }
    // Tensor sources → Dim*Dim component columns `<name>_ij`.
    for (auto const& s : tensor_srcs) {
      auto src = connection_source::parse(s);
      m_tensor_inputs.push_back(
          &base::template add_input<tensor2>(src.material, src.property,
                                             EdgeKind::Global));
      auto const stem = sanitize(s);
      for (std::size_t i = 0; i < Dim; ++i)
        for (std::size_t j = 0; j < Dim; ++j)
          m_buffer.declare_column(stem + "_" + std::to_string(i) +
                                  std::to_string(j));
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::vector<std::string>>("scalar_sources")
        .template add<set_default>(std::vector<std::string>{});
    para.template insert<std::vector<std::string>>("tensor_sources")
        .template add<set_default>(std::vector<std::string>{});
    return para;
  }

  /// Append one row: the current value of every source. Called by the engine.
  void update() override {
    std::size_t c = 0;
    for (auto const* in : m_scalar_inputs)
      m_buffer.push(c++, static_cast<double>(in->get()));
    for (auto const* in : m_tensor_inputs) {
      auto const t = in->get();
      for (std::size_t i = 0; i < Dim; ++i)
        for (std::size_t j = 0; j < Dim; ++j)
          m_buffer.push(c++, static_cast<double>(t(i, j)));
    }
  }

  /// The accumulated data (rows == number of update() calls).
  [[nodiscard]] record_buffer const& buffer() const noexcept { return m_buffer; }

  /// Serialize the buffer to `path` with the given writer.
  void write(output_writer const& writer, std::string const& path) const {
    std::ofstream os(path);
    if (!os)
      throw std::runtime_error("property_recorder '" + base::name() +
                               "': cannot open '" + path + "' for writing.");
    writer.write(m_buffer, os);
    os.flush();
    if (!os) // disk-full / write error would otherwise leave a silent truncation
      throw std::runtime_error("property_recorder '" + base::name() +
                               "': write to '" + path + "' failed.");
  }

private:
  /// A "material::property" source is not a valid CSV/VTK array name; map each
  /// run of non-alphanumeric characters to a single '_' (so `mat::prop` becomes
  /// `mat_prop`, not `mat__prop`) and trim a leading/trailing '_'.
  static std::string sanitize(std::string const& s) {
    auto is_alnum = [](char c) {
      return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
             (c >= '0' && c <= '9');
    };
    std::string out;
    out.reserve(s.size());
    bool prev_us = false;
    for (char ch : s) {
      if (is_alnum(ch)) {
        out.push_back(ch);
        prev_us = false;
      } else if (!prev_us && !out.empty()) {
        out.push_back('_');
        prev_us = true;
      }
    }
    while (!out.empty() && out.back() == '_')
      out.pop_back();
    return out;
  }

  std::vector<const input_property<value_type, property_traits>*> m_scalar_inputs;
  std::vector<const input_property<tensor2, property_traits>*> m_tensor_inputs;
  record_buffer m_buffer;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_PROPERTY_RECORDER_H
