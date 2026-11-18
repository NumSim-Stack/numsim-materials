#ifndef NUMSIM_MATERIALS_DRIVER_MATERIAL_POINT_H
#define NUMSIM_MATERIALS_DRIVER_MATERIAL_POINT_H

#include <algorithm>
#include <cmath>
#include <format>
#include <istream>
#include <memory>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/property_value.h"
#include "numsim-materials/materials/props_scalar.h"
#include "numsim-materials/core/errors.h"
#include "numsim-materials/umat/statev_map.h"

/// One material point over a finalized material_context: input variables, output variables and history
/// management. Step control (Newton, cutbacks, accepting a step) belongs to the host's adapter.
///
/// The adapter owns the history in one of two ways, by the calls it makes:
///  - host-owned (UMAT, MOOSE): reset(), load_history(), bind the inputs, evaluate(), store_history(); never accept().
///  - engine-owned (report, standalone): reset(), bind the inputs, evaluate(), then accept() or reset(); never load_history().
/// The load sources are history too: after accept() their old value is the accepted load.
namespace numsim::materials {

template <typename Traits>
class material_point {
public:
  using value_type = typename Traits::value_type;
  using context_type = material_context<Traits>;
  using tensor2 = tmech::tensor<value_type, 3, 2>;
  using parameter_handler = typename Traits::ParameterHandler;
  using input_parameter_controller = typename Traits::InputParameterController;

  static input_parameter_controller parameters() {
    input_parameter_controller para;
    // host-driven sources (external_*_source); their history stays with the host
    optional<std::vector<std::string>>(para, "inputs", {});
    // further "material::property" history kept out of load_history() / store_history()
    optional<std::vector<std::string>>(para, "host_owned", {});
    // deprecated, removed later: "material::property" evaluate() updates again until its magnitude is at most the
    // tolerance; a material declaring converges() is converged after the one update
    optional<std::string>(para, "residual_source", std::string{});
    optional<value_type>(para, "residual_tolerance", value_type{1e-10});
    optional<int>(para, "max_passes", 100);
    return para;
  }

  material_point(context_type &ctx, parameter_handler parameters)
      : m_ctx(ctx), m_parameters(checked(std::move(parameters))),
        m_inputs(m_parameters.template get<std::vector<std::string>>("inputs")),
        m_tolerance(m_parameters.template get<value_type>("residual_tolerance")),
        m_max_passes(m_parameters.template get<int>("max_passes")) {
    if (!m_ctx.is_finalized())
      throw fatal_error("material_point: the context must be finalized first");
    for (auto const &name : m_inputs) {
      if (!m_ctx.find(name))
        throw fatal_error("material_point: input '" + name + "' is not a material in this context");
      for (auto const *prop : m_ctx.property_execution_order())
        if (prop->is_history() && prop->traits().id.owner == name)
          m_exclusions.emplace_back(name, prop->traits().id.name);
    }
    for (auto const &source : m_parameters.template get<std::vector<std::string>>("host_owned")) {
      auto const src{connection_source::parse(source)};
      m_exclusions.emplace_back(src.material, src.property);
    }
    if (!(m_tolerance >= 0) || m_max_passes < 1)
      throw fatal_error("material_point: residual_tolerance must be >= 0 and max_passes >= 1");
    if (auto const &source{m_parameters.template get<std::string>("residual_source")}; !source.empty())
      m_residual = &output<value_type>(source);
    for (auto *material : m_ctx.materials())
      if (auto *reader = dynamic_cast<props_scalar<Traits> *>(material)) {
        m_props_readers.push_back(reader);
        m_props_needed = std::max(m_props_needed, reader->index() + 1);
      }
    m_live_slots.assign(m_props_needed, false);
    for (auto const *reader : m_props_readers)
      m_live_slots[reader->index()] = true;
  }

  // --- Inputs ---

  /// The input source `name`, to bind its old and new values.
  template <typename Source>
  Source &input(std::string const &name) const {
    if (!std::ranges::contains(m_inputs, name))
      throw fatal_error("material_point: '" + name + "' is not an input of this material point");
    auto *source = dynamic_cast<Source *>(m_ctx.find(name));
    if (!source)
      throw fatal_error("material_point: input '" + name + "' is another kind of source");
    return *source;
  }

  /// Copy the host's material constants into the props_scalar readers; a no-op without any.
  void bind_props(std::span<value_type const> props) {
    if (m_props_readers.empty())
      return;
    if (props.size() < m_props_needed)
      throw fatal_error("material_point: the model reads " + std::to_string(m_props_needed) +
                              " host constants but this call supplied " + std::to_string(props.size()) +
                              " — check *USER MATERIAL, CONSTANTS=");
    for (auto *reader : m_props_readers)
      reader->bind(props);
    m_props_bound = true;
  }

  /// True when any constant is read per call rather than baked into the graph.
  [[nodiscard]] bool has_live_props() const noexcept { return !m_props_readers.empty(); }

  /// True when host constant `slot` is read per call.
  [[nodiscard]] bool is_live_prop(std::size_t slot) const noexcept { return slot < m_live_slots.size() && m_live_slots[slot]; }

  // --- Outputs ---

  template <typename T>
  T const &output(std::string const &material, std::string const &property) const {
    if (auto const *value = property_value<T>(find(material, property)))
      return *value;
    throw fatal_error("material_point: property '" + material + "::" + property + "' has an unexpected type");
  }

  /// output() of "material::property".
  template <typename T>
  T const &output(std::string const &source) const {
    auto const src{connection_source::parse(source)};
    return output<T>(src.material, src.property);
  }

  /// A history output, for both ends of the step.
  template <typename T>
  history_property<T> const &history(std::string const &material, std::string const &property) const {
    if (auto const *hist = dynamic_cast<history_property<T> const *>(find(material, property)))
      return *hist;
    throw fatal_error("material_point: property '" + material + "::" + property +
                            "' is not a history property of this type");
  }

  // --- Evaluation ---

  /// One graph update; with the deprecated residual_source, repeated until that residual is converged (a graph
  /// whose fixed point is declared with converges() is converged after the first).
  void evaluate() {
    if (!m_props_readers.empty() && !m_props_bound)
      throw fatal_error("material_point: this model reads its material constants per call — call bind_props() "
                              "before evaluating");
    m_ctx.update();
    if (!m_residual)
      return;
    // the negated test also keeps a NaN residual unconverged
    for (int pass{1}; !(std::abs(*m_residual) <= m_tolerance); ++pass) {
      if (pass >= m_max_passes || std::isnan(*m_residual))
        throw cutback_error(std::format("material_point: '{}' did not reach {:g} in {} passes (last {:g})",
                                              m_parameters.template get<std::string>("residual_source"), m_tolerance,
                                              m_max_passes,
                                              *m_residual));
      m_ctx.update();
    }
  }

  // --- History ---

  /// History back to the start of the step; also clears a failed evaluation.
  void reset() { m_ctx.revert(); }

  /// Engine-owned: the evaluated step becomes the start of the next one.
  void accept() { m_ctx.commit(); }

  /// Host-owned: doubles the host stores per point.
  [[nodiscard]] std::size_t history_width() const { return host_map().nstatv(); }

  [[nodiscard]] std::vector<std::string> describe_history() const { return host_map().describe(); }

  /// Host-owned: the host's history at the start of the step, into both ends of every history output.
  void load_history(std::span<value_type const> buffer) const { host_map().unpack(checked(buffer).data()); }

  /// Host-owned: the evaluated history back to the host.
  void store_history(std::span<value_type> buffer) const { host_map().pack(checked(buffer).data()); }

  /// Host-owned: tensor history into the current configuration (after load_history()).
  void rotate_history(tensor2 const &R) const { host_map().rotate_history(R); }

  [[nodiscard]] bool has_rotatable_history() const { return host_map().has_rotatable_history(); }

  /// Engine-owned checkpoint: the accepted state, restored into a point over an identical context.
  void save(std::ostream &out) const { m_ctx.save_state(out); }
  void load(std::istream &in) { m_ctx.load_state(in); }

  [[nodiscard]] context_type &context() const noexcept { return m_ctx; }

private:
  static parameter_handler checked(parameter_handler parameters) {
    static auto const check{material_point::parameters()};
    check.check_parameter(parameters);
    return parameters;
  }

  property_base *find(std::string const &material, std::string const &property) const {
    if (auto *prop = m_ctx.find_property(material, property))
      return prop;
    throw fatal_error("material_point: property '" + material + "::" + property + "' not found");
  }

  template <typename Span>
  Span checked(Span buffer) const {
    if (buffer.size() < history_width())
      throw fatal_error(std::format("material_point: the host buffer holds {} values, the history needs {}",
                                          buffer.size(), history_width()));
    return buffer;
  }

  /// Built on first use: an engine-owned point may hold history no host buffer can.
  umat::statev_map<Traits> const &host_map() const {
    if (!m_host_map)
      m_host_map = std::make_unique<umat::statev_map<Traits>>(m_ctx, m_exclusions);
    return *m_host_map;
  }

  context_type &m_ctx;
  parameter_handler m_parameters;
  std::vector<std::string> m_inputs;
  value_type m_tolerance;
  int m_max_passes;
  std::vector<umat::statev_exclusion> m_exclusions;
  value_type const *m_residual{nullptr};
  std::vector<props_scalar<Traits> *> m_props_readers;
  std::vector<bool> m_live_slots;
  std::size_t m_props_needed{0};
  bool m_props_bound{false};
  mutable std::unique_ptr<umat::statev_map<Traits>> m_host_map;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DRIVER_MATERIAL_POINT_H
