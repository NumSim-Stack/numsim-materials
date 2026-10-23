#ifndef NUMSIM_MATERIALS_DRIVER_MATERIAL_POINT_H
#define NUMSIM_MATERIALS_DRIVER_MATERIAL_POINT_H

#include <algorithm>
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
#include "numsim-materials/core/property_value.h"
#include "numsim-materials/materials/props_scalar.h"
#include "numsim-materials/umat/errors.h"
#include "numsim-materials/umat/statev_map.h"

/// One material point over a finalized material_context: input variables, output variables and history
/// management. Step control (Newton, cutbacks, accepting a step) belongs to the host's adapter.
///
/// The adapter owns the history in one of two ways, by the calls it makes:
///  - host-owned (UMAT, MOOSE): reset(), load_history(), bind the inputs, evaluate(), store_history(); never accept().
///  - engine-owned (report, standalone): reset(), bind the inputs, evaluate(), then accept() or reset(); never load_history().
namespace numsim::materials {

template <typename Traits>
class material_point {
public:
  using value_type = typename Traits::value_type;
  using context_type = material_context<Traits>;
  using tensor2 = tmech::tensor<value_type, 3, 2>;

  struct config {
    /// Host-driven source materials (external_*_source); their history stays with the host.
    std::vector<std::string> inputs{};
    /// Further history kept out of load_history() / store_history(), as (material, property).
    std::vector<umat::statev_exclusion> host_owned{};
    /// Optional "material::property" of a residual: evaluate() updates again until it is at most the tolerance.
    std::string residual_property{};
    value_type residual_tolerance{1e-10};
    int max_passes{100};
  };

  material_point(context_type &ctx, config cfg) : m_ctx(ctx), m_cfg(std::move(cfg)) {
    if (!m_ctx.is_finalized())
      throw umat::fatal_error("material_point: the context must be finalized first");
    for (auto const &name : m_cfg.inputs) {
      if (!m_ctx.find(name))
        throw umat::fatal_error("material_point: input '" + name + "' is not a material in this context");
      for (auto const *prop : m_ctx.property_execution_order())
        if (prop->is_history() && prop->traits().id.owner == name)
          m_exclusions.emplace_back(name, prop->traits().id.name);
    }
    m_exclusions.insert(m_exclusions.end(), m_cfg.host_owned.begin(), m_cfg.host_owned.end());
    if (!m_cfg.residual_property.empty())
      m_residual = &output<value_type>(m_cfg.residual_property);
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
    if (!std::ranges::contains(m_cfg.inputs, name))
      throw umat::fatal_error("material_point: '" + name + "' is not an input of this material point");
    auto *source = dynamic_cast<Source *>(m_ctx.find(name));
    if (!source)
      throw umat::fatal_error("material_point: input '" + name + "' is another kind of source");
    return *source;
  }

  /// Copy the host's material constants into the props_scalar readers; a no-op without any.
  void bind_props(std::span<value_type const> props) {
    if (m_props_readers.empty())
      return;
    if (props.size() < m_props_needed)
      throw umat::fatal_error("material_point: the model reads " + std::to_string(m_props_needed) +
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
    throw umat::fatal_error("material_point: property '" + material + "::" + property + "' has an unexpected type");
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
    throw umat::fatal_error("material_point: property '" + material + "::" + property +
                            "' is not a history property of this type");
  }

  // --- Evaluation ---

  /// One graph update, repeated to the fixed point when a residual is configured.
  void evaluate() {
    if (!m_props_readers.empty() && !m_props_bound)
      throw umat::fatal_error("material_point: this model reads its material constants per call — call bind_props() "
                              "before evaluating");
    m_ctx.update();
    if (!m_residual)
      return;
    for (int pass{1}; *m_residual > m_cfg.residual_tolerance; ++pass) {
      if (pass >= m_cfg.max_passes)
        throw std::runtime_error(std::format("material_point: '{}' did not reach {:g} in {} passes (last {:g})",
                                             m_cfg.residual_property, m_cfg.residual_tolerance, m_cfg.max_passes,
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
  void load_history(value_type const *buffer) const { host_map().unpack(buffer); }

  /// Host-owned: the evaluated history back to the host.
  void store_history(value_type *buffer) const { host_map().pack(buffer); }

  /// Host-owned: tensor history into the current configuration (after load_history()).
  void rotate_history(tensor2 const &R) const { host_map().rotate_history(R); }

  [[nodiscard]] bool has_rotatable_history() const { return host_map().has_rotatable_history(); }

  /// Engine-owned checkpoint of every history output.
  void save(std::ostream &out) const { m_ctx.save_state(out); }
  void load(std::istream &in) { m_ctx.load_state(in); }

  [[nodiscard]] context_type &context() const noexcept { return m_ctx; }

private:
  property_base *find(std::string const &material, std::string const &property) const {
    if (auto *prop = m_ctx.find_property(material, property))
      return prop;
    throw umat::fatal_error("material_point: property '" + material + "::" + property + "' not found");
  }

  /// Built on first use: an engine-owned point may hold history no host buffer can.
  umat::statev_map<Traits> const &host_map() const {
    if (!m_host_map)
      m_host_map = std::make_unique<umat::statev_map<Traits>>(m_ctx, m_exclusions);
    return *m_host_map;
  }

  context_type &m_ctx;
  config m_cfg;
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
