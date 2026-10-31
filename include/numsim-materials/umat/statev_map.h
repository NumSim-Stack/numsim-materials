#ifndef STATEV_MAP_H
#define STATEV_MAP_H

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/history_slot.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/umat/errors.h"
#include "numsim-materials/umat/tensor_conversion.h"

/// Flat STATEV <-> the context's history properties.
///
/// Under a UMAT the host owns all persistent state: it hands STATEV in at the
/// start of every call and takes it back at the end. The material graph keeps
/// its state in `history_property` objects, so this class is the translation
/// between the two.
///
/// The mapping is built once, at construction, and is then a pure index
/// operation.
namespace numsim::materials::umat {

/// Convention for tensor-valued state variables in STATEV.
///
/// Rank-2 state variables are stored with ENGINEERING shear — off-diagonals
/// doubled — matching STRAN, DSTRAN and Abaqus's own PE/LE output. The library
/// would round-trip either convention exactly, so this is invisible internally
/// and only matters at the host boundary, which is precisely why it has to
/// match the host: an SDV compared against PE, or an initial plastic state
/// seeded through *INITIAL CONDITIONS, TYPE=SOLUTION, would otherwise be out by
/// a factor of two in shear, silently.
///
/// Each history is stored and rotated by the codec it was declared with
/// (core/history_codec.h), else by its value type's default_codec.
///
/// One `(material, property)` pair the host owns and STATEV must NOT contain.
using statev_exclusion = std::pair<std::string, std::string>;

template <typename Traits>
class statev_map {
public:
  using value_type = typename Traits::value_type;
  using tensor2 = tmech::tensor<value_type, 3, 2>;

  /// Build the map over every history property in @p ctx except those named in
  /// @p exclusions.
  ///
  /// Every exclusion MUST match an existing history property; an unmatched
  /// entry throws. A typo or a renamed property would otherwise silently leave
  /// a host-driven quantity in STATEV, shifting every subsequent slot — a
  /// corruption that shows up as wrong results far from its cause.
  statev_map(material_context<Traits>& ctx,
             const std::vector<statev_exclusion>& exclusions) {
    std::vector<bool> matched(exclusions.size(), false);

    // Collect first, then sort, so the STATEV layout does NOT depend on the
    // graph's execution order. That order comes from a topological sort seeded
    // by unordered_map iteration, which is not guaranteed stable across builds
    // or compilers — and a STATEV layout that moves between builds would
    // silently reinterpret every existing restart file. Sorting by
    // (owner, property) makes the layout deterministic and predictable enough
    // for a user to read STATEV by hand.
    std::vector<property_base*> history;
    for (auto* prop : ctx.property_execution_order()) {
      if (!prop->is_history()) continue;

      const auto& id = prop->traits().id;
      bool excluded = false;
      for (std::size_t e = 0; e < exclusions.size(); ++e) {
        if (exclusions[e].first == id.owner && exclusions[e].second == id.name) {
          matched[e] = true;
          excluded = true;
        }
      }
      if (!excluded) history.push_back(prop);
    }

    for (std::size_t e = 0; e < exclusions.size(); ++e)
      if (!matched[e])
        throw fatal_error(
            "statev_map: exclusion '" + exclusions[e].first +
            "::" + exclusions[e].second +
            "' does not name any history property — check for a typo or a "
            "renamed property; leaving it unmatched would place host-owned "
            "state into STATEV and shift every following slot");

    std::sort(history.begin(), history.end(),
              [](const property_base* a, const property_base* b) {
                const auto& ia = a->traits().id;
                const auto& ib = b->traits().id;
                return ia.owner != ib.owner ? ia.owner < ib.owner
                                            : ia.name < ib.name;
              });

    for (auto* prop : history) add_entry(ctx, prop);
  }

  /// Number of doubles this map occupies in STATEV.
  [[nodiscard]] std::size_t nstatv() const noexcept { return m_width; }

  /// Validate the host's NSTATV against what the configured materials need.
  void check_nstatv(std::size_t host_nstatv) const {
    if (host_nstatv < m_width)
      throw fatal_error(
          "statev_map: the host provides NSTATV=" +
          std::to_string(host_nstatv) + " but the configured materials need " +
          std::to_string(m_width) + " — increase *DEPVAR in the input deck");
  }

  /// Load STATEV into the history properties, setting BOTH sides.
  ///
  /// Both, because STATEV holds the converged state at t_n: `old` is that
  /// state, and `new` starts there too so an unconverged or elastic step that
  /// never writes `new` still reports t_n rather than a stale iterate.
  void unpack(const value_type* statev) const {
    for (const auto& e : m_entries) e.slot->unpack(statev + e.offset);
  }

  /// Store the updated history (the `new` side) back into STATEV.
  void pack(value_type* statev) const {
    for (const auto& e : m_entries) e.slot->pack(statev + e.offset);
  }

  /// Rotate every tensor-valued state variable into the current configuration.
  ///
  /// Under NLGEOM=YES Abaqus rotates STRESS and STRAN to the end-of-increment
  /// frame before calling the UMAT, but it does NOT touch STATEV — rotating
  /// user state variables is the UMAT's job. Skipping it leaves tensor state
  /// (plastic strain, back stress) in a stale frame, and the error accumulates
  /// silently over the rotation history with no diagnostic anywhere.
  ///
  /// Scalars are frame-indifferent and untouched, and so is reference-frame
  /// history (Cp, Fp): only a codec with a rotate() turns its history.
  void rotate_history(const tensor2& R) const {
    for (const auto& e : m_entries) e.slot->rotate(R);
  }

  /// Whether this map holds any state that a rotation would affect.
  [[nodiscard]] bool has_rotatable_history() const noexcept {
    return std::ranges::any_of(m_entries, [](const auto& e) { return e.slot->rotates(); });
  }

  /// Human-readable layout, for diagnostics and for documenting *DEPVAR.
  [[nodiscard]] std::vector<std::string> describe() const {
    std::vector<std::string> out;
    out.reserve(m_entries.size());
    for (const auto& e : m_entries) {
      const std::string tag{e.slot->tag()};
      out.push_back(std::to_string(e.offset) + ".." +
                    std::to_string(e.offset + e.slot->width() - 1) + "  " + e.owner +
                    "::" + e.name + (tag.empty() ? "" : "  " + tag));
    }
    return out;
  }

private:
  struct entry {
    std::string owner;
    std::string name;
    std::size_t offset;
    const history_slot<value_type>* slot;
  };

  void add_entry(const material_context<Traits>& ctx, const property_base* prop) {
    const auto& id = prop->traits().id;
    const auto* owner = ctx.find(id.owner);
    const auto* slot = owner ? owner->history_slot_of(prop) : nullptr;
    if (!slot)
      throw fatal_error(
          "statev_map: history property '" + id.owner + "::" + id.name +
          "' has a type that cannot be stored in STATEV — declare it with a "
          "codec (core/history_codec.h)");
    m_entries.push_back({id.owner, id.name, m_width, slot});
    m_width += slot->width();
  }

  std::vector<entry> m_entries;
  std::size_t m_width{0};
};

}  // namespace numsim::materials::umat

#endif  // STATEV_MAP_H
