#ifndef STATEV_MAP_H
#define STATEV_MAP_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tmech/tmech.h>
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
/// Note this assumes tensor-valued history is STRAIN-like, which is true of
/// every state variable the framework currently carries. A stress-like tensor
/// state variable (a back stress, say) would want the unscaled convention, and
/// would need this decision made per property rather than globally.
///
/// One `(material, property)` pair the host owns and STATEV must NOT contain.
using statev_exclusion = std::pair<std::string, std::string>;

template <typename Traits>
class statev_map {
public:
  using value_type = typename Traits::value_type;
  using tensor2 = tmech::tensor<value_type, 3, 2>;
  using tensor4 = tmech::tensor<value_type, 3, 4>;
  using hist_scalar = numsim_core::history_property<value_type, property_traits>;
  using hist_tensor2 = numsim_core::history_property<tensor2, property_traits>;
  using hist_tensor4 = numsim_core::history_property<tensor4, property_traits>;

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

    for (auto* prop : history) add_entry(prop);
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
    for (const auto& e : m_entries) e.unpack(statev + e.offset);
  }

  /// Store the updated history (the `new` side) back into STATEV.
  void pack(value_type* statev) const {
    for (const auto& e : m_entries) e.pack(statev + e.offset);
  }

  /// Rotate every tensor-valued state variable into the current configuration.
  ///
  /// Under NLGEOM=YES Abaqus rotates STRESS and STRAN to the end-of-increment
  /// frame before calling the UMAT, but it does NOT touch STATEV — rotating
  /// user state variables is the UMAT's job. Skipping it leaves tensor state
  /// (plastic strain, back stress) in a stale frame, and the error accumulates
  /// silently over the rotation history with no diagnostic anywhere.
  ///
  /// Scalars are frame-indifferent and untouched. Every tensor entry is
  /// rotated, which assumes tensor state is objective — true for the state
  /// variables constitutive models actually carry.
  void rotate_history(const tensor2& R) const {
    for (const auto& e : m_entries)
      if (e.rotate) e.rotate(R);
  }

  /// Whether this map holds any state that a rotation would affect.
  [[nodiscard]] bool has_rotatable_history() const noexcept {
    for (const auto& e : m_entries)
      if (e.rotate) return true;
    return false;
  }

  /// Human-readable layout, for diagnostics and for documenting *DEPVAR.
  [[nodiscard]] std::vector<std::string> describe() const {
    std::vector<std::string> out;
    out.reserve(m_entries.size());
    for (const auto& e : m_entries)
      out.push_back(std::to_string(e.offset) + ".." +
                    std::to_string(e.offset + e.width - 1) + "  " + e.owner +
                    "::" + e.name +
                    (e.width == canonical_width ? "  [engineering shear]" : ""));
    return out;
  }

private:
  struct entry {
    std::string owner;
    std::string name;
    std::size_t offset;
    std::size_t width;
    std::function<void(const value_type*)> unpack;
    std::function<void(value_type*)> pack;
    /// Empty for scalars, which no rotation affects.
    std::function<void(const tensor2&)> rotate;
  };

  /// A rank-2 history variable is stored in 6 slots, which can only represent a
  /// symmetric tensor. That is right for every state variable in practice
  /// (plastic strain, back stress), but silently wrong for one that is not —
  /// a deformation gradient, say. So the skew part is checked rather than
  /// assumed, and a violation is reported instead of quietly discarded.
  static void require_symmetric(const tensor2& t, const std::string& who) {
    // One pass: track the magnitude and the worst skew together. This runs on
    // every pack, i.e. once per state variable per Gauss point per iteration.
    value_type scale{0};
    value_type skew{0};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        scale = std::max(scale, std::abs(t(i, j)));
        if (j > i) skew = std::max(skew, std::abs(t(i, j) - t(j, i)));
      }
    const value_type tol = value_type{1e-12} * std::max(scale, value_type{1});

    if (skew > tol)
      throw fatal_error(
          "statev_map: history property '" + who +
          "' holds a non-symmetric rank-2 tensor, which cannot be stored in 6 "
          "STATEV slots without losing the skew part");
  }

  static void require_minor_symmetric(const tensor4& c, const std::string& who) {
    value_type scale{0};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
          for (int l = 0; l < 3; ++l)
            scale = std::max(scale, std::abs(c(i, j, k, l)));
    const value_type tol = value_type{1e-12} * std::max(scale, value_type{1});

    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
          for (int l = 0; l < 3; ++l)
            if (std::abs(c(i, j, k, l) - c(j, i, k, l)) > tol ||
                std::abs(c(i, j, k, l) - c(i, j, l, k)) > tol)
              throw fatal_error(
                  "statev_map: history property '" + who +
                  "' holds a rank-4 tensor without minor symmetry, which "
                  "cannot be stored in 36 STATEV slots without loss");
  }

  void add_entry(property_base* prop) {
    const auto& id = prop->traits().id;
    const std::string who = id.owner + "::" + id.name;
    const std::size_t offset = m_width;

    if (auto* h = dynamic_cast<hist_scalar*>(prop)) {
      m_entries.push_back(
          {id.owner, id.name, offset, 1,
           [h](const value_type* src) {
             h->old_value() = *src;
             h->new_value() = *src;
           },
           [h](value_type* dst) { *dst = h->new_value(); },
           nullptr});
      m_width += 1;
      return;
    }

    if (auto* h = dynamic_cast<hist_tensor2*>(prop)) {
      m_entries.push_back(
          {id.owner, id.name, offset, canonical_width,
           [h](const value_type* src) {
             h->old_value() = strain_from_buffer<value_type>(src);
             h->new_value() = h->old_value();
           },
           [h, who](value_type* dst) {
             require_symmetric(h->new_value(), who);
             strain_to_buffer<value_type>(h->new_value(), dst);
           },
           [h](const tensor2& R) {
             h->old_value() = rotate(h->old_value(), R);
             h->new_value() = rotate(h->new_value(), R);
           }});
      m_width += canonical_width;
      return;
    }

    if (auto* h = dynamic_cast<hist_tensor4*>(prop)) {
      constexpr std::size_t w = canonical_width * canonical_width;
      m_entries.push_back(
          {id.owner, id.name, offset, w,
           [h](const value_type* src) {
             h->old_value() = tangent_from_buffer<value_type>(src);
             h->new_value() = h->old_value();
           },
           [h, who](value_type* dst) {
             require_minor_symmetric(h->new_value(), who);
             tangent_to_buffer<value_type>(h->new_value(), dst);
           },
           [h](const tensor2& R) {
             h->old_value() = rotate(h->old_value(), R);
             h->new_value() = rotate(h->new_value(), R);
           }});
      m_width += w;
      return;
    }

    throw fatal_error(
        "statev_map: history property '" + who +
        "' has a type that cannot be stored in STATEV — supported types are "
        "the scalar value_type, tensor<T,3,2> and tensor<T,3,4>");
  }

  std::vector<entry> m_entries;
  std::size_t m_width{0};
};

}  // namespace numsim::materials::umat

#endif  // STATEV_MAP_H
