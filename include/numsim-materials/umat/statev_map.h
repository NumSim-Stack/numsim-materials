#ifndef STATEV_MAP_H
#define STATEV_MAP_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/umat/errors.h"
#include "numsim-materials/core/history_kind.h"
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
/// A vector or rank-2 history is stored and rotated as the history_kind it was
/// declared with (core/history_kind.h): stress-like and moduli without the
/// shear factor, vectors and non-symmetric tensors with all components,
/// material legs never rotated. Declared without a kind, a tensor<T,3,2> is a
/// spatial strain and a tensor<T,3,1> a spatial vector.
///
/// One `(material, property)` pair the host owns and STATEV must NOT contain.
using statev_exclusion = std::pair<std::string, std::string>;

template <typename Traits>
class statev_map {
public:
  using value_type = typename Traits::value_type;
  using tensor1 = tmech::tensor<value_type, 3, 1>;
  using tensor2 = tmech::tensor<value_type, 3, 2>;
  using tensor4 = tmech::tensor<value_type, 3, 4>;
  using hist_scalar = numsim_core::history_property<value_type, property_traits>;
  using hist_tensor1 = numsim_core::history_property<tensor1, property_traits>;
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
    for (const auto& e : m_entries) e->unpack(statev + e->offset);
  }

  /// Store the updated history (the `new` side) back into STATEV.
  void pack(value_type* statev) const {
    for (const auto& e : m_entries) e->pack(statev + e->offset);
  }

  /// Rotate every tensor-valued state variable into the current configuration.
  ///
  /// Under NLGEOM=YES Abaqus rotates STRESS and STRAN to the end-of-increment
  /// frame before calling the UMAT, but it does NOT touch STATEV — rotating
  /// user state variables is the UMAT's job. Skipping it leaves tensor state
  /// (plastic strain, back stress) in a stale frame, and the error accumulates
  /// silently over the rotation history with no diagnostic anywhere.
  ///
  /// Scalars are frame-indifferent and untouched. R acts on every spatial leg
  /// of a tensor entry and on no material leg, so reference-configuration
  /// history (Cp, Fp, a material vector) keeps its components.
  void rotate_history(const tensor2& R) const {
    for (const auto& e : m_entries)
      if (e->rotatable()) e->rotate(R);
  }

  /// Whether this map holds any state that a rotation would affect.
  [[nodiscard]] bool has_rotatable_history() const noexcept {
    return std::ranges::any_of(m_entries, [](const auto& e) { return e->rotatable(); });
  }

  /// Human-readable layout, for diagnostics and for documenting *DEPVAR.
  [[nodiscard]] std::vector<std::string> describe() const {
    std::vector<std::string> out;
    out.reserve(m_entries.size());
    for (const auto& e : m_entries) {
      const std::string tags = e->tags();
      out.push_back(std::to_string(e->offset) + ".." +
                    std::to_string(e->offset + e->width() - 1) + "  " + e->owner +
                    "::" + e->name + (tags.empty() ? "" : "  " + tags));
    }
    return out;
  }

private:
  /// One history property's slots in STATEV.
  struct statev_slot {
    statev_slot(const property_base& prop, std::size_t at)
        : owner(prop.traits().id.owner), name(prop.traits().id.name), offset(at) {}
    virtual ~statev_slot() = default;
    [[nodiscard]] virtual std::size_t width() const = 0;
    /// Sets both sides from STATEV.
    virtual void unpack(const value_type* src) const = 0;
    /// The new side into STATEV.
    virtual void pack(value_type* dst) const = 0;
    virtual void rotate(const tensor2& /*R*/) const {}
    /// Whether rotate() acts on anything; false for scalars and material-leg tensors.
    [[nodiscard]] virtual bool rotatable() const { return false; }
    /// Storage and leg marks for describe().
    [[nodiscard]] virtual std::string tags() const { return {}; }

    std::string owner;
    std::string name;
    std::size_t offset;
  };

  /// A rank-2 history variable is stored in 6 slots, which can only represent a
  /// symmetric tensor. That is right for every state variable in practice
  /// (plastic strain, back stress), but silently wrong for one that is not —
  /// a deformation gradient, say. So the skew part is checked rather than
  /// assumed, and a violation is reported instead of quietly discarded.
  static void require_symmetric(const tensor2& t, const statev_slot& slot) {
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
          "statev_map: history property '" + slot.owner + "::" + slot.name +
          "' holds a non-symmetric rank-2 tensor, which cannot be stored in 6 "
          "STATEV slots without losing the skew part");
  }

  static void require_minor_symmetric(const tensor4& c, const statev_slot& slot) {
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
                  "statev_map: history property '" + slot.owner + "::" + slot.name +
                  "' holds a rank-4 tensor without minor symmetry, which "
                  "cannot be stored in 36 STATEV slots without loss");
  }

  /// A scalar: one slot, frame-indifferent.
  struct scalar_slot final : statev_slot {
    scalar_slot(hist_scalar& h, std::size_t at) : statev_slot(h, at), history(h) {}
    std::size_t width() const override { return 1; }
    void unpack(const value_type* src) const override {
      history.old_value() = *src;
      history.new_value() = *src;
    }
    void pack(value_type* dst) const override { *dst = history.new_value(); }
    hist_scalar& history;
  };

  /// A vector or rank-2 history: width and shear factor from the kind's storage, rotation from its legs.
  template <std::size_t Rank>
  struct tensor_slot final : statev_slot {
    using hist = numsim_core::history_property<tmech::tensor<value_type, 3, Rank>, property_traits>;
    tensor_slot(hist& h, std::size_t at, history_kind k) : statev_slot(h, at), history(h), kind(k) {}
    std::size_t width() const override { return kind.store == storage::full ? (Rank == 1 ? 3 : 9) : canonical_width; }
    void unpack(const value_type* src) const override {
      history.old_value() = from_buffer(src);
      history.new_value() = history.old_value();
    }
    void pack(value_type* dst) const override {
      const auto& t = history.new_value();
      if constexpr (Rank == 2) {
        switch (kind.store) {
          case storage::engineering:
            require_symmetric(t, *this);
            return strain_to_buffer<value_type>(t, dst);
          case storage::plain:
            require_symmetric(t, *this);
            return stress_to_buffer<value_type>(t, dst);
          case storage::full:
            break;
        }
      }
      full_to_buffer<value_type, Rank>(t, dst);
    }
    void rotate(const tensor2& R) const override {
      history.old_value() = kind.rotated(history.old_value(), R);
      history.new_value() = kind.rotated(history.new_value(), R);
    }
    bool rotatable() const override { return kind.rotates(Rank); }
    std::string tags() const override { return kind.tags(Rank); }

  private:
    tmech::tensor<value_type, 3, Rank> from_buffer(const value_type* src) const {
      if constexpr (Rank == 2) {
        switch (kind.store) {
          case storage::engineering:
            return strain_from_buffer<value_type>(src);
          case storage::plain:
            return stress_from_buffer<value_type>(src);
          case storage::full:
            break;
        }
      }
      return full_from_buffer<value_type, Rank>(src);
    }
    hist& history;
    history_kind kind;
  };

  /// A minor-symmetric rank-4 history in 36 slots, all four legs spatial.
  struct tensor4_slot final : statev_slot {
    tensor4_slot(hist_tensor4& h, std::size_t at) : statev_slot(h, at), history(h) {}
    std::size_t width() const override { return canonical_width * canonical_width; }
    void unpack(const value_type* src) const override {
      history.old_value() = tangent_from_buffer<value_type>(src);
      history.new_value() = history.old_value();
    }
    void pack(value_type* dst) const override {
      require_minor_symmetric(history.new_value(), *this);
      tangent_to_buffer<value_type>(history.new_value(), dst);
    }
    void rotate(const tensor2& R) const override {
      history.old_value() = umat::rotate(history.old_value(), R);
      history.new_value() = umat::rotate(history.new_value(), R);
    }
    bool rotatable() const override { return true; }
    hist_tensor4& history;
  };

  void add_entry(const material_context<Traits>& ctx, property_base* prop) {
    std::unique_ptr<statev_slot> slot;
    if (auto* h = dynamic_cast<hist_scalar*>(prop))
      slot = std::make_unique<scalar_slot>(*h, m_width);
    else if (auto* h = dynamic_cast<hist_tensor1*>(prop))
      slot = std::make_unique<tensor_slot<1>>(*h, m_width, kind_of<tensor1>(ctx, *prop));
    else if (auto* h = dynamic_cast<hist_tensor2*>(prop))
      slot = std::make_unique<tensor_slot<2>>(*h, m_width, kind_of<tensor2>(ctx, *prop));
    else if (auto* h = dynamic_cast<hist_tensor4*>(prop))
      slot = std::make_unique<tensor4_slot>(*h, m_width);
    else
      throw fatal_error(
          "statev_map: history property '" + prop->traits().id.owner + "::" + prop->traits().id.name +
          "' has a type that cannot be stored in STATEV — supported types are "
          "the scalar value_type, tensor<T,3,1>, tensor<T,3,2> and tensor<T,3,4>");
    m_width += slot->width();
    m_entries.push_back(std::move(slot));
  }

  /// The kind the owning material declared the history with, else default_history_kind.
  template <typename V>
  static history_kind kind_of(const material_context<Traits>& ctx, const property_base& prop) {
    const auto* owner = ctx.find(prop.traits().id.owner);
    return (owner ? owner->history_kind_of(&prop) : std::nullopt).value_or(*default_history_kind<V>);
  }

  std::vector<std::unique_ptr<statev_slot>> m_entries;
  std::size_t m_width{0};
};

}  // namespace numsim::materials::umat

#endif  // STATEV_MAP_H
