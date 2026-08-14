#ifndef NUMSIM_MATERIALS_ISOTROPIC_TANGENT_H
#define NUMSIM_MATERIALS_ISOTROPIC_TANGENT_H

#include <cstddef>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/materials/plasticity_utils.h"

namespace numsim::materials {

/// Isotropic elastic stiffness as a material, with K and G as graph inputs.
///
/// linear_elasticity owns its tangent, and a material reading its own property
/// creates no graph edge — so elastic::stress sorts before elastic::tangent and
/// would consume a stale stiffness. Consuming ANOTHER material's property is a
/// real edge, which the topological sort honours.
///
/// The moduli are inputs rather than parameters for the same reason: a
/// parameter has no edge to the material that reads it, so nothing orders a
/// change to it against the values derived from it. Wire from constant_scalar
/// when they are fixed, or from any scalar producer when they vary (temperature
/// dependence) — set K_property/G_property if it does not publish under "value".
/// Which one you wire IS the choice, and no flag can disagree with it.
///
/// The callback is always bound: inputs are not wired until finalize(), so
/// nothing can be computed in the constructor. It self-guards on the moduli, so
/// the fixed case costs two comparisons rather than a rank-4 rebuild.
template <typename Traits>
class isotropic_tangent final
    : public material_base<isotropic_tangent<Traits>, Traits> {
public:
  using base = material_base<isotropic_tangent<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;

  template <typename... Args>
  explicit isotropic_tangent(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_C(base::template add_output<tensor4>(
            "tangent", &isotropic_tangent::update_tangent)),
        m_K(base::template add_input<value_type>(
            base::template get_parameter<std::string>("K_source"),
            base::template get_parameter<std::string>("K_property"),
            EdgeKind::Global)),
        m_G(base::template add_input<value_type>(
            base::template get_parameter<std::string>("G_source"),
            base::template get_parameter<std::string>("G_property"),
            EdgeKind::Global)) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("K_source").template add<is_required>();
    para.template insert<std::string>("G_source").template add<is_required>();
    // Defaults to the scalar-output convention constant_scalar and
    // scalar_identity_weight follow. Override for a source that publishes under
    // another name — external_scalar_source publishes "state", for instance.
    para.template insert<std::string>("K_property")
        .template add<set_default>(std::string{"value"});
    para.template insert<std::string>("G_property")
        .template add<set_default>(std::string{"value"});
    return para;
  }

  void update_tangent() {
    if (m_valid && m_K.get() == m_K_cached && m_G.get() == m_G_cached) return;
    m_K_cached = m_K.get();
    m_G_cached = m_G.get();
    m_valid = true;

    const auto I{tmech::eye<value_type, Dim, 2>()};
    const auto IIvol{tmech::otimes(I, I) / Dim};
    m_C = 3 * m_K_cached * IIvol +
          2 * m_G_cached * plasticity_detail::make_IIdev<value_type, Dim>();
    ++m_recomputations;
  }

  /// Force a rebuild on the next update.
  ///
  /// The guard keys on K and G, which assumes this material is the only writer
  /// of its "tangent" property. material_context::get_mutable() hands out a
  /// mutable reference to exactly that, so anything writing it without writing
  /// it back would otherwise wedge the memo permanently.
  void invalidate() noexcept { m_valid = false; }

  /// How often the stiffness was actually rebuilt. Diagnostics for the guard:
  /// with fixed moduli this stays at 1 however many updates run.
  [[nodiscard]] std::size_t recomputations() const noexcept {
    return m_recomputations;
  }

private:
  tensor4& m_C;
  const input_property<value_type, property_traits>& m_K;
  const input_property<value_type, property_traits>& m_G;
  value_type m_K_cached{};
  value_type m_G_cached{};
  bool m_valid{false};
  std::size_t m_recomputations{0};
};

}  // namespace numsim::materials

#endif  // NUMSIM_MATERIALS_ISOTROPIC_TANGENT_H
