#ifndef NUMSIM_MATERIALS_ISOTROPIC_TANGENT_H
#define NUMSIM_MATERIALS_ISOTROPIC_TANGENT_H

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/materials/plasticity_utils.h"

namespace numsim::materials {

/// Isotropic elastic stiffness as a material of its own.
///
/// linear_elasticity owns its tangent, and a material reading its own property
/// creates no graph edge — so elastic::stress sorts before elastic::tangent and
/// would consume a stale stiffness. Consuming ANOTHER material's property is a
/// real edge, which the topological sort honours. Also makes the stiffness
/// pluggable: anything producing "tangent" is a drop-in.
///
/// "recompute" binds the update callback. False (the default, matching
/// linear_elasticity) leaves the property with no callback, so the engine skips
/// it and per-call cost is zero — right when constants are fixed (Abaqus PROPS).
/// True is for constants that vary per call (CalculiX interpolates them by
/// temperature). Read once, at construction: that is when the callback is bound.
///
/// recompute=false while constants move leaves a stale tangent with the stress
/// still correct — costs convergence rate, not accuracy, and nothing detects it.
/// See RecomputeFalseIgnoresALaterParameterWrite.
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
        m_K(base::template get_parameter<value_type>("K")),
        m_G(base::template get_parameter<value_type>("G")),
        // By value: the callback is bound once, so a live reference would let
        // recomputes() claim tracking that does not exist.
        m_recompute(base::template get_parameter<bool>("recompute")),
        // add_output ignores a null callback.
        m_C(base::template add_output<tensor4>(
            "tangent",
            base::template get_parameter<bool>("recompute")
                ? &isotropic_tangent::update_tangent
                : nullptr)) {
    // Always compute once, so the tangent is valid before the first update()
    // whether or not it will ever be recomputed.
    update_tangent();
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<value_type>("K").template add<is_required>();
    para.template insert<value_type>("G").template add<is_required>();
    para.template insert<bool>("recompute").template add<set_default>(false);
    return para;
  }

  void update_tangent() {
    const auto I{tmech::eye<value_type, Dim, 2>()};
    const auto IIvol{tmech::otimes(I, I) / Dim};
    m_C = 3 * m_K * IIvol +
          2 * m_G * plasticity_detail::make_IIdev<value_type, Dim>();
  }

  /// Whether a callback was bound, i.e. whether K/G changes are followed.
  [[nodiscard]] bool recomputes() const noexcept { return m_recompute; }

private:
  const value_type& m_K;
  const value_type& m_G;
  const bool m_recompute;
  tensor4& m_C;
};

}  // namespace numsim::materials

#endif  // NUMSIM_MATERIALS_ISOTROPIC_TANGENT_H
