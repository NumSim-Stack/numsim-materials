#ifndef NUMSIM_MATERIALS_ISOTROPIC_TANGENT_H
#define NUMSIM_MATERIALS_ISOTROPIC_TANGENT_H

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/materials/plasticity_utils.h"

namespace numsim::materials {

/// Isotropic elastic stiffness as a material of its own.
///
/// `linear_elasticity` owns its tangent and computes it once in the
/// constructor, which is optimal while K and G are fixed but leaves no way to
/// recompute it — and no way to order that recomputation correctly if there
/// were one. A material reading its OWN property creates no edge in the
/// property graph, so `elastic::stress` sorts before `elastic::tangent` and
/// would consume a stale stiffness. Splitting the stiffness into a separate
/// material turns that invisible dependency into a real Global edge, which the
/// topological sort then honours.
///
/// It also makes the stiffness pluggable: anything producing a "tangent"
/// property — anisotropic, temperature-dependent, damage-degraded — is a drop-in
/// replacement, and consumers need not know which.
///
/// ### recompute
///
/// The update callback is bound ONLY when the "recompute" parameter is true.
/// With it false the tangent is built once, at construction, and the property
/// carries no callback at all — so `ctx.update()` skips it outright and the
/// per-call cost is exactly zero, matching `linear_elasticity`.
///
///   recompute = false   constants are fixed (Abaqus: PROPS cannot vary for a
///                       given material name)
///   recompute = true    constants change between calls (CalculiX interpolates
///                       *USER MATERIAL constants by temperature)
///
/// The flag is read ONCE, at construction, because that is when the callback is
/// bound — there is no way to attach one later. It is therefore stored by value
/// rather than as a reference into the parameter store: writing "recompute"
/// afterwards changes nothing, and an accessor reading the live parameter would
/// report that the tangent tracks K and G when in fact no callback exists.
/// `recomputes()` reports what was actually bound.
///
/// Setting it false while the constants do move yields a stale tangent, with
/// the stress still correct — the classic silently-wrong tangent, costing
/// convergence rate rather than accuracy. Nothing detects that automatically:
/// the material cannot see a write it has no callback to observe. Choosing the
/// flag correctly is the caller's responsibility, and
/// RecomputeFalseIgnoresALaterParameterWrite pins what happens if they do not.
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
        // By value, not by reference: see the class comment.
        m_recompute(base::template get_parameter<bool>("recompute")),
        // Bind the callback only when asked. add_output ignores a null one, so
        // with recompute=false the property has no callback and the engine
        // never visits it.
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
    // Defaults to the cheap behaviour, which matches linear_elasticity and is
    // right whenever the host cannot vary the constants.
    para.template insert<bool>("recompute").template add<set_default>(false);
    return para;
  }

  void update_tangent() {
    // IIdev comes from plasticity_utils rather than being spelled out again;
    // the isotropic basis was already written three times in this repo.
    const auto I{tmech::eye<value_type, Dim, 2>()};
    const auto IIvol{tmech::otimes(I, I) / Dim};
    m_C = 3 * m_K * IIvol +
          2 * m_G * plasticity_detail::make_IIdev<value_type, Dim>();
  }

  /// True when this material will follow a change to K or G.
  [[nodiscard]] bool recomputes() const noexcept { return m_recompute; }

private:
  const value_type& m_K;
  const value_type& m_G;
  const bool m_recompute;
  tensor4& m_C;
};

}  // namespace numsim::materials

#endif  // NUMSIM_MATERIALS_ISOTROPIC_TANGENT_H
