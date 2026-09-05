#ifndef NUMSIM_MATERIALS_LOCAL_NEWTON_H
#define NUMSIM_MATERIALS_LOCAL_NEWTON_H

#include <string>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/solvers/newton_scalar.h"

namespace numsim::materials {

/// A scalar Newton another MATERIAL drives, by calling solve() with its own
/// residual.
///
/// The counterpart to backward_euler, which the property graph drives. The two
/// used to be one class selected by whether a "function" parameter happened to
/// be set, which meant an empty "function" silently chose this behaviour --
/// leaving a graph-driven consumer reading a delta of 0 forever, with no error
/// anywhere. Splitting them makes the choice a type rather than a defaulted
/// string.
///
/// Publishes no properties and takes no inputs. It is a material only so its
/// tolerance and iteration budget are configurable from the same document as
/// everything else, and so one instance can serve several materials.
///
/// Used by the return maps, which cannot be graph-driven: they solve twice per
/// update with different residuals (smooth cone, then apex) and choose the
/// branch on the first solve's convergence. A graph edge carries one number,
/// not that.
///
/// Parameters:
///   "name", "tolerance", "max_iter"
template <typename Traits>
class local_newton final : public material_base<local_newton<Traits>, Traits> {
public:
  using base = material_base<local_newton<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using result = typename newton_scalar<value_type>::result;

  template <typename... Args>
  explicit local_newton(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_solver(base::template get_parameter<value_type>("tolerance"),
                 base::template get_parameter<int>("max_iter")) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<value_type>("tolerance")
        .template add<set_default>(value_type{1e-10});
    para.template insert<int>("max_iter").template add<set_default>(50);
    return para;
  }

  /// Solve with the caller's residual. @p eval maps x -> {residual, jacobian}.
  ///
  /// The result carries its own convergence flag, so it cannot be read without
  /// being available -- unlike a converged() queried separately, which a caller
  /// can forget and which goes stale between solves.
  template <typename Eval>
  result solve(Eval&& eval, value_type x0 = value_type{}) const {
    return m_solver.solve(std::forward<Eval>(eval), x0);
  }

private:
  newton_scalar<value_type> m_solver;
};

}  // namespace numsim::materials

#endif  // NUMSIM_MATERIALS_LOCAL_NEWTON_H
