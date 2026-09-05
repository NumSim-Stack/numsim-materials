#ifndef NUMSIM_MATERIALS_NEWTON_SCALAR_H
#define NUMSIM_MATERIALS_NEWTON_SCALAR_H

#include <cmath>
#include <utility>

namespace numsim::materials {

/// Scalar Newton iteration. Plain algorithm: no properties, no graph presence,
/// no material_base.
///
/// Split out of backward_euler, which had grown two Newton loops -- one driven
/// by graph properties, one by a caller's lambda -- with separate damping and
/// separate clamping. The algorithm is the same in both cases; only where the
/// residual comes from differs, and that belongs to the caller.
///
/// Deliberately does NOT clamp its result. backward_euler forced x >= 0 for
/// plasticity's dlambda and |x| for a curing degree; both are statements about
/// a particular unknown, not about Newton's method, and a general solver that
/// silently enforces one is wrong for every other caller (see #13). Callers
/// clamp what they own.
template <typename T>
class newton_scalar {
public:
  struct result {
    T x{};                  ///< the iterate, converged or not
    bool converged{false};
    int iterations{0};
  };

  newton_scalar(T tolerance, int max_iterations) noexcept
      : m_tol(tolerance), m_max_iter(max_iterations) {}

  /// @param eval  x -> {residual, jacobian}
  /// @param x0    initial iterate
  ///
  /// Reports convergence rather than returning a bare number: a caller that
  /// does not check gets a non-converged iterate, which is the same failure
  /// backward_euler's graph path had -- there it was unreportable, here it is
  /// merely unchecked.
  template <typename Eval>
  result solve(Eval&& eval, T x0 = T{}) const {
    result r{x0, false, 0};
    for (int i = 0; i < m_max_iter; ++i) {
      const auto [residual, jacobian] = eval(r.x);
      ++r.iterations;
      if (std::abs(residual) < m_tol) {
        r.converged = true;
        return r;
      }
      // A vanishing jacobian is a stall, not a convergence.
      if (std::abs(jacobian) < T{1e-30}) return r;
      r.x -= residual / jacobian;
    }
    // Exhausted the budget: the last step may still have landed on the root,
    // so the residual is checked once more rather than assumed bad.
    const auto [residual, jacobian] = eval(r.x);
    (void)jacobian;
    r.converged = std::abs(residual) < m_tol;
    return r;
  }

  [[nodiscard]] T tolerance() const noexcept { return m_tol; }
  [[nodiscard]] int max_iterations() const noexcept { return m_max_iter; }

private:
  T m_tol;
  int m_max_iter;
};

}  // namespace numsim::materials

#endif  // NUMSIM_MATERIALS_NEWTON_SCALAR_H
