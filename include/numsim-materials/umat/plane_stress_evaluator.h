#ifndef NUMSIM_MATERIALS_UMAT_PLANE_STRESS_EVALUATOR_H
#define NUMSIM_MATERIALS_UMAT_PLANE_STRESS_EVALUATOR_H

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "numsim-materials/umat/material_point_evaluator.h"

/// Plane stress on top of a 3D material.
///
/// Every other element family hands over a complete strain state. Plane stress
/// does not: the host supplies only {11, 22, 12} and the out-of-plane strain is
/// whatever makes sigma_33 vanish. Since eps_33 is generally NOT recoverable in
/// closed form once the material is nonlinear — plastic incompressibility makes
/// it depend on the whole loading path — it is solved for.
///
/// This lives outside the material graph, as a wrapper around
/// material_point_evaluator, because each trial value of eps_33 requires a full
/// re-evaluation of the graph.
namespace numsim::materials::umat {

template <typename Traits>
class plane_stress_evaluator {
public:
  using value_type = typename Traits::value_type;
  using inner_type = material_point_evaluator<Traits>;
  using context_type = typename inner_type::context_type;
  using config = typename inner_type::config;
  using call = typename inner_type::call;

  struct options {
    /// Relative tolerance on |sigma_33|, measured against the largest stress
    /// component so the test is scale-free.
    value_type tolerance{1e-10};
    int max_iter{25};
  };

  plane_stress_evaluator(context_type& ctx, config cfg, options opts = {})
      : m_opts(opts) {
    m_inner = std::make_unique<inner_type>(ctx, std::move(cfg));
  }

  /// STATEV width: the material's own history, plus one slot for the converged
  /// out-of-plane strain.
  ///
  /// That extra slot is not bookkeeping for its own sake. eps_33 at t_n is
  /// genuinely part of the state: it is the old side of the strain history that
  /// a rate-dependent material would read, and it is by far the best starting
  /// guess for this step's solve — under plasticity a cold start costs several
  /// extra full graph evaluations per Gauss point.
  [[nodiscard]] std::size_t nstatv() const noexcept {
    return m_inner->nstatv() + 1;
  }

  [[nodiscard]] std::vector<std::string> describe_statev() const {
    auto d = m_inner->describe_statev();
    const auto slot = std::to_string(m_inner->nstatv());
    d.push_back(slot + ".." + slot + "  <plane_stress>::out_of_plane_strain");
    return d;
  }

  /// Iterations the last evaluate() needed, for diagnostics.
  [[nodiscard]] int last_iterations() const noexcept { return m_last_iters; }

  void evaluate(const call& c) {
    if (c.statev.size() < nstatv())
      throw fatal_error(
          "plane_stress_evaluator: the host provides NSTATV=" +
          std::to_string(c.statev.size()) + " but plane stress needs " +
          std::to_string(nstatv()) +
          " (the material's history plus one slot for the out-of-plane strain)");
    const auto n = ntens(element_case::plane_stress);
    if (c.stran.size() != n || c.dstran.size() != n || c.stress.size() != n ||
        c.ddsdde.size() != n * n)
      throw fatal_error(
          "plane_stress_evaluator: stran/dstran/stress/ddsdde must be sized "
          "for plane stress (3 components, 3x3 tangent)");
    if (!c.drot.empty() && c.drot.size() != 9)
      throw fatal_error("plane_stress_evaluator: drot must be empty or 3x3");

    // The out-of-plane strain is carried as a bare scalar in STATEV and is the
    // one piece of state rotate_history cannot touch. That is safe exactly when
    // the rotation leaves the 3-axis alone, which is what the plane-stress
    // element families produce — but call::drot accepts an arbitrary 3x3, and a
    // general rotation would mix eps_33 into eps_22/eps_23 with nothing to
    // signal it. So it is checked rather than assumed.
    if (!c.drot.empty()) {
      const auto R = rotation_from_buffer<value_type>(c.drot.data());
      constexpr value_type axis_tol{1e-8};
      if (std::abs(R(2, 0)) > axis_tol || std::abs(R(2, 1)) > axis_tol ||
          std::abs(R(0, 2)) > axis_tol || std::abs(R(1, 2)) > axis_tol)
        throw fatal_error(
            "plane_stress_evaluator: DROT rotates out of the plane, which the "
            "scalar out-of-plane strain in STATEV cannot represent");
    }

    const std::size_t eps33_slot = m_inner->nstatv();
    const value_type eps33_old = c.statev[eps33_slot];

    value_type in_plane_new[3];
    for (std::size_t i = 0; i < 3; ++i)
      in_plane_new[i] = c.stran[i] + c.dstran[i];

    value_type old6[canonical_width];
    value_type new6[canonical_width];
    widen_vector<value_type>(c.stran.data(), element_case::plane_stress, old6,
                             eps33_old);

    // sigma at t_n, for the work increment. Read before STRESS is overwritten.
    const bool want_energy =
        m_inner->reports_energies() && c.sse && c.spd;
    value_type sig_old6[canonical_width]{};
    if (want_energy)
      widen_vector<value_type>(c.stress.data(), element_case::plane_stress,
                               sig_old6);

    value_type stress6[canonical_width];
    value_type tangent36[canonical_width * canonical_width];

    // Start from the converged out-of-plane strain: for a linear material this
    // is already within one Newton step, and for a plastic one it is far closer
    // than zero.
    value_type eps33 = eps33_old;
    bool converged = false;
    m_last_iters = 0;

    for (int iter = 0; iter < m_opts.max_iter; ++iter) {
      widen_vector<value_type>(in_plane_new, element_case::plane_stress, new6,
                               eps33);

      // Reload history from STATEV on every trial, so each candidate eps_33
      // integrates from the same converged state at t_n rather than from the
      // previous trial's result. Getting this wrong yields a plausible-looking
      // but path-corrupted answer.
      m_inner->evaluate_canonical(c.statev.data(), old6, new6, c.time, c.dtime,
                                  stress6, tangent36, c.drot);
      ++m_last_iters;

      const value_type r = stress6[canonical_33];

      value_type scale{0};
      for (std::size_t i = 0; i < canonical_width; ++i)
        scale = std::max(scale, std::abs(stress6[i]));
      if (std::abs(r) <= m_opts.tolerance * std::max(scale, value_type{1})) {
        converged = true;
        break;  // break BEFORE updating, so the graph state matches this eps_33
      }

      const value_type dr =
          tangent36[canonical_33 * canonical_width + canonical_33];
      if (dr == value_type{0})
        throw cutback_error(
            "plane_stress_evaluator: dsigma_33/deps_33 is zero — the "
            "out-of-plane direction carries no stiffness, so sigma_33 = 0 "
            "cannot be enforced");
      eps33 -= r / dr;
    }

    if (!converged)
      throw cutback_error(
          "plane_stress_evaluator: the out-of-plane strain solve did not "
          "converge in " + std::to_string(m_opts.max_iter) + " iterations");

    narrow_vector<value_type>(stress6, element_case::plane_stress,
                              c.stress.data());
    narrow_matrix<value_type>(tangent36, element_case::plane_stress,
                              c.ddsdde.data());

    // Energies use the CONVERGED canonical state, which already carries the
    // solved-for eps_33 — so the elastic split is the genuine 3D one.
    if (want_energy)
      m_inner->update_energies(c.sse, c.spd, old6, new6, sig_old6, stress6);

    m_inner->store_statev(c.statev.data());
    c.statev[eps33_slot] = eps33;
  }

private:
  std::unique_ptr<inner_type> m_inner;
  options m_opts;
  int m_last_iters{0};
};

}  // namespace numsim::materials::umat

#endif  // NUMSIM_MATERIALS_UMAT_PLANE_STRESS_EVALUATOR_H
