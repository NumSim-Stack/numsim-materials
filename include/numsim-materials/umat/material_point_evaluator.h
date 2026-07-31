#ifndef NUMSIM_MATERIALS_UMAT_MATERIAL_POINT_EVALUATOR_H
#define NUMSIM_MATERIALS_UMAT_MATERIAL_POINT_EVALUATOR_H

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/umat/errors.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/statev_map.h"
#include "numsim-materials/umat/tensor_conversion.h"

/// Drives a finalized material_context as a STATELESS material-point
/// evaluator: raw pointers in, raw pointers out, nothing retained between
/// calls.
///
/// Statelessness is the whole design, not an optimisation. A UMAT is re-called
/// on unconverged global Newton iterates, so anything the evaluator remembered
/// across calls would be state from a trial the host has since discarded. By
/// reloading every history variable from STATEV at the top of each call, a
/// repeated call with the same inputs gives the same answer — and one context
/// per thread is then sufficient, rather than one per integration point.
///
/// NOT thread-safe. evaluate() is deliberately non-const even though it reads
/// like a query, because it drives the shared context; a const method would
/// imply that a `const material_point_evaluator&` is safe to share between
/// threads, which it is not.
namespace numsim::materials::umat {

template <typename Traits>
class material_point_evaluator {
public:
  using value_type = typename Traits::value_type;
  using tensor2 = tmech::tensor<value_type, 3, 2>;
  using tensor4 = tmech::tensor<value_type, 3, 4>;
  using context_type = material_context<Traits>;

  struct config {
    /// Name of the external_strain_source material.
    std::string strain_source;
    /// Material producing the stress and tangent the host wants back.
    std::string stress_source;
    std::string stress_property{"stress"};
    std::string tangent_property{"tangent"};
    /// Optional external_scalar_source carrying time; empty to omit.
    std::string time_source{};
    /// Optional plastic-strain history, as "material::property".
    ///
    /// When set, SSE and SPD are reported to the host. When empty they are left
    /// untouched, so Abaqus's ALLSE/ALLPD will not account for this material —
    /// omission rather than a guess, because splitting work into stored and
    /// dissipated parts is impossible without knowing the elastic strain.
    std::string plastic_strain_property{};
    /// Additional host-owned history to keep out of STATEV, beyond the strain
    /// and time sources (which are excluded automatically).
    std::vector<statev_exclusion> extra_exclusions{};
  };

  /// One host call's arguments.
  ///
  /// Spans, not raw pointers: array lengths are dictated by `ec`, and passing a
  /// 4-element array with the default solid3d case reads six. That mistake is
  /// easy to make and invisible when it happens — carrying the length makes it
  /// checkable instead.
  struct call {
    std::span<const value_type> stran;   ///< strain at t_n, ntens(ec) slots
    std::span<const value_type> dstran;  ///< strain increment
    std::span<value_type> stress;        ///< in: sigma at t_n; out: at t_n+1
    std::span<value_type> ddsdde;        ///< out, ntens x ntens, column-major
    std::span<value_type> statev;        ///< in/out
    value_type time{0};   ///< TIME(2) at the start of the increment
    value_type dtime{0};  ///< DTIME
    /// Element family, per call rather than per evaluator: Abaqus may invoke
    /// one material for both solid and plane-strain elements in a single job.
    element_case ec{element_case::solid3d};
    /// Abaqus DROT(3,3), column-major. Empty means no rotation.
    std::span<const value_type> drot{};
    /// Optional energy outputs; null leaves the host's values alone.
    value_type* sse{nullptr};
    value_type* spd{nullptr};
    value_type* scd{nullptr};
  };

  material_point_evaluator(context_type& ctx, config cfg)
      : m_ctx(ctx), m_cfg(std::move(cfg)) {
    if (!m_ctx.is_finalized())
      throw fatal_error(
          "material_point_evaluator: the context must be finalized first");

    m_strain_src = resolve_source<external_strain_source<Traits>>(
        m_cfg.strain_source, "strain_source");

    std::vector<statev_exclusion> exclusions = m_cfg.extra_exclusions;
    exclusions.emplace_back(m_cfg.strain_source, "strain");

    if (!m_cfg.time_source.empty()) {
      m_time_src = resolve_source<external_scalar_source<Traits>>(
          m_cfg.time_source, "time_source");
      exclusions.emplace_back(m_cfg.time_source, "state");
    }

    m_stress =
        resolve_property<tensor2>(m_cfg.stress_source, m_cfg.stress_property);
    m_tangent =
        resolve_property<tensor4>(m_cfg.stress_source, m_cfg.tangent_property);

    if (!m_cfg.plastic_strain_property.empty()) {
      const auto src = connection_source::parse(m_cfg.plastic_strain_property);
      m_plastic_strain = resolve_property<tensor2>(src.material, src.property);
    }

    m_statev = std::make_unique<statev_map<Traits>>(m_ctx, exclusions);
  }

  /// Doubles this material needs in STATEV — what *DEPVAR must be at least.
  [[nodiscard]] std::size_t nstatv() const noexcept {
    return m_statev->nstatv();
  }

  [[nodiscard]] std::vector<std::string> describe_statev() const {
    return m_statev->describe();
  }

  /// Evaluate one material point. See the class comment for why this retains
  /// nothing, and why it is not const.
  void evaluate(const call& c) {
    if (c.ec == element_case::plane_stress)
      throw fatal_error(
          "material_point_evaluator: plane stress needs an outer solve for the "
          "out-of-plane strain — use plane_stress_evaluator instead");

    const auto n = validate(c);

    value_type old6[canonical_width];
    value_type new6[canonical_width];
    value_type total[canonical_width];
    for (std::size_t i = 0; i < n; ++i) total[i] = c.stran[i] + c.dstran[i];
    widen_vector<value_type>(c.stran.data(), c.ec, old6);
    widen_vector<value_type>(total, c.ec, new6);

    // sigma at t_n, for the work increment. Read BEFORE we overwrite STRESS.
    const bool want_energy = m_plastic_strain && c.sse && c.spd;
    value_type sig_old6[canonical_width]{};
    if (want_energy) widen_vector<value_type>(c.stress.data(), c.ec, sig_old6);

    value_type stress6[canonical_width];
    value_type tangent36[canonical_width * canonical_width];
    evaluate_canonical(c.statev.data(), old6, new6, c.time, c.dtime, stress6,
                       tangent36, c.drot);

    narrow_vector<value_type>(stress6, c.ec, c.stress.data());
    narrow_matrix<value_type>(tangent36, c.ec, c.ddsdde.data());

    if (want_energy)
      update_energies(c.sse, c.spd, old6, new6, sig_old6, stress6);

    store_statev(c.statev.data());
  }

  /// One graph evaluation from canonical 6-slot strain buffers, leaving the
  /// results in canonical buffers.
  ///
  /// Reloads history from @p statev but deliberately does NOT write it back.
  /// That split is what the plane-stress solve needs: every trial value of the
  /// out-of-plane strain must integrate from the SAME converged state at t_n,
  /// and only the accepted iterate may be stored. Callers that evaluate once
  /// should use evaluate() instead, which pairs this with store_statev().
  void evaluate_canonical(const value_type* statev, const value_type* old6,
                          const value_type* new6, value_type time,
                          value_type dtime, value_type* stress6,
                          value_type* tangent36,
                          std::span<const value_type> drot = {}) {
    // STATEV is the only state store: reload it every call, so a repeated call
    // on an unconverged iterate starts from t_n, not from the previous trial.
    m_statev->unpack(statev);

    // Co-rotational update. STATEV was written in the previous configuration,
    // so it must be rotated after unpack and before the material runs.
    if (!drot.empty() && m_statev->has_rotatable_history()) {
      const auto R = rotation_from_buffer<value_type>(drot.data());
      if (!is_identity_rotation(R)) m_statev->rotate_history(R);
    }

    m_strain_src->bind(old6, new6);
    if (m_time_src) m_time_src->bind(time, time + dtime);

    m_ctx.update();

    stress_to_buffer<value_type>(*m_stress, stress6);
    tangent_to_buffer<value_type>(*m_tangent, tangent36);
  }

  /// Write the updated history back. No commit(): the host owns the timestep.
  void store_statev(value_type* statev) const { m_statev->pack(statev); }

  void check_nstatv(std::size_t host_nstatv) const {
    m_statev->check_nstatv(host_nstatv);
  }

  [[nodiscard]] bool reports_energies() const noexcept {
    return m_plastic_strain != nullptr;
  }

  /// Update SSE and SPD from the converged state.
  ///
  ///   SSE  = 1/2 sigma : (eps - eps_p)             stored elastic energy
  ///   dW   = 1/2 (sigma_n + sigma_n+1) : d_eps     total work increment
  ///   SPD += dW - dSSE                             the part not stored
  ///
  /// The trapezoidal work increment is second-order accurate in the step — the
  /// same order as the stress update it accompanies.
  /// Everything is taken from the CANONICAL buffers, never re-derived from the
  /// call's host-sized arrays. The strain increment is eps_new6 - eps_old6,
  /// which is correct for every element family without knowing which one it is
  /// — and in the plane-stress case it correctly includes the solved-for
  /// out-of-plane increment (which does no work, since sigma_33 vanishes there,
  /// but costs nothing to carry).
  void update_energies(value_type* sse, value_type* spd,
                       const value_type* eps_old6, const value_type* eps_new6,
                       const value_type* sig_old6,
                       const value_type* sig_new6) const {
    value_type deps6[canonical_width];
    for (std::size_t i = 0; i < canonical_width; ++i)
      deps6[i] = eps_new6[i] - eps_old6[i];

    const auto eps = strain_from_buffer<value_type>(eps_new6);
    const auto deps = strain_from_buffer<value_type>(deps6);
    const auto sig_new = stress_from_buffer<value_type>(sig_new6);
    const auto sig_old = stress_from_buffer<value_type>(sig_old6);

    tensor2 eps_e;
    eps_e = eps - *m_plastic_strain;
    tensor2 sig_mid;
    sig_mid = value_type{0.5} * (sig_old + sig_new);

    const value_type sse_new =
        value_type{0.5} * tmech::dcontract(sig_new, eps_e);
    const value_type dW = tmech::dcontract(sig_mid, deps);

    *spd += dW - (sse_new - *sse);
    *sse = sse_new;
  }

private:
  /// Check the caller's buffers against the element family. Every mismatch is a
  /// setup fault, not something a smaller increment could fix.
  std::size_t validate(const call& c) const {
    const auto n = ntens(c.ec);
    auto require = [](bool ok, const char* what) {
      if (!ok)
        throw fatal_error(std::string("material_point_evaluator: ") + what);
    };
    require(c.stran.size() == n, "stran has the wrong length for this element type");
    require(c.dstran.size() == n, "dstran has the wrong length for this element type");
    require(c.stress.size() == n, "stress has the wrong length for this element type");
    require(c.ddsdde.size() == n * n, "ddsdde has the wrong length for this element type");
    require(c.drot.empty() || c.drot.size() == 9, "drot must be empty or 3x3");
    m_statev->check_nstatv(c.statev.size());
    return n;
  }

  template <typename Source>
  Source* resolve_source(const std::string& name, const char* role) {
    auto* mat = m_ctx.find(name);
    if (!mat)
      throw fatal_error(std::string("material_point_evaluator: ") + role + " '" +
                        name + "' is not a material in this context");
    auto* typed = dynamic_cast<Source*>(mat);
    if (!typed)
      throw fatal_error(
          std::string("material_point_evaluator: ") + role + " '" + name +
          "' exists but is not a host-driven source material — it cannot be "
          "bound to the host's pointers");
    return typed;
  }

  template <typename T>
  const T* resolve_property(const std::string& material,
                            const std::string& property) {
    auto* prop = m_ctx.find_property(material, property);
    if (!prop)
      throw fatal_error("material_point_evaluator: property '" + material +
                        "::" + property + "' not found");
    if (auto* typed =
            dynamic_cast<const numsim_core::property<T, property_traits>*>(prop))
      return &typed->get();
    if (auto* hist = dynamic_cast<
            const numsim_core::history_property<T, property_traits>*>(prop))
      return &hist->new_value();
    throw fatal_error("material_point_evaluator: property '" + material +
                      "::" + property + "' has an unexpected type");
  }

  context_type& m_ctx;
  config m_cfg;
  external_strain_source<Traits>* m_strain_src{nullptr};
  external_scalar_source<Traits>* m_time_src{nullptr};
  const tensor2* m_stress{nullptr};
  const tensor4* m_tangent{nullptr};
  const tensor2* m_plastic_strain{nullptr};
  std::unique_ptr<statev_map<Traits>> m_statev;
};

}  // namespace numsim::materials::umat

#endif  // NUMSIM_MATERIALS_UMAT_MATERIAL_POINT_EVALUATOR_H
