#ifndef NUMSIM_MATERIALS_VECTOR_NEWTON_H
#define NUMSIM_MATERIALS_VECTOR_NEWTON_H

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/solvers/unknown_layout.h"

namespace numsim::materials {

/// Coupled Newton solver for R(x) = 0 with a mixed scalar/tensor unknown set.
///
/// The vector analogue of backward_euler::solve. Topology mirrors it: the
/// solver PRODUCES the unknowns as output properties and CONSUMES the residual
/// entries and Jacobian blocks as Local inputs, so the function material reads
/// the trial state back through the graph.
///
/// Properties, by convention from the unknown names:
///   produces  <solver>::<name>                       (one per unknown)
///   consumes  <function>::residual_<name>
///             <function>::jacobian_<name_i>_<name_j>
///
/// Every piece carries its own layout (see unknown_layout.h); the solver only
/// sees layout_base and never touches Mandel packing itself.
///
/// Deliberately has NO hidden projection. backward_euler::solve once clamped
/// every result to non-negative, which silently corrupted any signed unknown;
/// a caller needing a KKT projection must apply it explicitly.
template<typename Traits>
class vector_newton final
    : public material_base<vector_newton<Traits>, Traits> {
public:
  using base = material_base<vector_newton<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  static constexpr auto Dim = base::Dim;

  using scalar_k = scalar_unknown<value_type>;
  using sym_k    = sym_tensor_unknown<value_type, Dim>;

  using vector_t = Eigen::Matrix<value_type, Eigen::Dynamic, 1>;
  using matrix_t = Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic>;

  template<typename... Args>
  explicit vector_newton(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_func(base::template get_parameter<std::string>("function")),
        m_tol(base::template get_parameter<value_type>("tolerance")),
        m_lin_tol(base::template get_parameter<value_type>("linear_tolerance")),
        m_max_iter(base::template get_parameter<int>("max_iter")),
        m_strain_derivative(
            base::template get_parameter<bool>("strain_derivative")),
        m_verify_zeros(base::template get_parameter<bool>("verify_zero_blocks"))
  {
    const auto& specs =
        base::template get_parameter<std::vector<unknown_spec>>("unknowns");

    std::vector<block_ref> zeros;
    if (base::m_parameter_handler.contains("zero_blocks"))
      zeros = base::template get_parameter<std::vector<block_ref>>("zero_blocks");

    // Validate the name strings BEFORE creating any property. See the comment
    // on check_names() — past this point a bad name is undiagnosable.
    check_names(specs, zeros, base::name(), m_strain_derivative);

    const auto n = specs.size();
    m_jacobian.resize(n);
    for (auto& row : m_jacobian) row.resize(n);

    // State outputs + residual inputs.
    for (const auto& s : specs) add_state_and_residual(s);

    // Jacobian blocks: dense by default, so a forgotten block is a wiring
    // error rather than a silently-wrong Jacobian. Sparsity is opt-in.
    //
    // 'zero_blocks' asserts that a block is IDENTICALLY zero — as when two
    // yield surfaces genuinely do not interact. It must not be used to
    // approximate a nonzero block away. The Newton iteration would tolerate
    // that (an inexact J costs iterations, not accuracy, because the root is
    // pinned by R == 0), but the linearization would not: the consistent
    // tangent comes from the implicit function theorem,
    //
    //     J * dx/deps = -dR/deps,   J = dR/dx at the converged point
    //
    // which inverts J as the actual derivative. A zeroed-out nonzero block
    // therefore yields a silently wrong dx/deps — and so a wrong dsigma/deps,
    // costing quadratic convergence in the global Newton. See
    // solve_with_factorization(), which hands back exactly this J.
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        const auto is_zero =
            std::find(zeros.begin(), zeros.end(),
                      block_ref{specs[i].name, specs[j].name}) != zeros.end();
        if (is_zero) m_zero_ij.emplace_back(i, j);
        // Verify mode wires the block anyway, so the claim can be checked
        // against its value. See gather_jacobian().
        if (!is_zero || m_verify_zeros) add_block(specs[i], specs[j], i, j);
      }

    m_N = 0;
    for (const auto& s : m_state) m_N += s->rows();
    m_x.setZero(m_N);
    m_R.setZero(m_N);
    m_dx.setZero(m_N);
    m_J.setZero(m_N, m_N);

    if (m_strain_derivative) {
      m_W = sym_k::width;
      m_dR_de.setZero(m_N, m_W);
      m_dx_de.setZero(m_N, m_W);
    }

    // Graph-driven mode: bind solve() to the first unknown so that a downstream
    // update_source() on it drives the iteration, as backward_euler does with
    // "delta".
    if (auto p = base::m_property_handler.find(base::m_name, specs.front().name))
      (*p)->traits().update = [this]() { this->update(); };
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("function").template add<is_required>();
    para.template insert<std::vector<unknown_spec>>("unknowns")
        .template add<is_required>();
    para.template insert<value_type>("tolerance")
        .template add<set_default>(value_type{1e-10});
    // Backward-error tolerance for the linear solve; guards a singular Jacobian.
    para.template insert<value_type>("linear_tolerance")
        .template add<set_default>(value_type{1e-8});
    para.template insert<int>("max_iter").template add<set_default>(50);
    // Opt-in: wires d_residual_<name>_d_strain inputs and produces
    // d_<name>_d_strain outputs, solved from the converged factorization.
    para.template insert<bool>("strain_derivative")
        .template add<set_default>(false);
    // Optional, and read with contains(): a block set naming a block that is
    // NOT identically zero converges anyway and only corrupts the consistent
    // tangent, so it is worth being able to check.
    para.template insert<std::vector<block_ref>>("zero_blocks");
    // Opt-in, default off: wires the declared-zero blocks anyway and asserts
    // they really are zero. Turn it on once while developing a model.
    para.template insert<bool>("verify_zero_blocks")
        .template add<set_default>(false);
    return para;
  }

  /// Run the coupled Newton iteration. The converged unknowns are left in the
  /// output properties.
  ///
  /// On failure the output properties hold the last raw iterate — not a
  /// projected or otherwise doctored value. That mirrors backward_euler::solve,
  /// which also returns its raw iterate on the failure paths so that a failed
  /// solve cannot be mistaken for a plausible answer. The difference here is
  /// that those values sit in graph properties, which downstream materials will
  /// read whether or not anyone consulted converged() — so a caller that cares
  /// must check it and react, as small_strain_plasticity does by throwing.
  void update() override { solve(); }

  void solve() {
    m_converged = false;
    gather_state(m_x);   // seed from whatever the properties currently hold

    // Evaluate-first: the residual AFTER the final update is the one tested, so
    // a solve that converges on its last allowed update is reported converged.
    for (int iter = 0; iter <= m_max_iter; ++iter) {
      scatter_state(m_x);
      refresh_sources();
      gather_residual();

      if (m_R.template lpNorm<Eigen::Infinity>() < m_tol) {
        m_converged = true;
        gather_jacobian();      // factorize at the CONVERGED point so that
        m_lu.compute(m_J);      // solve_with_factorization() is usable (#12)
        if (m_strain_derivative) compute_strain_derivative();
        return;
      }
      if (iter == m_max_iter) return;

      gather_jacobian();
      m_lu.compute(m_J);
      m_dx = m_lu.solve(m_R);

      // Guard a singular / rank-deficient Jacobian. A backward-error check is
      // cheaper than rank-revealing pivoting every iteration and catches the
      // same failures; finiteness alone would not.
      if (!m_dx.allFinite()) return;
      const auto rn = m_R.template lpNorm<Eigen::Infinity>();
      const auto back_err =
          (m_J * m_dx - m_R).template lpNorm<Eigen::Infinity>();
      if (back_err > m_lin_tol * rn) return;

      m_x -= m_dx;
    }
  }

  bool converged() const noexcept { return m_converged; }

  /// Size of the flat system.
  std::size_t size() const noexcept { return m_N; }

  /// Reuse the Jacobian factorization from the converged solve, so a consistent
  /// tangent dx/deps can be assembled by solving J * (dx/deps) = -dR/deps
  /// without ever forming an inverse. Accepts a multi-column right-hand side,
  /// so the whole N x 6 strain derivative can be solved in one call.
  ///
  /// The result is only the consistent tangent if J is the EXACT dR/dx. In
  /// particular any block named in 'zero_blocks' must be identically zero, not
  /// an approximation — see the note at the block wiring in the constructor.
  ///
  /// Throws if the last solve did not converge. m_lu would otherwise still hold
  /// the factorization from some earlier iterate — or from an earlier call
  /// entirely — and quietly hand back a plausible-looking tangent built on it.
  template<typename Rhs>
  auto solve_with_factorization(const Rhs& rhs) const {
    if (!m_converged)
      throw std::runtime_error(
          "vector_newton '" + std::string(base::name()) +
          "': solve_with_factorization() requires a converged solve");
    return m_lu.solve(rhs);
  }

private:
  // --- input validation ----------------------------------------------------

  /// Pre-check every name string before a single property is created.
  ///
  /// This has to run first because the property registry cannot report the
  /// problem afterwards: add_property() silently returns the EXISTING property
  /// when the name is already taken, and downcasts it to the requested type
  /// (numsim-core property_registry_interface.h). So a duplicate unknown name
  /// would alias two unknowns onto one storage slot while m_N still counts
  /// them separately — and a duplicate whose kind differs would reinterpret a
  /// property<double> as a property<tensor2>, which is undefined behaviour with
  /// no diagnostic at all.
  static void check_names(const std::vector<unknown_spec>& specs,
                          const std::vector<block_ref>& zeros,
                          std::string_view owner, bool strain_derivative) {
    const auto fail = [owner](const std::string& what) {
      throw std::runtime_error("vector_newton '" + std::string(owner) +
                               "': " + what);
    };

    if (specs.empty()) fail("'unknowns' must not be empty");

    std::unordered_set<std::string> names;
    for (const auto& s : specs) {
      if (s.name.empty()) fail("unknown names must not be empty");
      if (!names.insert(s.name).second)
        fail("duplicate unknown name '" + s.name +
             "' — names must be unique, or the two unknowns would silently "
             "share one property");
    }

    // The wire-up derives property names by concatenation, so distinct unknown
    // sets can collide through an underscore: unknowns {a, b_c, a_b, c} yield
    // "jacobian_a_b_c" from both (a, b_c) and (a_b, c). Rather than forbid
    // underscores — "back_stress" is a reasonable name — just require that
    // every generated name comes out distinct.
    std::unordered_set<std::string> generated;
    const auto claim = [&](const std::string& prop) {
      if (!generated.insert(prop).second)
        fail("unknown names generate a colliding property name '" + prop +
             "' — rename one of the unknowns");
    };
    for (const auto& si : specs) {
      claim("residual_" + si.name);
      for (const auto& sj : specs) claim("jacobian_" + si.name + "_" + sj.name);
      if (strain_derivative) claim("d_residual_" + si.name + "_d_strain");
    }

    // The solver's own outputs share one owner with the state properties, so
    // they need their own uniqueness check — an unknown literally named
    // "d_g_d_strain" would otherwise collide with the derivative of "g".
    if (strain_derivative) {
      auto owned = names;   // the state property names
      for (const auto& s : specs)
        if (!owned.insert("d_" + s.name + "_d_strain").second)
          fail("unknown name 'd_" + s.name +
               "_d_strain' collides with a generated strain-derivative output");
    }

    // A typo here is not harmless: naming a block that the function material
    // does supply would silently drop it from the Jacobian, which degrades
    // convergence with no error anywhere.
    for (const auto& [row, col] : zeros) {
      if (!names.contains(row) || !names.contains(col))
        fail("'zero_blocks' entry [" + row + ", " + col +
             "] names an unknown that is not declared in 'unknowns'");
    }
  }

  // --- construction helpers ------------------------------------------------

  void add_state_and_residual(const unknown_spec& s) {
    switch (s.kind) {
      case unknown_kind::scalar:     emplace_state<scalar_k>(s.name); break;
      case unknown_kind::sym_tensor: emplace_state<sym_k>(s.name);    break;
    }
  }

  template<typename U>
  void emplace_state(const std::string& name) {
    auto& ref = base::template add_output<typename U::value_type>(name);
    m_state.push_back(
        std::make_unique<state_layout<value_type, U>>(ref));

    const auto& in = base::template add_input<typename U::value_type>(
        m_func, "residual_" + name, EdgeKind::Local);
    m_residual.push_back(
        std::make_unique<residual_layout<value_type, U>>(in));

    if (m_strain_derivative) {
      // dR_I/deps has the shape of a Jacobian block whose column kind is the
      // strain, so the existing block layout already serializes it; only the
      // scatter direction on the output side is new.
      using de_block = jacobian_block_layout<value_type, U, sym_k>;
      const auto& din = base::template add_input<typename de_block::value_type>(
          m_func, "d_residual_" + name + "_d_strain", EdgeKind::Local);
      m_dR_de_layout.push_back(std::make_unique<de_block>(din));

      auto& dout = base::template add_output<typename de_block::value_type>(
          "d_" + name + "_d_strain");
      m_dx_de_layout.push_back(
          std::make_unique<strain_derivative_layout<value_type, Dim, U>>(dout));
    }
  }

  // Bounded runtime -> compile-time dispatch on the kind PAIR. The pair matters:
  // (scalar, sym) and (sym, scalar) are both rank-2 tensors but pack as a row
  // and a column respectively, so the value type alone cannot disambiguate.
  void add_block(const unknown_spec& si, const unknown_spec& sj,
                 std::size_t i, std::size_t j) {
    switch (si.kind) {
      case unknown_kind::scalar:     add_block_j<scalar_k>(sj, si.name, i, j); break;
      case unknown_kind::sym_tensor: add_block_j<sym_k>(sj, si.name, i, j);    break;
    }
  }

  template<typename UI>
  void add_block_j(const unknown_spec& sj, const std::string& name_i,
                   std::size_t i, std::size_t j) {
    switch (sj.kind) {
      case unknown_kind::scalar:
        emplace_block<UI, scalar_k>(name_i, sj.name, i, j); break;
      case unknown_kind::sym_tensor:
        emplace_block<UI, sym_k>(name_i, sj.name, i, j); break;
    }
  }

  template<typename UI, typename UJ>
  void emplace_block(const std::string& name_i, const std::string& name_j,
                     std::size_t i, std::size_t j) {
    using block_t = jacobian_block_layout<value_type, UI, UJ>;
    const auto& in = base::template add_input<typename block_t::value_type>(
        m_func, "jacobian_" + name_i + "_" + name_j, EdgeKind::Local);
    m_jacobian[i][j] = std::make_unique<block_t>(in);
  }

  // --- serialization -------------------------------------------------------

  void scatter_state(const vector_t& x) {
    std::size_t r = 0;
    for (auto& s : m_state) { s->scatter(x.data(), r); r += s->rows(); }
  }

  void gather_state(vector_t& x) {
    if (x.size() != static_cast<Eigen::Index>(m_N)) x.setZero(m_N);
    std::size_t r = 0;
    for (auto& s : m_state) { s->gather(x.data(), r, 0, m_N); r += s->rows(); }
  }

  /// Re-evaluate the function material once for the current iterate.
  ///
  /// update_source() fires the update callback of the SPECIFIC property it is
  /// called on, so this must touch every input, not a chosen one. A material
  /// may bind its compute to any of its outputs — including a Jacobian block —
  /// and narrowing this to the residual inputs makes such a material silently
  /// gather default-constructed values: with R == 0 the solver then reports
  /// convergence at whatever the initial iterate happened to be.
  ///
  /// Inputs whose source carries no callback cost an empty std::function
  /// check, and in the normal case exactly one callback fires, so compute runs
  /// once per iteration rather than once per gather.
  void refresh_sources() {
    for (auto& L : m_residual) L->update_source();
    for (auto& row : m_jacobian)
      for (auto& B : row)
        if (B) B->update_source();
    for (auto& L : m_dR_de_layout) L->update_source();
  }

  /// Consistent tangent contribution, by the implicit function theorem.
  ///
  ///     R(x(eps), eps) = 0   =>   J * dx/deps = -dR/deps
  ///
  /// Runs only after convergence, reusing the factorization of J at the
  /// converged point — so no inverse is ever formed, and the N x W system is a
  /// single multi-column back-substitution.
  ///
  /// The material completes the chain rule itself:
  ///     dsigma/deps = dsigma/deps|_x + (dsigma/dx) : (dx/deps)
  ///
  /// Correct only if J is the EXACT dR/dx — see the note on 'zero_blocks'.
  void compute_strain_derivative() {
    m_dR_de.setZero();
    std::size_t r = 0;
    for (auto& L : m_dR_de_layout) {
      L->gather(m_dR_de.data(), r, 0, m_N);
      r += L->rows();
    }

    m_dx_de.noalias() = -m_lu.solve(m_dR_de);

    r = 0;
    for (auto& L : m_dx_de_layout) {
      L->scatter_block(m_dx_de.data(), r, 0, m_N);
      r += L->rows();
    }
  }

  /// Pack the residual. Pure serialization — refresh_sources() has already run.
  void gather_residual() {
    std::size_t r = 0;
    for (auto& L : m_residual) {
      L->gather(m_R.data(), r, 0, m_N);
      r += L->rows();
    }
  }

  /// Pack the Jacobian. Pure serialization, as above.
  void gather_jacobian() {
    m_J.setZero();   // absent blocks stay structurally zero
    std::size_t r = 0;
    for (std::size_t i = 0; i < m_state.size(); ++i) {
      std::size_t c = 0;
      for (std::size_t j = 0; j < m_state.size(); ++j) {
        if (auto& B = m_jacobian[i][j]) B->gather(m_J.data(), r, c, m_N);
        c += m_state[j]->rows();
      }
      r += m_state[i]->rows();
    }
    if (m_verify_zeros) verify_zero_blocks();
  }

  /// Check that every block declared zero actually is, then clear it so the
  /// solve proceeds exactly as it would in production.
  ///
  /// Only runs in verify mode, where the blocks were wired for this purpose.
  /// The claim is otherwise taken on trust and is unfalsifiable from the
  /// outside: a wrong one still converges -- the residual pins the root -- and
  /// shows up only as a wrong consistent tangent and slow global Newton.
  void verify_zero_blocks() {
    std::vector<std::size_t> off(m_state.size() + 1, 0);
    for (std::size_t k = 0; k < m_state.size(); ++k)
      off[k + 1] = off[k] + m_state[k]->rows();

    for (const auto& [i, j] : m_zero_ij) {
      const auto rows = off[i + 1] - off[i], cols = off[j + 1] - off[j];
      const auto blk = m_J.block(off[i], off[j], rows, cols);
      const auto peak = blk.template lpNorm<Eigen::Infinity>();
      if (peak > m_tol)
        throw std::runtime_error(
            "vector_newton '" + base::name() + "': 'zero_blocks' entry [" +
            std::to_string(i) + ", " + std::to_string(j) + "] is NOT zero " +
            "(largest entry " + std::to_string(peak) +
            ") — the solve would still converge, but the consistent tangent "
            "would be wrong");
      m_J.block(off[i], off[j], rows, cols).setZero();
    }
  }

private:
  const std::string& m_func;
  const value_type& m_tol;
  const value_type& m_lin_tol;
  const int& m_max_iter;

  std::vector<std::unique_ptr<state_layout_base<value_type>>> m_state;
  std::vector<std::unique_ptr<layout_base<value_type>>> m_residual;
  std::vector<std::vector<std::unique_ptr<layout_base<value_type>>>> m_jacobian;

  const bool& m_strain_derivative;
  std::vector<std::unique_ptr<layout_base<value_type>>> m_dR_de_layout;
  std::vector<std::unique_ptr<block_scatter_base<value_type>>> m_dx_de_layout;

  std::size_t m_N{0};
  std::size_t m_W{0};
  vector_t m_x, m_R, m_dx;
  matrix_t m_J;
  const bool m_verify_zeros;
  /// (row, col) of every block declared zero, for verify mode.
  std::vector<std::pair<std::size_t, std::size_t>> m_zero_ij;
  matrix_t m_dR_de, m_dx_de;
  Eigen::PartialPivLU<matrix_t> m_lu;
  bool m_converged{false};
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_VECTOR_NEWTON_H
