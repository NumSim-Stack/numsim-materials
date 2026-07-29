#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/solvers/vector_newton.h"

// nlohmann/json is an optional dependency (see README), so the JSON round-trip
// below degrades to nothing rather than making the suite fail to build.
#if __has_include(<nlohmann/json.hpp>)
#define NUMSIM_HAVE_JSON 1
#include <nlohmann/json.hpp>
#include "numsim-materials/default_materials.h"
#include "numsim-materials/io/json_material_factory.h"
#endif

namespace {

using namespace numsim::materials;

using policy = material_policy_default;
using T = policy::value_type;
using ctx_type = material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using solver_type = vector_newton<policy>;

// ---------------------------------------------------------------------------
// Test function material: two coupled scalar unknowns.
// ---------------------------------------------------------------------------
//
// Mirrors autocatalytic_reaction's coupling: it reads the solver's trial state
// over Local edges (which the topological sort excludes, breaking the cycle)
// and publishes residual/Jacobian pieces the solver gathers.
//
// The compute callback is bound to one property here, but the solver does not
// rely on that: it fires update_source() on every residual input, so a material
// may bind wherever it likes. It does NOT re-fire for the Jacobian blocks,
// since this one callback produces residual and Jacobian together.

enum class system_mode { linear, nonlinear, singular, decoupled };

class scalar_system_2 final : public material_base<scalar_system_2, policy> {
public:
  using base = material_base<scalar_system_2, policy>;

  template<typename... Args>
  explicit scalar_system_2(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_rx(base::add_output<T>("residual_x", &scalar_system_2::compute)),
        m_ry(base::add_output<T>("residual_y")),
        m_jxx(base::add_output<T>("jacobian_x_x")),
        m_jxy(base::add_output<T>("jacobian_x_y")),
        m_jyx(base::add_output<T>("jacobian_y_x")),
        m_jyy(base::add_output<T>("jacobian_y_y")),
        m_mode(static_cast<system_mode>(base::get_parameter<int>("mode"))),
        m_a(base::get_parameter<T>("a")),
        m_b(base::get_parameter<T>("b")),
        m_solver(base::get_parameter<std::string>("solver_name")),
        m_x(base::add_input<T>(m_solver, "x", EdgeKind::Local)),
        m_y(base::add_input<T>(m_solver, "y", EdgeKind::Local)) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.insert<std::string>("solver_name").add<is_required>();
    para.insert<int>("mode").add<set_default>(0);
    para.insert<T>("a").add<set_default>(T{0});
    para.insert<T>("b").add<set_default>(T{0});
    return para;
  }

  void compute() {
    const auto x = m_x.get();
    const auto y = m_y.get();
    switch (m_mode) {
      case system_mode::linear:
        // [2 1; 1 3] (x,y) = (a,b) — SPD, so Newton lands in one update.
        m_rx = 2 * x + y - m_a;   m_jxx = 2;  m_jxy = 1;
        m_ry = x + 3 * y - m_b;   m_jyx = 1;  m_jyy = 3;
        break;
      case system_mode::nonlinear:
        // x^2 + y = a ; x + y^2 = b — genuinely iterates.
        m_rx = x * x + y - m_a;   m_jxx = 2 * x;  m_jxy = 1;
        m_ry = x + y * y - m_b;   m_jyx = 1;      m_jyy = 2 * y;
        break;
      case system_mode::singular:
        // Two identical Jacobian rows AND inconsistent right-hand sides
        // (a != b), so the residual can never vanish. A rank-deficient but
        // *consistent* system is not a failure case — the solver would
        // legitimately land on a point of the solution manifold.
        m_rx = x + y - m_a;   m_jxx = 1;  m_jxy = 1;
        m_ry = x + y - m_b;   m_jyx = 1;  m_jyy = 1;
        break;
      case system_mode::decoupled:
        // dR_x/dy is IDENTICALLY zero, so declaring that block zero is exact
        // and the factorization stays the true Jacobian.
        m_rx = 2 * x - m_a;       m_jxx = 2;  m_jxy = 0;
        m_ry = x + 3 * y - m_b;   m_jyx = 1;  m_jyy = 3;
        break;
    }
  }

private:
  T& m_rx; T& m_ry;
  T& m_jxx; T& m_jxy; T& m_jyx; T& m_jyy;
  system_mode m_mode;
  const T& m_a;
  const T& m_b;
  const std::string& m_solver;
  const input_property<T, property_traits>& m_x;
  const input_property<T, property_traits>& m_y;
};

/// Build a context with the 2-scalar system driven by vector_newton.
struct scalar_fixture {
  ctx_type ctx;
  solver_type* solver{nullptr};

  scalar_fixture(system_mode mode, T a, T b, int max_iter = 50) {
    param_type p;

    p.clear();
    p.insert<std::string>("name", "solver");
    p.insert<std::string>("function", "sys");
    p.insert<int>("max_iter", max_iter);
    p.insert<std::vector<unknown_spec>>(
        "unknowns", {{"x", unknown_kind::scalar}, {"y", unknown_kind::scalar}});
    solver = &ctx.create<solver_type>(p);

    p.clear();
    p.insert<std::string>("name", "sys");
    p.insert<std::string>("solver_name", "solver");
    p.insert<int>("mode", static_cast<int>(mode));
    p.insert<T>("a", a);
    p.insert<T>("b", b);
    ctx.create<scalar_system_2>(p);

    ctx.finalize();
  }
};

// ---------------------------------------------------------------------------
// #14 acceptance
// ---------------------------------------------------------------------------

TEST(VectorNewton, LinearSystemIsExact) {
  scalar_fixture f(system_mode::linear, 5.0, 10.0);
  f.solver->solve();

  ASSERT_TRUE(f.solver->converged());
  // [2 1; 1 3] x = (5,10) -> x = (1,3)
  EXPECT_NEAR(f.ctx.get<T>("solver", "x"), 1.0, 1e-12);
  EXPECT_NEAR(f.ctx.get<T>("solver", "y"), 3.0, 1e-12);
  EXPECT_EQ(f.solver->size(), 2u);
}

TEST(VectorNewton, NonlinearSystemConvergesToAnalyticRoot) {
  // x = 1, y = 2  ->  a = 1 + 2 = 3, b = 1 + 4 = 5
  scalar_fixture f(system_mode::nonlinear, 3.0, 5.0);

  // A single Newton update cannot reach the root of a genuinely nonlinear
  // system, so this exercises the loop rather than a one-step linear solve.
  scalar_fixture one_step(system_mode::nonlinear, 3.0, 5.0, /*max_iter=*/1);
  one_step.solver->solve();
  EXPECT_FALSE(one_step.solver->converged())
      << "must take more than one update — otherwise the test is vacuous";

  f.solver->solve();
  ASSERT_TRUE(f.solver->converged());
  EXPECT_NEAR(f.ctx.get<T>("solver", "x"), 1.0, 1e-9);
  EXPECT_NEAR(f.ctx.get<T>("solver", "y"), 2.0, 1e-9);
}

TEST(VectorNewton, SingularJacobianReportsFailure) {
  scalar_fixture f(system_mode::singular, 4.0, 7.0);   // inconsistent: a != b
  f.solver->solve();

  EXPECT_FALSE(f.solver->converged())
      << "a rank-deficient Jacobian must not be reported as a converged solve";
  const auto x = f.ctx.get<T>("solver", "x");
  const auto y = f.ctx.get<T>("solver", "y");
  EXPECT_TRUE(std::isfinite(x) && std::isfinite(y))
      << "and must not scatter NaN/inf into the state";
}

TEST(VectorNewton, MaxIterExhaustedReportsFailure) {
  scalar_fixture f(system_mode::nonlinear, 3.0, 5.0, /*max_iter=*/0);
  f.solver->solve();
  EXPECT_FALSE(f.solver->converged());
}

TEST(VectorNewton, ConvergesOnExactlyMaxIter) {
  // The linear system needs exactly one update. With evaluate-first ordering
  // the residual after that update is still tested, so max_iter == 1 must be
  // reported converged — this is the off-by-one guard.
  scalar_fixture f(system_mode::linear, 5.0, 10.0, /*max_iter=*/1);
  f.solver->solve();

  EXPECT_TRUE(f.solver->converged())
      << "a solve converging on its last allowed update must not be reported failed";
  EXPECT_NEAR(f.ctx.get<T>("solver", "x"), 1.0, 1e-12);
}

TEST(VectorNewton, GraphDrivenUpdateDrivesTheIteration) {
  // #12: ctx.update() reaches the solver through the graph rather than a
  // direct solve() call.
  scalar_fixture f(system_mode::linear, 5.0, 10.0);
  f.ctx.update();

  EXPECT_TRUE(f.solver->converged());
  EXPECT_NEAR(f.ctx.get<T>("solver", "x"), 1.0, 1e-12);
  EXPECT_NEAR(f.ctx.get<T>("solver", "y"), 3.0, 1e-12);
}

// ---------------------------------------------------------------------------
// #15 acceptance: mixed scalar + symmetric tensor
// ---------------------------------------------------------------------------
//
//   R_g = g + tr(B) - a          (scalar)
//   R_B = B - g*P                (symmetric tensor)
//
// Closed form: B = g*P and g(1 + tr P) = a, so g = a / (1 + tr P).
// Exercises all four block shapes: 1x1, 1x6 row, 6x1 column, 6x6.

// Templated on the policy so the same system runs at Dim 3 and Dim 2 — the
// width-3 sym_tensor path would otherwise never be exercised.
template<typename Policy>
class mixed_system_t final : public material_base<mixed_system_t<Policy>, Policy> {
public:
  using base = material_base<mixed_system_t<Policy>, Policy>;
  using value_type = typename base::value_type;
  static constexpr auto D = base::Dim;
  using t2 = tmech::tensor<value_type, D, 2>;
  using t4 = tmech::tensor<value_type, D, 4>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template<typename... Args>
  explicit mixed_system_t(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_rg(base::template add_output<value_type>("residual_g", &mixed_system_t::compute)),
        m_rB(base::template add_output<t2>("residual_B")),
        m_jgg(base::template add_output<value_type>("jacobian_g_g")),
        m_jgB(base::template add_output<t2>("jacobian_g_B")),
        m_jBg(base::template add_output<t2>("jacobian_B_g")),
        m_jBB(base::template add_output<t4>("jacobian_B_B")),
        m_a(base::template get_parameter<value_type>("a")),
        m_solver(base::template get_parameter<std::string>("solver_name")),
        m_g(base::template add_input<value_type>(m_solver, "g", EdgeKind::Local)),
        m_B(base::template add_input<t2>(m_solver, "B", EdgeKind::Local)) {
    // Any fixed symmetric tensor will do; only tr(P) enters the closed form.
    for (std::size_t i = 0; i < D; ++i)
      for (std::size_t j = 0; j < D; ++j)
        m_P(i, j) = (i == j) ? value_type(0.5) + value_type(i) * value_type(0.2)
                             : value_type(0.1) * value_type(i + j);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("solver_name").template add<is_required>();
    para.template insert<value_type>("a").template add<set_default>(value_type{1});
    return para;
  }

  const t2& P() const noexcept { return m_P; }

  void compute() {
    const auto g = m_g.get();
    const auto& B = m_B.get();
    const auto I = tmech::eye<value_type, D, 2>();
    const auto IIsym = t4{
        (tmech::otimesu(I, I) + tmech::otimesl(I, I)) * value_type{0.5}};

    m_rg = g + tmech::trace(B) - m_a;
    m_rB = B - g * m_P;

    m_jgg = value_type{1};
    m_jgB = t2{I};        // d(tr B)/dB = I, packed as a 1xW row
    m_jBg = t2{-m_P};     // dR_B/dg,     packed as a Wx1 column
    m_jBB = IIsym;        // dR_B/dB
  }

private:
  value_type& m_rg;
  t2& m_rB;
  value_type& m_jgg;
  t2& m_jgB;
  t2& m_jBg;
  t4& m_jBB;
  const value_type& m_a;
  const std::string& m_solver;
  const input_property<value_type, property_traits>& m_g;
  const input_property<t2, property_traits>& m_B;
  t2 m_P{};
};

using mixed_system = mixed_system_t<policy>;

TEST(VectorNewton, MixedScalarAndSymmetricTensorSystem) {
  ctx_type ctx;
  param_type p;
  const T a = 2.0;

  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<std::string>("function", "sys");
  p.insert<std::vector<unknown_spec>>(
      "unknowns", {{"g", unknown_kind::scalar}, {"B", unknown_kind::sym_tensor}});
  auto& solver = ctx.create<solver_type>(p);

  p.clear();
  p.insert<std::string>("name", "sys");
  p.insert<std::string>("solver_name", "solver");
  p.insert<T>("a", a);
  auto& sys = ctx.create<mixed_system>(p);

  ctx.finalize();

  EXPECT_EQ(solver.size(), 7u) << "1 scalar + 1 symmetric 3D tensor";

  solver.solve();
  ASSERT_TRUE(solver.converged());

  const auto trP = tmech::trace(sys.P());
  const T g_ref = a / (T{1} + trP);
  const tensor2 B_ref = g_ref * sys.P();

  EXPECT_NEAR(ctx.get<T>("solver", "g"), g_ref, 1e-10);
  const auto& B = ctx.get<tensor2>("solver", "B");
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      EXPECT_NEAR(B(i, j), B_ref(i, j), 1e-10) << "B(" << i << "," << j << ")";
}

TEST(VectorNewton, MixedSystemIn2D) {
  // Dim comes from the Traits policy, so the width-3 sym_tensor path only runs
  // under a 2D policy — previously untested.
  using policy2d = material_policy_2d;
  using ctx2d = material_context<policy2d>;
  using t2d = tmech::tensor<T, 2, 2>;
  using solver2d = vector_newton<policy2d>;
  using sys2d = mixed_system_t<policy2d>;

  ctx2d ctx;
  param_type p;
  const T a = 2.0;

  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<std::string>("function", "sys");
  p.insert<std::vector<unknown_spec>>(
      "unknowns", {{"g", unknown_kind::scalar}, {"B", unknown_kind::sym_tensor}});
  auto& solver = ctx.create<solver2d>(p);

  p.clear();
  p.insert<std::string>("name", "sys");
  p.insert<std::string>("solver_name", "solver");
  p.insert<T>("a", a);
  auto& sys = ctx.create<sys2d>(p);

  ctx.finalize();

  EXPECT_EQ(solver.size(), 4u) << "1 scalar + 1 symmetric 2D tensor (width 3)";

  solver.solve();
  ASSERT_TRUE(solver.converged());

  const T g_ref = a / (T{1} + tmech::trace(sys.P()));
  const t2d B_ref = g_ref * sys.P();

  EXPECT_NEAR(ctx.get<T>("solver", "g"), g_ref, 1e-10);
  const auto& B = ctx.get<t2d>("solver", "B");
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_NEAR(B(i, j), B_ref(i, j), 1e-10) << "B(" << i << "," << j << ")";
}

TEST(VectorNewton, StructurallyZeroBlockIsSkipped) {
  // The 'decoupled' system has dR_x/dy identically zero, so declaring that
  // block zero is EXACT — it costs a property and a packing step, and leaves
  // the assembled J equal to the true Jacobian.
  //
  // That exactness is the whole contract. 'zero_blocks' must never be used to
  // approximate a nonzero block away: the Newton iteration would still find the
  // root (R == 0 pins it), but the implicit-function-theorem linearization
  // J * dx/deps = -dR/deps inverts J as the actual derivative, so an
  // approximated J silently corrupts the consistent tangent.
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<std::string>("function", "sys");
  p.insert<std::vector<unknown_spec>>(
      "unknowns", {{"x", unknown_kind::scalar}, {"y", unknown_kind::scalar}});
  p.insert<std::vector<block_ref>>("zero_blocks", {{"x", "y"}});
  auto& solver = ctx.create<solver_type>(p);

  p.clear();
  p.insert<std::string>("name", "sys");
  p.insert<std::string>("solver_name", "solver");
  p.insert<int>("mode", static_cast<int>(system_mode::decoupled));
  p.insert<T>("a", 5.0);
  p.insert<T>("b", 10.0);
  ctx.create<scalar_system_2>(p);

  ctx.finalize();
  solver.solve();

  // 2x = 5 -> x = 2.5 ; x + 3y = 10 -> y = 2.5. Exact in one update, because
  // the assembled J is the true Jacobian despite the omitted block.
  ASSERT_TRUE(solver.converged());
  EXPECT_NEAR(ctx.get<T>("solver", "x"), 2.5, 1e-12);
  EXPECT_NEAR(ctx.get<T>("solver", "y"), 2.5, 1e-12);

  // The factorization must be usable as the IFT tangent: solving J*z = e_0
  // reproduces the first column of J^-1, which for [[2,0],[1,3]] is (0.5, -1/6).
  Eigen::VectorXd e0(2); e0 << 1.0, 0.0;
  const Eigen::VectorXd z = solver.solve_with_factorization(e0);
  EXPECT_NEAR(z(0), 0.5, 1e-12);
  EXPECT_NEAR(z(1), -1.0 / 6.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------
//
// These must fail at construction. The property registry cannot report the
// problem afterwards: add_property() silently returns the existing property
// when a name is taken and downcasts it to the requested type, so a duplicate
// name aliases two unknowns onto one slot, and a duplicate with a different
// kind reinterprets the storage — undefined behaviour, no diagnostic.

solver_type& make_solver_with(ctx_type& ctx, std::vector<unknown_spec> unknowns,
                              std::vector<block_ref> zeros = {}) {
  param_type p;
  p.insert<std::string>("name", "solver");
  p.insert<std::string>("function", "sys");
  p.insert<std::vector<unknown_spec>>("unknowns", std::move(unknowns));
  if (!zeros.empty()) p.insert<std::vector<block_ref>>("zero_blocks", std::move(zeros));
  return ctx.create<solver_type>(p);
}

TEST(VectorNewtonValidation, RejectsDuplicateUnknownNames) {
  ctx_type ctx;
  EXPECT_THROW(make_solver_with(ctx, {{"x", unknown_kind::scalar},
                                      {"x", unknown_kind::scalar}}),
               std::runtime_error);
}

TEST(VectorNewtonValidation, RejectsDuplicateNameWithDifferentKind) {
  // The dangerous case: property<double> reinterpreted as property<tensor2>.
  ctx_type ctx;
  EXPECT_THROW(make_solver_with(ctx, {{"B", unknown_kind::scalar},
                                      {"B", unknown_kind::sym_tensor}}),
               std::runtime_error);
}

TEST(VectorNewtonValidation, RejectsEmptyUnknownName) {
  ctx_type ctx;
  EXPECT_THROW(make_solver_with(ctx, {{"", unknown_kind::scalar}}),
               std::runtime_error);
}

TEST(VectorNewtonValidation, RejectsEmptyUnknownList) {
  ctx_type ctx;
  EXPECT_THROW(make_solver_with(ctx, {}), std::runtime_error);
}

TEST(VectorNewtonValidation, RejectsUnderscoreCollidingNames) {
  // {a, b_c, a_b, c} generates "jacobian_a_b_c" from both (a, b_c) and (a_b, c).
  ctx_type ctx;
  EXPECT_THROW(make_solver_with(ctx, {{"a", unknown_kind::scalar},
                                      {"b_c", unknown_kind::scalar},
                                      {"a_b", unknown_kind::scalar},
                                      {"c", unknown_kind::scalar}}),
               std::runtime_error);
}

TEST(VectorNewtonValidation, AcceptsPlainUnderscoreNames) {
  // Underscores are not banned outright — only genuine collisions are.
  ctx_type ctx;
  EXPECT_NO_THROW(make_solver_with(ctx, {{"back_stress", unknown_kind::sym_tensor},
                                         {"d_gamma", unknown_kind::scalar}}));
}

TEST(VectorNewtonValidation, RejectsZeroBlockNamingUnknownVariable) {
  // A typo here would otherwise silently drop a real block from the Jacobian.
  ctx_type ctx;
  EXPECT_THROW(make_solver_with(ctx,
                                {{"x", unknown_kind::scalar},
                                 {"y", unknown_kind::scalar}},
                                {{"x", "typo"}}),
               std::runtime_error);
}

TEST(VectorNewtonValidation, FactorizationReuseRequiresConvergence) {
  scalar_fixture f(system_mode::nonlinear, 3.0, 5.0, /*max_iter=*/0);
  f.solver->solve();
  ASSERT_FALSE(f.solver->converged());

  Eigen::VectorXd rhs = Eigen::VectorXd::Ones(2);
  EXPECT_THROW((void)f.solver->solve_with_factorization(rhs), std::runtime_error)
      << "a stale factorization must not be handed out as a tangent";
}

// ---------------------------------------------------------------------------
// Wrong-typed producer property fails to wire
// ---------------------------------------------------------------------------
//
// Claimed as a benefit of per-block typed properties; now actually tested.

class mistyped_system final : public material_base<mistyped_system, policy> {
public:
  using base = material_base<mistyped_system, policy>;

  template<typename... Args>
  explicit mistyped_system(Args&&... args)
      : base(std::forward<Args>(args)...),
        // Declared scalar, but the solver expects a tensor2 for a sym_tensor
        // unknown's residual.
        m_rB(base::add_output<T>("residual_B", &mistyped_system::compute)),
        m_jBB(base::add_output<T>("jacobian_B_B")),
        m_solver(base::get_parameter<std::string>("solver_name")),
        m_B(base::add_input<tensor2>(m_solver, "B", EdgeKind::Local)) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.insert<std::string>("solver_name").add<is_required>();
    return para;
  }

  void compute() { m_rB = T{0}; m_jBB = T{1}; }

private:
  T& m_rB; T& m_jBB;
  const std::string& m_solver;
  const input_property<tensor2, property_traits>& m_B;
};

TEST(VectorNewtonValidation, MistypedProducerPropertyFailsToWire) {
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<std::string>("function", "sys");
  p.insert<std::vector<unknown_spec>>("unknowns", {{"B", unknown_kind::sym_tensor}});
  ctx.create<solver_type>(p);

  p.clear();
  p.insert<std::string>("name", "sys");
  p.insert<std::string>("solver_name", "solver");
  ctx.create<mistyped_system>(p);

  EXPECT_THROW(ctx.finalize(), std::runtime_error)
      << "a scalar residual where a tensor is expected must fail at wire time";
}

// ---------------------------------------------------------------------------
// Callback placement must not matter
// ---------------------------------------------------------------------------
//
// update_source() fires the callback of the specific property it is called on.
// A material is free to bind its compute to any of its outputs, so the solver
// must not depend on that binding landing on a residual property.

class jacobian_bound_system final
    : public material_base<jacobian_bound_system, policy> {
public:
  using base = material_base<jacobian_bound_system, policy>;

  template<typename... Args>
  explicit jacobian_bound_system(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_rx(base::add_output<T>("residual_x")),
        m_ry(base::add_output<T>("residual_y")),
        // compute is bound to a JACOBIAN block, not a residual.
        m_jxx(base::add_output<T>("jacobian_x_x", &jacobian_bound_system::compute)),
        m_jxy(base::add_output<T>("jacobian_x_y")),
        m_jyx(base::add_output<T>("jacobian_y_x")),
        m_jyy(base::add_output<T>("jacobian_y_y")),
        m_solver(base::get_parameter<std::string>("solver_name")),
        m_x(base::add_input<T>(m_solver, "x", EdgeKind::Local)),
        m_y(base::add_input<T>(m_solver, "y", EdgeKind::Local)) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.insert<std::string>("solver_name").add<is_required>();
    return para;
  }

  void compute() {
    const auto x = m_x.get();
    const auto y = m_y.get();
    m_rx = 2 * x + y - T{5};   m_jxx = 2;  m_jxy = 1;
    m_ry = x + 3 * y - T{10};  m_jyx = 1;  m_jyy = 3;
  }

private:
  T& m_rx; T& m_ry;
  T& m_jxx; T& m_jxy; T& m_jyx; T& m_jyy;
  const std::string& m_solver;
  const input_property<T, property_traits>& m_x;
  const input_property<T, property_traits>& m_y;
};

TEST(VectorNewton, WorksWhenComputeIsBoundToAJacobianBlock) {
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "solver");
  p.insert<std::string>("function", "sys");
  p.insert<std::vector<unknown_spec>>(
      "unknowns", {{"x", unknown_kind::scalar}, {"y", unknown_kind::scalar}});
  auto& solver = ctx.create<solver_type>(p);

  p.clear();
  p.insert<std::string>("name", "sys");
  p.insert<std::string>("solver_name", "solver");
  ctx.create<jacobian_bound_system>(p);

  ctx.finalize();
  solver.solve();

  ASSERT_TRUE(solver.converged())
      << "the solver must re-evaluate regardless of which output carries the "
         "material's compute callback";
  EXPECT_NEAR(ctx.get<T>("solver", "x"), 1.0, 1e-12);
  EXPECT_NEAR(ctx.get<T>("solver", "y"), 3.0, 1e-12);
}

// ---------------------------------------------------------------------------
// JSON round-trip
// ---------------------------------------------------------------------------

#ifdef NUMSIM_HAVE_JSON
TEST(VectorNewtonJson, ConfigMatchesHandWiredSetup) {
  register_default_materials<policy>();
  material_factory<policy>::instance()
      .template register_type<scalar_system_2>("scalar_system_2");

  const auto json = nlohmann::json::parse(R"({
    "materials": [
      {"type": "vector_newton", "name": "solver",
       "function": "sys",
       "tolerance": 1e-12,
       "max_iter": 50,
       "unknowns": [
         {"name": "x", "kind": "scalar"},
         {"name": "y", "kind": "scalar"}
       ]},
      {"type": "scalar_system_2", "name": "sys",
       "solver_name": "solver", "mode": 0, "a": 5.0, "b": 10.0}
    ]
  })");

  ctx_type ctx;
  for (const auto& mat : json["materials"]) create_from_json(ctx, mat);
  ctx.finalize();
  ctx.update();

  // Same system as LinearSystemIsExact, reached through the JSON path.
  EXPECT_NEAR(ctx.get<T>("solver", "x"), 1.0, 1e-12);
  EXPECT_NEAR(ctx.get<T>("solver", "y"), 3.0, 1e-12);
}

TEST(VectorNewtonJson, RejectsUnrecognisedKind) {
  register_default_materials<policy>();

  const auto json = nlohmann::json::parse(R"({
    "type": "vector_newton", "name": "solver", "function": "sys",
    "unknowns": [{"name": "x", "kind": "nonsuch"}]
  })");

  ctx_type ctx;
  EXPECT_THROW(create_from_json(ctx, json), std::runtime_error)
      << "an unrecognised kind must fail loudly, not default to scalar";
}
#endif

} // namespace
