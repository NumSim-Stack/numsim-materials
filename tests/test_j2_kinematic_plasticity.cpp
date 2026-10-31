/// j2_kinematic_plasticity: the first shipped material on vector_newton (#162).
///
/// gamma = 0 (Prager) has a closed form and is the oracle. gamma > 0
/// (Armstrong-Frederick) has none; it is checked against the equations it
/// claims to satisfy, and against finite differences for the derivatives.
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/unknown_spec.h"
#include "numsim-materials/materials/j2_kinematic_plasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/solvers/vector_newton.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/umat/errors.h"
#include "numsim-materials/umat/statev_map.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using t2 = tmech::tensor<T, 3, 2>;
using t4 = tmech::tensor<T, 3, 4>;

T sigma_eq(const t2& x) { return std::sqrt(T{1.5} * tmech::dcontract(x, x)); }

/// EXPECT_THROW pinned to the message, so a different throw -- a solve that
/// did not converge, a negative multiplier -- cannot stand in for the one
/// the test is about.
template <typename Fn>
void expect_throw_saying(Fn&& fn, const char* needle) {
  try {
    fn();
    ADD_FAILURE() << "expected a std::runtime_error saying '" << needle << "'";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
        << "wrong throw: " << e.what();
  }
}

/// A strain that is loaded through the solver, and a second one that is not
/// proportional to it, so the committed back stress is not parallel to the
/// second trial deviator.
t2 strain_a() {
  t2 e; e.fill(T{0});
  e(0, 0) = 0.010; e(1, 1) = -0.003; e(0, 1) = e(1, 0) = 0.004;
  return e;
}
t2 strain_b() {
  t2 e; e.fill(T{0});
  e(0, 0) = -0.002; e(2, 2) = 0.006; e(0, 1) = e(1, 0) = 0.012;
  return e;
}

struct rig {
  static constexpr T K = 166.67, G = 76.92, sigma_0 = 1.0, H_iso = 5.0, C = 30.0;
  T H_mod;

  ctx_type ctx;
  nm::external_strain_source<policy>* src{};
  nm::vector_newton<policy>* solver{};

  explicit rig(T gamma, int max_iter = 50, bool graph_driven = false, T hardening = H_iso)
      : H_mod(hardening) {
    param_type p;
    p.insert<std::string>("name", "stepper");
    src = &ctx.create<nm::external_strain_source<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "solver");
    p.insert<std::string>("function", "kin");
    p.insert<T>("tolerance", T{1e-12});
    p.insert<int>("max_iter", max_iter);
    p.insert<bool>("strain_derivative", true);
    p.insert<bool>("graph_driven", graph_driven);
    p.insert<std::vector<nm::unknown_spec>>(
        "unknowns", {{"dlambda", nm::unknown_kind::scalar},
                     {"dbeta", nm::unknown_kind::sym_tensor}});
    solver = &ctx.create<nm::vector_newton<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "hardening");
    p.insert<std::string>("source", "kin");
    p.insert<T>("K", H_mod);
    ctx.create<nm::linear_isotropic_hardening<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "kin");
    p.insert<std::string>("hardening_source", "hardening");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<std::string>("solver_source", "solver");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    p.insert<T>("sigma_0", sigma_0);
    p.insert<T>("C", C);
    p.insert<T>("gamma", gamma);
    ctx.create<nm::j2_kinematic_plasticity<policy>>(p);
    ctx.finalize();
  }

  void step(const t2& eps) {
    src->bind(eps, eps);
    ctx.update();
  }

  const t2& stress() { return ctx.get<t2>("kin", "stress"); }
  const t4& tangent() { return ctx.get<t4>("kin", "tangent"); }
  const t2& eps_p() { return ctx.get<t2>("kin", "plastic_strain"); }
  const t2& beta() { return ctx.get<t2>("kin", "back_stress"); }
  T kappa() { return ctx.get<T>("kin", "equivalent_plastic_strain"); }
  T dlambda() { return ctx.get<T>("solver", "dlambda"); }

  /// sigma_eq(dev sigma - beta) - (sigma_0 + H kappa): zero on the surface.
  T shifted_yield() {
    const t2 xi{tmech::dev(stress()) - beta()};
    return sigma_eq(xi) - (sigma_0 + H_iso * kappa());
  }
};

/// Prager: eta stays parallel to eta_tr, so the multiplier is
/// (q_tr - sigma_0 - H_n)/(3G + C + H'). Checked on a second step whose
/// committed back stress is NOT parallel to the new trial deviator, so the
/// closed form is reached through a real update of all three histories.
TEST(J2Kinematic, PragerStepMatchesTheClosedForm) {
  rig r(0.0);
  r.step(strain_a());
  ASSERT_GT(r.kappa(), 1e-6) << "the first step must yield";
  r.ctx.commit();

  const t2 eps_p_n{r.eps_p()};
  const t2 beta_n{r.beta()};
  const T kappa_n = r.kappa();

  const t2 s_tr{T{2} * rig::G * tmech::dev(t2(strain_b() - eps_p_n))};
  const t2 eta_tr{s_tr - beta_n};
  const T q_tr = sigma_eq(eta_tr);
  ASSERT_LT(std::abs(tmech::dcontract(beta_n, eta_tr)),
            0.95 * tmech::norm(beta_n) * tmech::norm(eta_tr))
      << "beta_n must not be parallel to the trial deviator";

  const T dl_ref = (q_tr - rig::sigma_0 - rig::H_iso * kappa_n) /
                   (T{3} * rig::G + rig::C + rig::H_iso);
  const t2 N_tr{T{1.5} * eta_tr / q_tr};
  const t2 beta_ref{beta_n + (T{2} / T{3} * rig::C * dl_ref) * N_tr};
  ASSERT_GT(dl_ref, 1e-6);

  r.step(strain_b());
  EXPECT_TRUE(r.solver->converged());
  EXPECT_NEAR(r.dlambda(), dl_ref, 1e-10 * dl_ref);
  EXPECT_LT(tmech::norm(t2{r.beta() - beta_ref}), 1e-9);
  EXPECT_NEAR(r.kappa(), kappa_n + dl_ref, 1e-12);
  EXPECT_NEAR(r.shifted_yield(), 0.0, 1e-9);
}

/// Prager converges in one Newton step: the seed is the trial state (dbeta = 0)
/// and the first step IS the radial return, whatever beta_n is. Armstrong-
/// Frederick does not, which is what makes it a solver test rather than a
/// packing test. Both facts are pinned so a change in either direction is
/// noticed.
TEST(J2Kinematic, OnlyRecoveryNeedsMoreThanOneNewtonStep) {
  rig prager(0.0, /*max_iter=*/1);
  prager.step(strain_a());
  prager.ctx.commit();
  EXPECT_NO_THROW(prager.step(strain_b()));
  EXPECT_TRUE(prager.solver->converged());

  rig af(40.0, /*max_iter=*/1);
  expect_throw_saying([&] { af.step(strain_a()); }, "did not converge");
  EXPECT_FALSE(af.solver->converged())
      << "with recovery the return is not radial and one step cannot land";
}

/// Armstrong-Frederick has no closed form. The converged state has to satisfy
/// the equations it was solved from -- on the shifted surface, with the
/// evolution law met by the FINAL flow direction -- and the flow direction
/// must have rotated away from the trial one, or the recovery term did nothing.
TEST(J2Kinematic, RecoveryReturnsOnTheShiftedSurfaceAndRotatesTheFlow) {
  rig r(40.0);
  r.step(strain_a());
  r.ctx.commit();
  const t2 beta_n{r.beta()};
  const t2 eps_p_n{r.eps_p()};

  r.step(strain_b());
  ASSERT_TRUE(r.solver->converged());
  const T dl = r.dlambda();
  ASSERT_GT(dl, 1e-6);

  EXPECT_NEAR(r.shifted_yield(), 0.0, 1e-9);

  const t2 xi{tmech::dev(r.stress()) - r.beta()};
  const t2 N{T{1.5} * xi / sigma_eq(xi)};
  const t2 evolution{(T{1} + 40.0 * dl) * r.beta() - beta_n -
                     (T{2} / T{3} * rig::C * dl) * N};
  EXPECT_LT(tmech::norm(evolution), 1e-9) << "Armstrong-Frederick not satisfied";
  EXPECT_LT(tmech::norm(t2{r.eps_p() - eps_p_n - dl * N}), 1e-9)
      << "plastic strain must follow the final flow direction";

  const t2 eta_tr{T{2} * rig::G * tmech::dev(t2(strain_b() - eps_p_n)) - beta_n};
  const T cosine = tmech::dcontract(xi, eta_tr) /
                   (tmech::norm(xi) * tmech::norm(eta_tr));
  EXPECT_LT(cosine, 1.0 - 1e-6)
      << "the return stayed radial; recovery had no effect on the direction";
}

/// Below yield nothing is solved: the stress is the elastic one, the history
/// is carried, and the solver has never run. The second half loads plastically
/// first, so the solver's slots hold a converged dl and dx/deps when the
/// elastic step comes; the engine fires compute_residual() with whatever is
/// there, and that used to add the stale dl to kappa. The strain derivatives
/// are exported properties too and must not carry over either.
TEST(J2Kinematic, AnElasticStepDoesNotRunTheSolver) {
  const t4 C_e{nm::plasticity_detail::make_isotropic_tangent<T, 3>(rig::K, rig::G)};
  rig r(40.0);

  const t2 small{0.1 * strain_a()};
  r.step(small);
  EXPECT_FALSE(r.solver->converged()) << "converged() is false until a solve ran";
  EXPECT_DOUBLE_EQ(r.kappa(), 0.0);
  EXPECT_EQ(tmech::norm(r.beta()), 0.0);
  EXPECT_LT(tmech::norm(t2{r.stress() - tmech::dcontract(C_e, small)}), 1e-12);
  EXPECT_TRUE(tmech::almost_equal(r.tangent(), C_e, 1e-12));

  r.step(strain_a());
  ASSERT_TRUE(r.solver->converged());
  r.ctx.commit();
  const T kappa_n = r.kappa();
  const t2 beta_n{r.beta()};
  const t2 eps_p_n{r.eps_p()};
  ASSERT_GT(tmech::norm(r.ctx.get<t2>("solver", "d_dlambda_d_strain")), 0.1);
  ASSERT_GT(tmech::norm(r.ctx.get<t4>("solver", "d_dbeta_d_strain")), 0.1);

  const t2 unloaded{0.9 * strain_a()};
  r.step(unloaded);
  EXPECT_FALSE(r.solver->converged()) << "an elastic step leaves no converged solve";
  EXPECT_DOUBLE_EQ(r.kappa(), kappa_n);
  EXPECT_EQ(tmech::norm(t2{r.beta() - beta_n}), 0.0);
  EXPECT_DOUBLE_EQ(r.dlambda(), 0.0) << "the unknowns are zero, not stale";
  EXPECT_EQ(tmech::norm(r.ctx.get<t2>("solver", "d_dlambda_d_strain")), 0.0)
      << "dx/deps is zero, not the previous point's";
  EXPECT_EQ(tmech::norm(r.ctx.get<t4>("solver", "d_dbeta_d_strain")), 0.0);
  EXPECT_LT(tmech::norm(t2{r.stress() - tmech::dcontract(C_e, t2{unloaded - eps_p_n})}),
            1e-12);
  EXPECT_TRUE(tmech::almost_equal(r.tangent(), C_e, 1e-12));
}

/// The material needs the solver to itself. A graph-driven solver would run a
/// second, cold-start solve at every point, elastic ones included. Rejected at
/// the first update, here an elastic one, not at the first yield.
TEST(J2Kinematic, RejectsAGraphDrivenSolver) {
  rig r(40.0, 50, /*graph_driven=*/true);
  expect_throw_saying([&] { r.step(t2{0.1 * strain_a()}); }, "graph_driven = false");
}

/// A failed return map is retryable at a smaller step; a misconfigured solver is not.
TEST(J2Kinematic, ANonConvergedReturnMapAsksForASmallerStep) {
  rig af(40.0, /*max_iter=*/1);
  EXPECT_THROW(af.step(strain_a()), nm::umat::cutback_error);
}

TEST(J2Kinematic, ANegativeMultiplierAsksForASmallerStep) {
  rig soft(0.0, 50, false, /*H_iso=*/-1000.0);
  EXPECT_THROW(soft.step(strain_a()), nm::umat::cutback_error);
}

TEST(J2Kinematic, AGraphDrivenSolverIsAConfigurationErrorNotACutback) {
  rig r(40.0, 50, /*graph_driven=*/true);
  try {
    r.step(t2{0.1 * strain_a()});
    FAIL() << "a graph-driven solver was accepted";
  } catch (const nm::umat::cutback_error& e) {
    FAIL() << "retrying a smaller step cannot fix the configuration: " << e.what();
  } catch (const std::runtime_error&) {
  }
}

/// Jacobian blocks against central differences of the residual, as directional
/// derivatives along symmetric directions: J applied to a direction must equal
/// the derivative of R along it. Directional, because the Mandel space the
/// solver works in sees IIsym as the identity, so perturbing one off-diagonal
/// component alone would compare against the wrong object.
///
/// The residual is re-evaluated at a perturbed iterate by writing the solver's
/// unknowns and pulling residual_dlambda through the graph; the trial state
/// stays the one compute() cached.
TEST(J2Kinematic, JacobianMatchesDirectionalFiniteDifferences) {
  rig r(40.0);
  r.step(strain_a());
  r.ctx.commit();
  r.step(strain_b());
  ASSERT_TRUE(r.solver->converged());

  auto& dl_slot = r.ctx.get_mutable<T>("solver", "dlambda");
  auto& b_slot = r.ctx.get_mutable<t2>("solver", "dbeta");
  // Off the root, so no block is evaluated where a term vanishes.
  const T dl0 = dl_slot * 0.7;
  const t2 b0{b_slot * 0.8};

  auto R = [&](T dl, const t2& b) {
    dl_slot = dl; b_slot = b;
    r.ctx.update_property("kin", "residual_dlambda");
    return std::pair<T, t2>{r.ctx.get<T>("kin", "residual_dlambda"),
                            r.ctx.get<t2>("kin", "residual_dbeta")};
  };
  R(dl0, b0);
  const T j_ll = r.ctx.get<T>("kin", "jacobian_dlambda_dlambda");
  const t2 j_lb{r.ctx.get<t2>("kin", "jacobian_dlambda_dbeta")};
  const t2 j_bl{r.ctx.get<t2>("kin", "jacobian_dbeta_dlambda")};
  const t4 j_bb{r.ctx.get<t4>("kin", "jacobian_dbeta_dbeta")};

  const T h = 1e-6;
  {
    const auto [rp, Rp] = R(dl0 + h, b0);
    const auto [rm, Rm] = R(dl0 - h, b0);
    EXPECT_NEAR((rp - rm) / (2 * h), j_ll, 1e-6 * std::abs(j_ll)) << "J_ll";
    EXPECT_LT(tmech::norm(t2{(Rp - Rm) / (2 * h) - j_bl}), 1e-6 * tmech::norm(j_bl))
        << "J_bl";
  }
  const t2 directions[] = {t2{1, 0, 0, 0, 0, 0, 0, 0, 0},
                           t2{0, 1, 0, 1, 0, 0, 0, 0, 0},
                           t2{0.3, -0.2, 0.5, -0.2, 0.1, 0.4, 0.5, 0.4, -0.7}};
  for (const auto& d : directions) {
    const auto [rp, Rp] = R(dl0, t2{b0 + h * d});
    const auto [rm, Rm] = R(dl0, t2{b0 - h * d});
    EXPECT_NEAR((rp - rm) / (2 * h), tmech::dcontract(j_lb, d), 1e-6) << "J_lb";
    EXPECT_LT(tmech::norm(t2{(Rp - Rm) / (2 * h) - tmech::dcontract(j_bb, d)}),
              1e-6)
        << "J_bb";
  }
}

/// The consistent tangent against central differences of the stress, on a
/// plastic step with a committed, non-parallel history. Each perturbed update
/// re-solves from the same committed state (no commit in between), so this is
/// the derivative the implicit function theorem is supposed to reproduce.
TEST(J2Kinematic, TangentMatchesDirectionalFiniteDifferencesOfTheStress) {
  for (const T gamma : {0.0, 40.0}) {
    rig r(gamma);
    r.step(strain_a());
    r.ctx.commit();
    r.step(strain_b());
    ASSERT_TRUE(r.solver->converged());
    const t4 tangent{r.tangent()};

    const T h = 1e-6;
    const t2 directions[] = {t2{1, 0, 0, 0, 0, 0, 0, 0, 0},
                             t2{0, 1, 0, 1, 0, 0, 0, 0, 0},
                             t2{0.3, -0.2, 0.5, -0.2, 0.1, 0.4, 0.5, 0.4, -0.7}};
    for (const auto& d : directions) {
      r.step(t2{strain_b() + h * d});
      const t2 sp{r.stress()};
      r.step(t2{strain_b() - h * d});
      const t2 sm{r.stress()};
      const t2 fd{(sp - sm) / (2 * h)};
      const t2 an{tmech::dcontract(tangent, d)};
      EXPECT_LT(tmech::norm(t2{fd - an}), 1e-6 * tmech::norm(an))
          << "gamma = " << gamma << ", |fd - an| = " << tmech::norm(t2{fd - an});
    }
  }
}

/// Bauschinger: after forward shear the reverse elastic limit sits at
/// beta_12 - Y/sqrt(3), not at -Y/sqrt(3). Both sides of that point are
/// probed from the same committed state.
TEST(J2Kinematic, ReverseYieldIsShiftedByTheBackStress) {
  rig r(40.0);
  auto shear = [](T e) { t2 x; x.fill(T{0}); x(0, 1) = x(1, 0) = e; return x; };

  r.step(shear(0.01));
  ASSERT_GT(r.kappa(), 1e-6);
  r.ctx.commit();
  const T beta12 = r.beta()(0, 1);
  const T kappa_n = r.kappa();
  const T ep12 = r.eps_p()(0, 1);
  ASSERT_GT(beta12, 0.05 * rig::sigma_0) << "the surface must have translated";

  // Pure shear: s12 = 2G (e12 - ep12), sigma_eq(s - beta) = sqrt(3) |s12 - beta12|.
  const T Y = rig::sigma_0 + rig::H_iso * kappa_n;
  const T e_reverse = ep12 + (beta12 - Y / std::sqrt(T{3})) / (T{2} * rig::G);
  const T e_forward = ep12 + (beta12 + Y / std::sqrt(T{3})) / (T{2} * rig::G);
  ASSERT_LT(std::abs(e_reverse), e_forward)
      << "the reverse limit must be nearer than the forward one";

  const T delta = 1e-5;
  r.step(shear(e_reverse + delta));
  EXPECT_DOUBLE_EQ(r.kappa(), kappa_n) << "just inside the reverse limit: elastic";
  r.step(shear(e_reverse - delta));
  EXPECT_GT(r.kappa(), kappa_n) << "just past the reverse limit: plastic";
  r.step(shear(-e_forward));
  EXPECT_GT(r.kappa(), kappa_n)
      << "the mirror of the forward limit is well past reverse yield";
}

/// The back stress is stress-like: STATEV holds beta_12 as is, not doubled like a strain.
TEST(J2Kinematic, BackStressIsStoredAsAStress) {
  rig r(0.0);
  r.step(strain_a());
  ASSERT_GT(std::abs(r.beta()(0, 1)), 1e-6) << "the step must yield with shear";
  const nm::umat::statev_map<policy> map{r.ctx, {{"stepper", "strain"}}};
  ASSERT_EQ(map.describe()[0], "0..5  kin::back_stress  [stress]");
  std::vector<T> statev(map.nstatv());
  map.pack(statev.data());
  EXPECT_EQ(statev[3], r.beta()(0, 1));
}

}  // namespace
