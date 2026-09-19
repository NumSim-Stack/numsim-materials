/// J2 with a tensor back-stress, solved as ONE coupled system.
///
/// The acceptance criterion of #15 and the motivating case for vector_newton's
/// mixed scalar/tensor design: the unknowns are {dgamma (scalar), beta
/// (symmetric tensor)} -- 1 + 6 = 7 -- and they are genuinely coupled, because
/// the flow direction depends on beta and beta's evolution depends on the flow
/// direction.
///
/// Until this file, vector_newton had only ever been driven through synthetic
/// systems with closed-form roots. Nothing had put it through a return map.
#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/unknown_spec.h"
#include "numsim-materials/solvers/vector_newton.h"
#include "numsim-materials/umat/external_state_source.h"

namespace {
namespace nm = numsim::materials;
using namespace numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using t2 = tmech::tensor<T, 3, 2>;
using t4 = tmech::tensor<T, 3, 4>;

/// Linear (Prager) kinematic hardening, posed as a coupled residual.
///
/// With eta = s_trial - dev(beta), q = sqrt(3/2 eta:eta) and N = (3/2) eta/q:
///
///     R_g = q - 3G*g - sigma_y                 yield on the RETURNED state,
///                                              because sigma_eq(s - beta)
///                                              collapses to q - 3G*g exactly
///     R_B = beta - beta_n - (2/3)*H*g*N        Prager evolution
///
/// Jacobian, derived rather than differenced:
///
///     dq/dB = -N,   dN/dB = -((3/2)IIdev - N(x)N)/q
///
///     J_gg = -3G                J_gB = -N
///     J_Bg = -(2/3)H N          J_BB = IIsym + (2/3)H g ((3/2)IIdev - N(x)N)/q
///
/// Nothing here pre-substitutes the radial-return simplification: N is taken
/// from the current iterate, so the system really does have to be solved as a
/// coupled 7x7 rather than decoupling into a scalar equation.
template <typename Traits>
class j2_kinematic_system final
    : public nm::material_base<j2_kinematic_system<Traits>, Traits> {
public:
  using base = nm::material_base<j2_kinematic_system<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  static constexpr auto D = base::Dim;
  using tensor2 = tmech::tensor<value_type, D, 2>;
  using tensor4 = tmech::tensor<value_type, D, 4>;

  template <typename... Args>
  explicit j2_kinematic_system(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_rg(base::template add_output<value_type>(
            "residual_g", &j2_kinematic_system::compute_residual)),
        m_rB(base::template add_output<tensor2>("residual_B")),
        m_jgg(base::template add_output<value_type>("jacobian_g_g")),
        m_jgB(base::template add_output<tensor2>("jacobian_g_B")),
        m_jBg(base::template add_output<tensor2>("jacobian_B_g")),
        m_jBB(base::template add_output<tensor4>("jacobian_B_B")),
        m_stress(base::template add_output<tensor2>(
            "stress", &j2_kinematic_system::compute_stress)),
        m_eps_p(base::template add_history_output<tensor2>("plastic_strain")),
        m_beta(base::template add_history_output<tensor2>("back_stress")),
        m_K(base::template get_parameter<value_type>("K")),
        m_G(base::template get_parameter<value_type>("G")),
        m_sigma_y(base::template get_parameter<value_type>("sigma_y")),
        m_H(base::template get_parameter<value_type>("H")),
        m_solver(base::template get_parameter<std::string>("solver_name")),
        m_strain(base::template add_input<tensor2>(
            base::template get_parameter<std::string>("strain_source"),
            "strain", EdgeKind::Global)),
        m_g(base::template add_input<value_type>(m_solver, "g", EdgeKind::Local)),
        m_B(base::template add_input<tensor2>(m_solver, "B", EdgeKind::Local)) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("solver_name").template add<is_required>();
    para.template insert<std::string>("strain_source").template add<is_required>();
    para.template insert<value_type>("K").template add<is_required>();
    para.template insert<value_type>("G").template add<is_required>();
    para.template insert<value_type>("sigma_y").template add<is_required>();
    para.template insert<value_type>("H").template add<is_required>();
    return para;
  }

  /// s_trial = 2G dev(eps - eps_p_n): the deviatoric stress with the plastic
  /// strain frozen at the start of the step.
  tensor2 trial_deviator() const {
    return tensor2{value_type{2} * m_G *
                   tmech::dev(m_strain.get() - m_eps_p.old_value())};
  }

  void compute_residual() {
    const auto g = m_g.get();
    const tensor2 eta{trial_deviator() - tmech::dev(m_B.get())};
    const auto q = std::sqrt(value_type{1.5} * tmech::dcontract(eta, eta));

    const auto I = tmech::eye<value_type, D, 2>();
    const tensor4 IIsym{(tmech::otimesu(I, I) + tmech::otimesl(I, I)) *
                        value_type{0.5}};
    const tensor4 IIdev{IIsym - tmech::otimes(I, I) / value_type{D}};

    // Elastic branch. The trial check is on the SHIFTED trial stress, with
    // beta frozen at beta_n -- the same quantity a scalar J2 return map tests.
    // Without it the system is still solvable: R_g = q - 3G g - sigma_y has a
    // root with NEGATIVE g and a spurious back stress for any trial state, and
    // the solver will find it. Measured before this branch existed, on an
    // elastic step: g = -0.0148 with |beta| = 36.3, converged and wrong.
    //
    // Posing the elastic case as g = 0, beta = beta_n keeps it one system
    // rather than a branch the solver cannot see.
    const tensor2 eta_tr{trial_deviator() - tmech::dev(m_beta.old_value())};
    const auto q_tr =
        std::sqrt(value_type{1.5} * tmech::dcontract(eta_tr, eta_tr));

    if (q_tr <= m_sigma_y || q <= value_type{1e-30}) {
      m_rg = g;
      m_rB = tensor2{m_B.get() - m_beta.old_value()};
      m_jgg = value_type{1};
      m_jgB = tensor2{};
      m_jBg = tensor2{};
      m_jBB = IIsym;
      return;
    }

    const tensor2 N{value_type{1.5} * eta / q};
    const auto two_thirds_H = value_type{2} / value_type{3} * m_H;

    m_rg = q - value_type{3} * m_G * g - m_sigma_y;
    m_rB = tensor2{m_B.get() - m_beta.old_value() - (two_thirds_H * g) * N};

    m_jgg = value_type{-3} * m_G;
    m_jgB = tensor2{-N};
    m_jBg = tensor2{-two_thirds_H * N};
    m_jBB = tensor4{IIsym + (two_thirds_H * g / q) *
                                (value_type{1.5} * IIdev -
                                 tmech::otimes(N, N))};
  }

  /// Runs after the solver, because it reads the solver's outputs. Applies the
  /// converged step and advances the history the driver will commit.
  void compute_stress() {
    const auto g = m_g.get();
    const tensor2 eta{trial_deviator() - tmech::dev(m_B.get())};
    const auto q = std::sqrt(value_type{1.5} * tmech::dcontract(eta, eta));
    const auto I = tmech::eye<value_type, D, 2>();

    if (g <= value_type{0} || q <= value_type{1e-30}) {
      m_eps_p.new_value() = m_eps_p.old_value();
      m_beta.new_value() = m_beta.old_value();
    } else {
      const tensor2 N{value_type{1.5} * eta / q};
      m_eps_p.new_value() = tensor2{m_eps_p.old_value() + g * N};
      m_beta.new_value() = tensor2{m_B.get()};
    }
    const tensor2 eps_e{m_strain.get() - m_eps_p.new_value()};
    m_stress = tensor2{value_type{2} * m_G * tmech::dev(eps_e) +
                       m_K * tmech::trace(eps_e) * I};
  }

private:
  value_type& m_rg;
  tensor2& m_rB;
  value_type& m_jgg;
  tensor2& m_jgB;
  tensor2& m_jBg;
  tensor4& m_jBB;
  tensor2& m_stress;
  history_property<tensor2>& m_eps_p;
  history_property<tensor2>& m_beta;
  const value_type& m_K;
  const value_type& m_G;
  const value_type& m_sigma_y;
  const value_type& m_H;
  const std::string& m_solver;
  const input_property<tensor2, property_traits>& m_strain;
  const input_property<value_type, property_traits>& m_g;
  const input_property<tensor2, property_traits>& m_B;
};


/// The closed form this coupled solve must reproduce.
///
/// For linear kinematic hardening the return is radial in the SHIFTED space:
/// eta stays parallel to eta_trial, so sigma_eq(s - beta) = q_tr - (3G + H)*g
/// and the multiplier has an exact expression. That gives an independent
/// oracle -- the coupled 7x7 is not being checked against itself.
struct closed_form {
  T g;          ///< dgamma
  t2 N;         ///< flow direction
  t2 beta;      ///< converged back stress
  t2 stress;
};

closed_form solve_analytically(const t2& eps, const t2& eps_p_n, const t2& beta_n,
                               T K, T G, T sigma_y, T H) {
  const auto I = tmech::eye<T, 3, 2>();
  const t2 s_tr{T{2} * G * tmech::dev(eps - eps_p_n)};
  const t2 eta_tr{s_tr - tmech::dev(beta_n)};
  const T q_tr = std::sqrt(T{1.5} * tmech::dcontract(eta_tr, eta_tr));

  closed_form r;
  const bool yielding = q_tr > sigma_y;
  r.g = yielding ? (q_tr - sigma_y) / (T{3} * G + H) : T{0};
  r.N = yielding ? t2{T{1.5} * eta_tr / q_tr} : t2{};
  r.beta = t2{beta_n + (T{2} / T{3} * H * r.g) * r.N};
  const t2 eps_p{eps_p_n + r.g * r.N};
  const t2 eps_e{eps - eps_p};
  r.stress = t2{T{2} * G * tmech::dev(eps_e) + K * tmech::trace(eps_e) * I};
  return r;
}

struct rig {
  ctx_type ctx;
  nm::external_strain_source<policy>* src{};
  vector_newton<policy>* solver{};
  static constexpr T K = 166.67, G = 76.92, sigma_y = 1.0, H = 30.0;

  rig() {
    param_type p;
    p.insert<std::string>("name", "stepper");
    src = &ctx.create<nm::external_strain_source<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "solver");
    p.insert<std::string>("function", "kin");
    p.insert<T>("tolerance", T{1e-12});
    p.insert<int>("max_iter", 50);
    p.insert<std::vector<unknown_spec>>(
        "unknowns", {{"g", unknown_kind::scalar},
                     {"B", unknown_kind::sym_tensor}});
    solver = &ctx.create<vector_newton<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "kin");
    p.insert<std::string>("solver_name", "solver");
    p.insert<std::string>("strain_source", "stepper");
    p.insert<T>("K", K);
    p.insert<T>("G", G);
    p.insert<T>("sigma_y", sigma_y);
    p.insert<T>("H", H);
    ctx.create<j2_kinematic_system<policy>>(p);
    ctx.finalize();
  }
};

/// The coupled 7x7 must land on the closed form, and on the shifted yield
/// surface. This is the criterion #15 was accepted with and PR #17 deferred.
TEST(KinematicHardening, CoupledSolveMatchesTheClosedForm) {
  rig r;
  t2 eps; eps.fill(T{0});
  eps(0, 0) = 0.01; eps(1, 1) = -0.003; eps(0, 1) = eps(1, 0) = 0.004;

  r.src->bind(eps, eps);
  r.ctx.update();
  ASSERT_TRUE(r.solver->converged()) << "the coupled return map did not converge";

  const t2 zero{};
  const auto ref = solve_analytically(eps, zero, zero, rig::K, rig::G,
                                      rig::sigma_y, rig::H);
  EXPECT_GT(ref.g, 1e-6) << "the path must actually yield";

  EXPECT_NEAR(r.ctx.get<T>("solver", "g"), ref.g, 1e-10 * ref.g);
  const auto& B = r.ctx.get<t2>("solver", "B");
  const t2 dB{B - ref.beta};
  EXPECT_LT(std::sqrt(tmech::dcontract(dB, dB)), 1e-9)
      << "back stress disagrees with the closed form";

  // The point of the whole system: the RETURNED state sits on the shifted
  // surface, sigma_eq(dev(sigma) - beta) == sigma_y.
  const auto& sig = r.ctx.get<t2>("kin", "stress");
  const t2 xi{tmech::dev(sig) - tmech::dev(B)};
  EXPECT_NEAR(std::sqrt(T{1.5} * tmech::dcontract(xi, xi)), rig::sigma_y, 1e-8)
      << "the converged stress is not on the shifted yield surface";
}


/// The elastic branch, which the coupled system has to encode rather than
/// branch around -- the solver sees one system and cannot be told "skip this".
///
/// Without it, R_g = q - 3G g - sigma_y has a root for ANY trial state: on a
/// step with q_tr = 2.10 against sigma_y = 50 the solver converged happily to
/// g = -0.0148 with a spurious back stress of |beta| = 36.3. Negative plastic
/// flow, reported as success.
TEST(KinematicHardening, AnElasticStepLeavesTheStateAlone) {
  rig r;
  t2 eps; eps.fill(T{0});
  eps(0, 1) = eps(1, 0) = 0.0005;   // well inside the initial surface

  r.src->bind(eps, eps);
  r.ctx.update();
  ASSERT_TRUE(r.solver->converged());

  EXPECT_DOUBLE_EQ(r.ctx.get<T>("solver", "g"), 0.0)
      << "an elastic step must not produce plastic flow";
  const auto& B = r.ctx.get<t2>("solver", "B");
  EXPECT_LT(std::sqrt(tmech::dcontract(B, B)), 1e-12)
      << "an elastic step must not move the back stress";

  const auto& ep = r.ctx.get<t2>("kin", "plastic_strain");
  EXPECT_LT(std::sqrt(tmech::dcontract(ep, ep)), 1e-12);
}

/// The Bauschinger effect: the signature that separates KINEMATIC hardening
/// from isotropic, and the reason a tensor back-stress is worth solving for.
///
/// In pure shear everything lives in the 01 component: s01 = 2G*e01_elastic and
/// sigma_eq = sqrt(3)*|s01|. The surface translates rather than expanding, so
/// the elastic span in s01 between forward yield and reverse yield stays
/// 2*sigma_y/sqrt(3) no matter how far the forward loading went. Under
/// ISOTROPIC hardening that span would grow with the accumulated strain.
TEST(KinematicHardening, ReverseYieldingShowsTheBauschingerEffect) {
  rig r;
  const T step = 2.0e-4;
  auto shear = [](T e) { t2 x; x.fill(T{0}); x(0, 1) = x(1, 0) = e; return x; };
  auto s01 = [&] { return tmech::dev(r.ctx.get<t2>("kin", "stress"))(0, 1); };
  auto yielded = [&] { return r.ctx.get<T>("solver", "g") > 1e-12; };

  // Forward: load well past first yield.
  T e = 0, s_forward = 0;
  for (int i = 1; i <= 60; ++i) {
    e = step * i;
    r.src->bind(shear(e), shear(e));
    r.ctx.update();
    ASSERT_TRUE(r.solver->converged()) << "forward step " << i;
    r.ctx.commit();
  }
  s_forward = s01();
  ASSERT_GT(std::abs(s_forward), 0.0);

  // Reverse until plastic flow resumes, and record where.
  T s_reverse = 0;
  bool resumed = false;
  for (int i = 1; i <= 400 && !resumed; ++i) {
    r.src->bind(shear(e - step * i), shear(e - step * i));
    r.ctx.update();
    ASSERT_TRUE(r.solver->converged()) << "reverse step " << i;
    if (yielded()) { s_reverse = s01(); resumed = true; }
    r.ctx.commit();
  }
  ASSERT_TRUE(resumed) << "reverse yielding never occurred";

  // The elastic span, in stress, between the two yield points.
  const T span = s_forward - s_reverse;
  const T expected = 2.0 * rig::sigma_y / std::sqrt(3.0);
  EXPECT_NEAR(span, expected, 2.0 * 2 * rig::G * step)
      << "elastic span " << span << " vs 2*sigma_y/sqrt(3) = " << expected
      << " -- a span that grew with the forward loading would mean the surface "
         "expanded (isotropic) rather than translated (kinematic)";

  // And the surface really did translate: the back stress is nonzero and
  // aligned with the forward loading.
  const auto& B = r.ctx.get<t2>("solver", "B");
  EXPECT_GT(std::abs(B(0, 1)), 0.1 * rig::sigma_y)
      << "the back stress never moved";
}


/// The hand-derived jacobian, against a finite difference of the residual.
///
/// The three tests above cannot catch a jacobian error, and I checked rather
/// than assumed: flipping the sign of jacobian_g_B leaves all three passing.
/// Newton converges to the root of the RESIDUAL; a wrong jacobian changes only
/// the path taken to get there. So the physics tests verify the physics, and
/// this one verifies the derivation:
///
///     dq/dB = -N        dN/dB = -((3/2)IIdev - N(x)N)/q
///
/// which is the part a consistent tangent would inherit.
TEST(KinematicHardening, TheJacobianMatchesAFiniteDifferenceOfTheResidual) {
  rig r;
  t2 eps; eps.fill(T{0});
  eps(0, 0) = 0.02; eps(1, 1) = -0.006; eps(0, 1) = eps(1, 0) = 0.008;
  r.src->bind(eps, eps);
  r.ctx.update();
  ASSERT_TRUE(r.solver->converged());

  auto* mat = dynamic_cast<j2_kinematic_system<policy>*>(r.ctx.find("kin"));
  ASSERT_NE(mat, nullptr);

  // Evaluate at a state slightly off the solution, so no block is trivially
  // zero and the derivative is not being sampled at a stationary point.
  auto& g_slot = r.ctx.get_mutable<T>("solver", "g");
  auto& B_slot = r.ctx.get_mutable<t2>("solver", "B");
  const T g0 = g_slot * 0.7;
  const t2 B0{B_slot * 0.8};

  auto residuals_at = [&](T g, const t2& B) {
    g_slot = g; B_slot = B;
    mat->compute_residual();
    return std::pair<T, t2>{r.ctx.get<T>("kin", "residual_g"),
                            r.ctx.get<t2>("kin", "residual_B")};
  };

  const auto base = residuals_at(g0, B0);
  const T jgg = r.ctx.get<T>("kin", "jacobian_g_g");
  const t2 jgB{r.ctx.get<t2>("kin", "jacobian_g_B")};
  const t2 jBg{r.ctx.get<t2>("kin", "jacobian_B_g")};
  const t4 jBB{r.ctx.get<t4>("kin", "jacobian_B_B")};

  const T h = 1e-7;
  // d/dg
  {
    const auto plus = residuals_at(g0 + h, B0);
    EXPECT_NEAR((plus.first - base.first) / h, jgg, 1e-4 * std::abs(jgg))
        << "jacobian_g_g";
    const t2 fd{(plus.second - base.second) / h};
    const t2 d{fd - jBg};
    EXPECT_LT(std::sqrt(tmech::dcontract(d, d)),
              1e-4 * std::sqrt(tmech::dcontract(jBg, jBg)) + 1e-8)
        << "jacobian_B_g";
  }
  // d/dB, component by component (symmetric perturbations)
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = i; j < 3; ++j) {
      t2 Bp{B0};
      Bp(i, j) += h;
      if (i != j) Bp(j, i) += h;
      const auto plus = residuals_at(g0, Bp);

      const T fd_g = (plus.first - base.first) / h;
      const T an_g = (i == j) ? jgB(i, j) : jgB(i, j) + jgB(j, i);
      EXPECT_NEAR(fd_g, an_g, 1e-4 * std::abs(an_g) + 1e-7)
          << "jacobian_g_B at (" << i << "," << j << ")";

      for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t l = k; l < 3; ++l) {
          const T fd_B = (plus.second(k, l) - base.second(k, l)) / h;
          const T an_B = (i == j) ? jBB(k, l, i, j)
                                  : jBB(k, l, i, j) + jBB(k, l, j, i);
          EXPECT_NEAR(fd_B, an_B, 1e-4 * std::abs(an_B) + 1e-7)
              << "jacobian_B_B at (" << k << "," << l << ";" << i << "," << j
              << ")";
        }
    }
}

}  // namespace
