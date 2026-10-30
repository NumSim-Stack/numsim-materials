// j2_kinematic_plasticity (#166) as a residual system (#218): a flow node holds the residuals and the history,
// a stress node reads the plastic strain, and the system solves {dlambda, dbeta}. The tangent is the
// derivative the engine accumulates through the system, compared with the shipped material.
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/partial_check.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/core/unknown_spec.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/materials/j2_kinematic_plasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/plasticity_utils.h"
#include "numsim-materials/solvers/system.h"
#include "numsim-materials/solvers/vector_newton.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using t2 = tmech::tensor<T, 3, 2>;
using t4 = tmech::tensor<T, 3, 4>;

constexpr T K{166.67}, G{76.92}, sigma_0{1.0}, H_iso{5.0}, C{30.0};

T equivalent(t2 const &x) { return std::sqrt(T{1.5} * tmech::dcontract(x, x)); }

/// Yield and Armstrong-Frederick residuals of j2_kinematic_plasticity at the unknowns of "plastic", and the
/// history they imply. An elastic trial state turns the yield row into dlambda = 0.
class kinematic_flow final : public nm::material_base<kinematic_flow, policy> {
public:
  using base = nm::material_base<kinematic_flow, policy>;

  template <typename... Args>
  explicit kinematic_flow(Args &&...args)
      : base(std::forward<Args>(args)...), m_r_l(base::add_output<T>("r_l")), m_r_b(base::add_output<t2>("r_b")),
        m_eps_p(base::add_history_output<t2>("plastic_strain")), m_kappa(base::add_history_output<T>("equivalent_plastic_strain")),
        m_beta(base::add_history_output<t2>("back_stress")), m_gamma(base::get_parameter<T>("gamma")),
        m_strain(base::add_input<t2>("stepper", "strain")), m_dl(base::add_input<T>("plastic", "dlambda")),
        m_db(base::add_input<t2>("plastic", "dbeta")) {
    base::add_callback(&kinematic_flow::compute, m_r_l, m_r_b, m_eps_p, m_kappa, m_beta)
        .reads(m_strain, m_dl, m_db)
        .partials(m_r_l, m_strain, m_dl, m_db)
        .partials(m_r_b, m_strain, m_dl, m_db)
        .partials(m_eps_p, m_strain, m_dl, m_db);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<T>(para, "gamma", T{0});
    return para;
  }

  void compute() {
    auto const &IIdev{nm::dev_projector<T, 3>()};
    auto const dl{m_dl.get()};
    auto const kappa_n{m_kappa.old_value()};
    t2 const beta_n{m_beta.old_value()};
    t2 const s_tr{2 * G * tmech::dev(t2{m_strain.get() - m_eps_p.old_value()})};
    t2 const beta{beta_n + m_db.get()};
    t2 const eta{s_tr - tmech::dev(beta)};
    auto const q{equivalent(eta)};
    bool const has_direction{q > 1e-10 * sigma_0};
    t2 const N{has_direction ? t2{T{1.5} * eta / q} : t2{}};
    t4 const A{has_direction ? t4{(T{1.5} * IIdev - tmech::otimes(N, N)) / q} : t4{}};
    bool const plastic{equivalent(t2{s_tr - beta_n}) - sigma_0 - H_iso * kappa_n > 0};
    auto const c{T{2} / 3 * C};

    m_r_l = plastic ? q - 3 * G * dl - sigma_0 - H_iso * (kappa_n + dl) : dl;
    m_r_b = t2{(1 + m_gamma * dl) * beta - beta_n - (c * dl) * N};
    m_eps_p.new_value() = t2{m_eps_p.old_value() + dl * N};
    m_kappa.new_value() = kappa_n + dl;
    m_beta.new_value() = beta;
    if (!base::want_partials())
      return;
    base::partial(m_r_l, m_strain) = plastic ? t2{2 * G * N} : t2{};
    base::partial(m_r_l, m_dl) = plastic ? -3 * G - H_iso : T{1};
    base::partial(m_r_l, m_db) = plastic ? t2{-N} : t2{};
    base::partial(m_r_b, m_strain) = t4{-(2 * G * c * dl) * A};
    base::partial(m_r_b, m_dl) = t2{m_gamma * beta - c * N};
    base::partial(m_r_b, m_db) = t4{(1 + m_gamma * dl) * nm::sym_identity<T, 3>() + (c * dl) * A};
    base::partial(m_eps_p, m_strain) = t4{(2 * G * dl) * A};
    base::partial(m_eps_p, m_dl) = N;
    base::partial(m_eps_p, m_db) = t4{-dl * A};
  }

private:
  T &m_r_l;
  t2 &m_r_b;
  nm::history_property<t2> &m_eps_p;
  nm::history_property<T> &m_kappa;
  nm::history_property<t2> &m_beta;
  T const &m_gamma;
  nm::input_property<t2, nm::property_traits> const &m_strain;
  nm::input_property<T, nm::property_traits> const &m_dl;
  nm::input_property<t2, nm::property_traits> const &m_db;
};

/// sigma = C_e : (eps - flow::plastic_strain).
class elastic_stress final : public nm::material_base<elastic_stress, policy> {
public:
  using base = nm::material_base<elastic_stress, policy>;

  template <typename... Args>
  explicit elastic_stress(Args &&...args)
      : base(std::forward<Args>(args)...), m_stress(base::add_output<t2>("stress")), m_strain(base::add_input<t2>("stepper", "strain")),
        m_eps_p(base::add_input<t2>("flow", "plastic_strain")) {
    base::add_callback(&elastic_stress::compute, m_stress).reads(m_strain, m_eps_p).partials(m_stress, m_strain, m_eps_p);
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() {
    m_stress = tmech::dcontract(m_C, t2{m_strain.get() - m_eps_p.get()});
    if (base::want_partials()) {
      base::partial(m_stress, m_strain) = m_C;
      base::partial(m_stress, m_eps_p) = t4{-m_C};
    }
  }

private:
  t2 &m_stress;
  nm::input_property<t2, nm::property_traits> const &m_strain;
  nm::input_property<t2, nm::property_traits> const &m_eps_p;
  t4 const m_C{nm::plasticity_detail::make_isotropic_tangent<T, 3>(K, G)};
};

/// The shipped material on vector_newton, as test_j2_kinematic_plasticity builds it.
struct shipped {
  ctx_type ctx;
  nm::external_strain_source<policy> *src{};

  explicit shipped(T gamma) {
    param_type p;
    p.insert<std::string>("name", "stepper");
    src = &ctx.create<nm::external_strain_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "solver");
    p.insert<std::string>("function", "kin");
    p.insert<T>("tolerance", T{1e-12});
    p.insert<bool>("strain_derivative", true);
    p.insert<bool>("graph_driven", false);
    p.insert<std::vector<nm::unknown_spec>>("unknowns", {{"dlambda", nm::unknown_kind::scalar}, {"dbeta", nm::unknown_kind::sym_tensor}});
    ctx.create<nm::vector_newton<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "hardening");
    p.insert<std::string>("source", "kin");
    p.insert<T>("K", H_iso);
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

  void step(t2 const &eps) {
    src->bind(eps, eps);
    ctx.update();
  }
  t2 const &stress() { return ctx.get<t2>("kin", "stress"); }
  t4 const &tangent() { return ctx.get<t4>("kin", "tangent"); }
  t2 const &back_stress() { return ctx.get<t2>("kin", "back_stress"); }
};

/// The same model as a system over a flow node and a stress node.
struct as_system {
  ctx_type ctx;
  nm::external_strain_source<policy> *src{};
  nm::residual_system<policy> *system{};
  t4 const *tangent{};

  explicit as_system(T gamma) {
    param_type p;
    p.insert<std::string>("name", "stepper");
    src = &ctx.create<nm::external_strain_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "plastic");
    p.insert<T>("tolerance", T{1e-12});
    p.insert<std::vector<nm::system_unknown>>("unknowns", {{"dlambda", nm::unknown_kind::scalar}, {"dbeta", nm::unknown_kind::sym_tensor}});
    p.insert<std::vector<nm::system_equation>>("equations", {{"flow::r_l", "dlambda"}, {"flow::r_b", "dbeta"}});
    system = &ctx.create<nm::residual_system<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "flow");
    p.insert<T>("gamma", gamma);
    ctx.create<kinematic_flow>(p);
    p.clear();
    p.insert<std::string>("name", "kin");
    ctx.create<elastic_stress>(p);
    ctx.finalize();
    tangent = &ctx.derivative<t4>("kin", "stress", "stepper", "strain");
  }

  void step(t2 const &eps) {
    src->bind(eps, eps);
    ctx.update();
  }
  t2 const &stress() { return ctx.get<t2>("kin", "stress"); }
  t2 const &back_stress() { return ctx.get<t2>("flow", "back_stress"); }
};

t2 strain_a() {
  t2 e;
  e.fill(T{0});
  e(0, 0) = 0.010;
  e(1, 1) = -0.003;
  e(0, 1) = e(1, 0) = 0.004;
  return e;
}
t2 strain_b() {
  t2 e;
  e.fill(T{0});
  e(0, 0) = -0.002;
  e(2, 2) = 0.006;
  e(0, 1) = e(1, 0) = 0.012;
  return e;
}

template <typename A>
T relative(A const &a, A const &b) {
  return tmech::norm(A{a - b}) / tmech::norm(b);
}

TEST(SystemPlasticity, MatchesTheShippedStressAndTangent) {
  for (T const gamma : {0.0, 40.0}) {
    shipped ref{gamma};
    as_system sys{gamma};
    for (auto const &eps : {t2{0.1 * strain_a()}, strain_a(), strain_b(), t2{0.9 * strain_b()}}) {
      ref.step(eps);
      sys.step(eps);
      EXPECT_LT(relative(sys.stress(), ref.stress()), 1e-12) << "gamma " << gamma;
      EXPECT_LT(relative(*sys.tangent, ref.tangent()), 1e-12) << "gamma " << gamma;
      if (tmech::norm(ref.back_stress()) > 0) {
        EXPECT_LT(relative(sys.back_stress(), ref.back_stress()), 1e-12);
      }
      ref.ctx.commit();
      sys.ctx.commit();
    }
    EXPECT_GT(tmech::norm(sys.back_stress()), 0) << "the path yields";
  }
}

TEST(SystemPlasticity, OnlyRecoveryNeedsMoreThanOneNewtonStep) {
  for (T const gamma : {0.0, 40.0}) {
    as_system sys{gamma};
    sys.step(strain_a());
    EXPECT_EQ(sys.system->iterations() == 1, gamma == 0) << "Prager is a radial return: one step from the trial state";
  }
}

TEST(SystemPlasticity, NoZeroBlocksAndThePartialsMatchFiniteDifferences) {
  as_system sys{40.0};
  EXPECT_TRUE(sys.system->zero_blocks().empty());
  sys.step(strain_a());
  sys.ctx.commit();
  sys.step(strain_b());
  for (auto const *name : {"flow", "kin"})
    for (auto const &mismatch : nm::check_partials(sys.ctx, name))
      ADD_FAILURE() << mismatch;
}

TEST(SystemPlasticity, UpdatePropertySolvesTheSystemLikeUpdate) {
  for (T const gamma : {0.0, 40.0}) {
    as_system full{gamma}, single{gamma};
    full.step(strain_a());
    single.src->bind(strain_a(), strain_a());
    single.ctx.update_property("kin", "stress");
    EXPECT_GT(tmech::norm(full.back_stress()), 0) << "the path yields";
    EXPECT_LT(relative(single.stress(), full.stress()), 1e-14) << "gamma " << gamma;
    EXPECT_LT(relative(single.back_stress(), full.back_stress()), 1e-14) << "gamma " << gamma;
  }
}

TEST(SystemPlasticity, TheTangentMatchesCentralDifferencesOfTheStress) {
  for (T const gamma : {0.0, 40.0}) {
    as_system sys{gamma};
    sys.step(t2{0.1 * strain_a()});
    sys.ctx.commit();
    sys.step(strain_a());
    ASSERT_GT(tmech::norm(sys.back_stress()), 0) << "the step yields";
    t4 const tangent{*sys.tangent};
    constexpr T h{1e-6};
    for (int i{0}; i < 3; ++i)
      for (int j{i}; j < 3; ++j) {
        t2 E;
        E.fill(T{0});
        E(i, j) = E(j, i) = i == j ? T{1} : T{0.5};
        auto const stress_at{[&](T step) {
          sys.ctx.revert();
          sys.step(t2{strain_a() + step * E});
          return t2{sys.stress()};
        }};
        t2 const plus{stress_at(h)}, minus{stress_at(-h)};
        t2 const fd{(T{1} / (2 * h)) * (plus - minus)};
        t2 const analytic{tmech::dcontract(tangent, E)};
        EXPECT_LT(relative(fd, analytic), 1e-5) << "gamma " << gamma << ", component " << i << j;
      }
  }
}

} // namespace
