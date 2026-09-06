#include <gtest/gtest.h>
#include <cmath>
#include <print>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/history_property.h"
#include "numsim-materials/solvers/butcher_tableau.h"
#include "numsim-materials/solvers/rk_integrator.h"
#include "numsim-materials/materials/scalar_stepper.h"
#include "numsim-materials/materials/curing_rate.h"

namespace {

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param_type = policy::ParameterHandler;

/// Simple exponential decay rate function: dy/dt = -lambda * y
/// Also provides df/dy = -lambda (for implicit methods).
template<typename Traits>
class exponential_decay final
    : public numsim::materials::material_base<exponential_decay<Traits>, Traits> {
public:
  using base = numsim::materials::material_base<exponential_decay<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template<typename... Args>
  explicit exponential_decay(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_rate(base::template add_output<value_type>(
            "rate", &exponential_decay::compute)),
        m_drate(base::template add_output<value_type>("rate_derivative")),
        m_lambda(base::template get_parameter<value_type>("lambda")),
        m_source(base::template get_parameter<std::string>("source")),
        m_y(base::template add_input<value_type>(
            m_source, "state", numsim::materials::EdgeKind::Local))
  {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("source").template add<numsim_core::is_required>();
    para.template insert<value_type>("lambda").template add<numsim_core::is_required>();
    return para;
  }

  void compute() {
    m_rate = -m_lambda * m_y.get();
    m_drate = -m_lambda;
  }

private:
  value_type& m_rate;
  value_type& m_drate;
  const value_type& m_lambda;
  const std::string& m_source;
  const numsim::materials::input_property<value_type, numsim::materials::property_traits>& m_y;
};

/// Run exponential decay with a given integrator type and tableau.
/// Returns y at t=1.0 with N steps of size h=1/N.
template<typename Integrator>
T run_decay(int N, const std::string& tableau, T lambda = 1.0) {
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "integrator");
  p.insert<std::string>("function", "decay");
  p.insert<T>("step_size", T{1.0} / T(N));
  p.insert<std::string>("tableau", tableau);
  auto& integ = ctx.create<Integrator>(p);

  p.clear();
  p.insert<std::string>("name", "decay");
  p.insert<std::string>("source", "integrator");
  p.insert<T>("lambda", lambda);
  ctx.create<exponential_decay<policy>>(p);

  ctx.finalize();

  // Set initial condition y(0) = 1 (both old and new)
  auto* prop = ctx.find_property("integrator", "state");
  auto* hist = dynamic_cast<numsim_core::history_property<T, numsim::materials::property_traits>*>(prop);
  hist->old_value() = T{1};
  hist->new_value() = T{1};

  for (int i = 0; i < N; ++i) {
    ctx.update();
    ctx.commit();
  }

  return ctx.get<T>("integrator", "state");
}

const T exact = std::exp(-1.0);  // y(1) = e^(-1) ≈ 0.367879...

// --- Explicit RK tests ---

using RK = numsim::materials::rk_integrator<policy>;

TEST(ExplicitRK, ForwardEulerConverges) {
  const std::string tab = "forward_euler";
  auto y = run_decay<RK>(100, tab);
  EXPECT_NEAR(y, exact, 0.01) << "Forward Euler with 100 steps should be close";
}

TEST(ExplicitRK, ForwardEulerOrder1) {
  const std::string tab = "forward_euler";
  auto err_10 = std::abs(run_decay<RK>(10, tab) - exact);
  auto err_20 = std::abs(run_decay<RK>(20, tab) - exact);
  auto ratio = err_10 / err_20;
  std::println("  Forward Euler: err_10={:.6e} err_20={:.6e} ratio={:.2f} (expect ~2)",
               err_10, err_20, ratio);
  EXPECT_NEAR(ratio, 2.0, 0.3) << "Order 1: halving h should halve error";
}

TEST(ExplicitRK, RK4Order4) {
  const std::string tab = "rk4";
  auto err_10 = std::abs(run_decay<RK>(10, tab) - exact);
  auto err_20 = std::abs(run_decay<RK>(20, tab) - exact);
  auto ratio = err_10 / err_20;
  std::println("  RK4: err_10={:.6e} err_20={:.6e} ratio={:.2f} (expect ~16)",
               err_10, err_20, ratio);
  EXPECT_NEAR(ratio, 16.0, 2.0) << "Order 4: halving h should reduce error by 16x";
}

TEST(ExplicitRK, RK4HighAccuracy) {
  const std::string tab = "rk4";
  auto y = run_decay<RK>(100, tab);
  EXPECT_NEAR(y, exact, 1e-10) << "RK4 with 100 steps should be very accurate";
}

// --- DIRK tests ---


TEST(DIRK, ImplicitEulerConverges) {
  const std::string tab = "implicit_euler";
  auto y = run_decay<RK>(100, tab);
  EXPECT_NEAR(y, exact, 0.01) << "Implicit Euler with 100 steps";
}

TEST(DIRK, ImplicitMidpointOrder2) {
  const std::string tab = "implicit_midpoint";
  auto err_10 = std::abs(run_decay<RK>(10, tab) - exact);
  auto err_20 = std::abs(run_decay<RK>(20, tab) - exact);
  auto ratio = err_10 / err_20;
  std::println("  Implicit midpoint: err_10={:.6e} err_20={:.6e} ratio={:.2f} (expect ~4)",
               err_10, err_20, ratio);
  EXPECT_NEAR(ratio, 4.0, 1.0) << "Order 2: halving h should reduce error by 4x";
}

TEST(DIRK, CrankNicolsonOrder2) {
  const std::string tab = "crank_nicolson";
  auto err_10 = std::abs(run_decay<RK>(10, tab) - exact);
  auto err_20 = std::abs(run_decay<RK>(20, tab) - exact);
  auto ratio = err_10 / err_20;
  std::println("  Crank-Nicolson: err_10={:.6e} err_20={:.6e} ratio={:.2f} (expect ~4)",
               err_10, err_20, ratio);
  EXPECT_NEAR(ratio, 4.0, 1.0) << "Order 2";
}

// --- Fully implicit RK tests ---


TEST(ImplicitRK, GaussLegendreOrder4) {
  const std::string tab = "gauss_legendre_4";
  auto err_10 = std::abs(run_decay<RK>(10, tab) - exact);
  auto err_20 = std::abs(run_decay<RK>(20, tab) - exact);
  auto ratio = err_10 / err_20;
  std::println("  Gauss-Legendre: err_10={:.6e} err_20={:.6e} ratio={:.2f} (expect ~16)",
               err_10, err_20, ratio);
  EXPECT_NEAR(ratio, 16.0, 3.0) << "Order 4: 2-stage Gauss-Legendre";
}

TEST(ImplicitRK, GaussLegendreHighAccuracy) {
  const std::string tab = "gauss_legendre_4";
  auto y = run_decay<RK>(50, tab);
  EXPECT_NEAR(y, exact, 1e-10) << "Gauss-Legendre with 50 steps";
}

// --- Curing simulation with RK integrators ---
// Compare RK4 (explicit) and implicit midpoint against backward_euler reference.
// All should converge to z ≈ 1 after enough steps at 80°C.

template<typename Integrator>
T run_curing(int N, const std::string& tableau, T step_size = T{10}) {
  ctx_type ctx;
  param_type p;

  // Temperature (constant 80°C)
  p.clear();
  p.insert<std::string>("name", "temperature");
  p.insert<T>("increment", T{0});
  ctx.create<numsim::materials::scalar_stepper<policy>>(p);

  // RK integrator owns the curing state
  p.clear();
  p.insert<std::string>("name", "integrator");
  p.insert<std::string>("function", "curing_rate");
  p.insert<T>("step_size", step_size);
  p.insert<std::string>("tableau", tableau);
  ctx.create<Integrator>(p);

  // Curing rate function — reads state from integrator
  p.clear();
  p.insert<std::string>("name", "curing_rate");
  p.insert<std::string>("integrator_source", "integrator");
  p.insert<T>("A", T{1e6});
  p.insert<T>("E", T{50000});
  p.insert<T>("n", T{1.2});
  p.insert<T>("m", T{0.8});
  ctx.create<numsim::materials::curing_rate<policy>>(p);

  ctx.finalize();

  // Initial conditions
  auto* temp_prop = ctx.find_property("temperature", "state");
  auto* temp_hist = dynamic_cast<numsim_core::history_property<T,
      numsim::materials::property_traits>*>(temp_prop);
  temp_hist->old_value() = T{80};
  temp_hist->new_value() = T{80};

  auto* state_prop = ctx.find_property("integrator", "state");
  auto* state_hist = dynamic_cast<numsim_core::history_property<T,
      numsim::materials::property_traits>*>(state_prop);
  state_hist->old_value() = T{1e-8};
  state_hist->new_value() = T{1e-8};

  for (int i = 0; i < N; ++i) {
    ctx.update();
    ctx.commit();
  }

  return ctx.get<T>("integrator", "state");
}

/// A converged reference from RK4 at a hundredth of the coarsest step under
/// test, sampled at t = 200 s.
///
/// The sampling time is the point of these tests. They used to run to t = 500 s
/// and assert only z > 0.90, which the autocatalytic ODE makes very nearly
/// free: the reaction is self-accelerating past its ignition threshold and then
/// self-terminating at z = 1, so by 500 s every trajectory has been pulled to
/// the same attractor. Halving the whole reaction rate still finished at 0.97
/// and passed all three. Tightening the bound does not help either -- a
/// reference measured at 500 s is just as insensitive, and every scheme lands
/// within 4e-5 of it whatever it does on the way.
///
/// At t = 200 s the cure is mid-ignition and still transient, where the
/// trajectories actually differ: full rate reaches z = 0.90, half rate only
/// 0.28. That is the observable worth bounding.
constexpr int reference_time = 200;

T converged_cure() {
  static const T reference =
      run_curing<RK>(reference_time * 10, "rk4", T{0.1});
  return reference;
}

TEST(CuringRK, ExplicitRK4MatchesTheRefinedSolution) {
  const auto z = run_curing<RK>(reference_time / 10, "rk4", T{10});
  std::println("  RK4 (t=200, h=10): z = {:.8f}  ref = {:.8f}",
               z, converged_cure());
  EXPECT_NEAR(z, converged_cure(), 1.5e-2);   // measured 6.7e-3
}

TEST(CuringRK, DIRKImplicitMidpointMatchesTheRefinedSolution) {
  // Smaller step for implicit — stiff initial phase needs h < 1/df_dy
  const auto z = run_curing<RK>(reference_time, "implicit_midpoint", T{1});
  std::println("  Implicit midpoint (t=200, h=1): z = {:.8f}  ref = {:.8f}",
               z, converged_cure());
  EXPECT_NEAR(z, converged_cure(), 2e-3);     // measured 8.5e-4
}

TEST(CuringRK, FullyImplicitGaussLegendreMatchesTheRefinedSolution) {
  const auto z = run_curing<RK>(reference_time, "gauss_legendre_4", T{1});
  std::println("  Gauss-Legendre (t=200, h=1): z = {:.8f}  ref = {:.8f}",
               z, converged_cure());
  EXPECT_NEAR(z, converged_cure(), 5e-5);     // measured 9.1e-6
}

/// Reaching full cure is a separate property from integrating accurately, and
/// it is the one the old thresholds were actually testing. Kept, under a name
/// that says so.
TEST(CuringRK, ReachesFullCureByFiveHundredSeconds) {
  EXPECT_GT(run_curing<RK>(50, "rk4", T{10}), 0.99);
}


/// The rate law itself, which no integrator test can pin.
///
/// CuringRK.* compare a scheme against a refined solution of the SAME ODE, so
/// they are correctly blind to the ODE being wrong: halve the reaction rate and
/// the reference halves with it. That separation is right -- those tests own
/// integrator accuracy -- but it left k(T) z^m (1-z)^n itself unchecked by
/// anything, and the old z > 0.90 bound could not see a 2x rate error either.
///
/// The state source here is a scalar_stepper rather than an integrator, so z is
/// prescribed instead of being solved for.
TEST(CuringRate, MatchesTheArrheniusAutocatalyticLaw) {
  constexpr T A{1e6}, E{50000}, n{1.2}, m{0.8}, R{8.31446261815324}, celsius{80};
  constexpr T dz{0.05};

  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "temperature");
  p.insert<T>("increment", T{0});
  ctx.create<numsim::materials::scalar_stepper<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "state_source");
  p.insert<T>("increment", dz);
  ctx.create<numsim::materials::scalar_stepper<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "curing_rate");
  p.insert<std::string>("integrator_source", "state_source");
  p.insert<T>("A", A);
  p.insert<T>("E", E);
  p.insert<T>("n", n);
  p.insert<T>("m", m);
  ctx.create<numsim::materials::curing_rate<policy>>(p);
  ctx.finalize();

  auto* temp = dynamic_cast<numsim_core::history_property<T,
      numsim::materials::property_traits>*>(
          ctx.find_property("temperature", "state"));
  ASSERT_NE(temp, nullptr);
  temp->old_value() = celsius;
  temp->new_value() = celsius;

  const T k = A * std::exp(-E / (R * (T{273.15} + celsius)));
  for (int i = 1; i <= 15; ++i) {
    ctx.update();
    const T z = dz * i;
    const T expected = k * std::pow(z, m) * std::pow(T{1} - z, n);
    EXPECT_NEAR(ctx.get<T>("curing_rate", "rate"), expected, 1e-12 * expected)
        << "rate law disagrees at z = " << z;
    ctx.commit();
  }
}

} // namespace

namespace {
namespace nm_tb = numsim::materials;

/// The scheme must be selectable by NAME, because that is the explicit-vs-
/// implicit choice and it belongs in the deck rather than in a recompile.
TEST(TableauByName, ResolvesEveryPublishedScheme) {
  for (const char* n : {"forward_euler", "explicit_midpoint", "rk4",
                        "implicit_euler", "implicit_midpoint",
                        "crank_nicolson", "sdirk3", "gauss_legendre_4"}) {
    const auto t = nm_tb::tableau_by_name(n);
    EXPECT_GT(t.stages(), 0) << n;
  }
  // and the explicit/implicit split really is what the name selects
  EXPECT_TRUE(nm_tb::tableau_by_name("forward_euler").is_explicit());
  EXPECT_TRUE(nm_tb::tableau_by_name("rk4").is_explicit());
  EXPECT_FALSE(nm_tb::tableau_by_name("sdirk3").is_explicit());
  EXPECT_FALSE(nm_tb::tableau_by_name("gauss_legendre_4").is_explicit());
}

/// A typo must name itself and list the alternatives, not fall back to a
/// default scheme -- silently integrating with the wrong method would change
/// results without changing anything visible.
TEST(TableauByName, RejectsAnUnknownSchemeAndListsTheValidOnes) {
  try {
    nm_tb::tableau_by_name("sdirk4");
    FAIL() << "an unknown scheme must throw";
  } catch (const std::invalid_argument& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("sdirk4"), std::string::npos) << msg;
    EXPECT_NE(msg.find("sdirk3"), std::string::npos)
        << "the message should list what IS valid: " << msg;
  }
}
}  // namespace
