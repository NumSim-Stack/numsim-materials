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
T run_decay(int N, const numsim::materials::butcher_tableau& tab, T lambda = 1.0) {
  ctx_type ctx;
  param_type p;

  p.clear();
  p.insert<std::string>("name", "integrator");
  p.insert<std::string>("function", "decay");
  p.insert<T>("step_size", T{1.0} / T(N));
  p.insert<const numsim::materials::butcher_tableau*>("tableau", &tab);
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
  auto tab = numsim::materials::forward_euler();
  auto y = run_decay<RK>(100, tab);
  EXPECT_NEAR(y, exact, 0.01) << "Forward Euler with 100 steps should be close";
}

TEST(ExplicitRK, ForwardEulerOrder1) {
  auto tab = numsim::materials::forward_euler();
  auto err_10 = std::abs(run_decay<RK>(10, tab) - exact);
  auto err_20 = std::abs(run_decay<RK>(20, tab) - exact);
  auto ratio = err_10 / err_20;
  std::println("  Forward Euler: err_10={:.6e} err_20={:.6e} ratio={:.2f} (expect ~2)",
               err_10, err_20, ratio);
  EXPECT_NEAR(ratio, 2.0, 0.3) << "Order 1: halving h should halve error";
}

TEST(ExplicitRK, RK4Order4) {
  auto tab = numsim::materials::rk4();
  auto err_10 = std::abs(run_decay<RK>(10, tab) - exact);
  auto err_20 = std::abs(run_decay<RK>(20, tab) - exact);
  auto ratio = err_10 / err_20;
  std::println("  RK4: err_10={:.6e} err_20={:.6e} ratio={:.2f} (expect ~16)",
               err_10, err_20, ratio);
  EXPECT_NEAR(ratio, 16.0, 2.0) << "Order 4: halving h should reduce error by 16x";
}

TEST(ExplicitRK, RK4HighAccuracy) {
  auto tab = numsim::materials::rk4();
  auto y = run_decay<RK>(100, tab);
  EXPECT_NEAR(y, exact, 1e-10) << "RK4 with 100 steps should be very accurate";
}

// --- DIRK tests ---


TEST(DIRK, ImplicitEulerConverges) {
  auto tab = numsim::materials::implicit_euler();
  auto y = run_decay<RK>(100, tab);
  EXPECT_NEAR(y, exact, 0.01) << "Implicit Euler with 100 steps";
}

TEST(DIRK, ImplicitMidpointOrder2) {
  auto tab = numsim::materials::implicit_midpoint();
  auto err_10 = std::abs(run_decay<RK>(10, tab) - exact);
  auto err_20 = std::abs(run_decay<RK>(20, tab) - exact);
  auto ratio = err_10 / err_20;
  std::println("  Implicit midpoint: err_10={:.6e} err_20={:.6e} ratio={:.2f} (expect ~4)",
               err_10, err_20, ratio);
  EXPECT_NEAR(ratio, 4.0, 1.0) << "Order 2: halving h should reduce error by 4x";
}

TEST(DIRK, CrankNicolsonOrder2) {
  auto tab = numsim::materials::crank_nicolson();
  auto err_10 = std::abs(run_decay<RK>(10, tab) - exact);
  auto err_20 = std::abs(run_decay<RK>(20, tab) - exact);
  auto ratio = err_10 / err_20;
  std::println("  Crank-Nicolson: err_10={:.6e} err_20={:.6e} ratio={:.2f} (expect ~4)",
               err_10, err_20, ratio);
  EXPECT_NEAR(ratio, 4.0, 1.0) << "Order 2";
}

// --- Fully implicit RK tests ---


TEST(ImplicitRK, GaussLegendreOrder4) {
  auto tab = numsim::materials::gauss_legendre_4();
  auto err_10 = std::abs(run_decay<RK>(10, tab) - exact);
  auto err_20 = std::abs(run_decay<RK>(20, tab) - exact);
  auto ratio = err_10 / err_20;
  std::println("  Gauss-Legendre: err_10={:.6e} err_20={:.6e} ratio={:.2f} (expect ~16)",
               err_10, err_20, ratio);
  EXPECT_NEAR(ratio, 16.0, 3.0) << "Order 4: 2-stage Gauss-Legendre";
}

TEST(ImplicitRK, GaussLegendreHighAccuracy) {
  auto tab = numsim::materials::gauss_legendre_4();
  auto y = run_decay<RK>(50, tab);
  EXPECT_NEAR(y, exact, 1e-10) << "Gauss-Legendre with 50 steps";
}

// --- Curing simulation with RK integrators ---
// Compare RK4 (explicit) and implicit midpoint against backward_euler reference.
// All should converge to z ≈ 1 after enough steps at 80°C.

template<typename Integrator>
T run_curing(int N, const numsim::materials::butcher_tableau& tab, T step_size = T{10}) {
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
  p.insert<const numsim::materials::butcher_tableau*>("tableau", &tab);
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

TEST(CuringRK, ExplicitRK4ConvergesToFullCure) {
  auto tab = numsim::materials::rk4();
  auto z = run_curing<RK>(50, tab);
  std::println("  RK4 curing (500s): z = {:.6f}", z);
  EXPECT_GT(z, 0.90) << "RK4 should approach full cure";
}

TEST(CuringRK, DIRKImplicitMidpointConverges) {
  auto tab = numsim::materials::implicit_midpoint();
  // Smaller step for implicit — stiff initial phase needs h < 1/df_dy
  auto z = run_curing<RK>(500, tab, T{1});  // h=1, 500 steps
  std::println("  Implicit midpoint curing (500s, h=1): z = {:.6f}", z);
  EXPECT_GT(z, 0.90) << "Implicit midpoint should approach full cure";
}

TEST(CuringRK, FullyImplicitGaussLegendreConverges) {
  auto tab = numsim::materials::gauss_legendre_4();
  auto z = run_curing<RK>(500, tab, T{1});  // h=1, 500 steps
  std::println("  Gauss-Legendre curing (500s, h=1): z = {:.6f}", z);
  EXPECT_GT(z, 0.90) << "Gauss-Legendre should approach full cure";
}

} // namespace
