#include <gtest/gtest.h>
#include <optional>
#include <sstream>
#include <string>
#include <tmech/tmech.h>
#include <vector>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/driver/material_point.h"
#include "numsim-materials/materials/j2_plasticity.h"
#include "numsim-materials/solvers/local_newton.h"
#include "numsim-materials/driver/external_state_source.h"

// The proving case of the engine / integrator / driver split (#184, phase 1): temperature-dependent J2
// plasticity driven step by step, against the contract a material library owes its host:
// - a trial evaluation is repeatable and computes from the accepted state only;
// - unloading is elastic;
// - a rejected step leaves the accepted state intact;
// - a restart continues as the uninterrupted run;
// - the tangent is the derivative of the stress;
// - neither the creation order nor folding constants changes a result.

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

/// Linear hardening that softens with temperature: H = K kappa - s (T - T_ref), dH/dkappa = K. The yield stress
/// sigma_0 + H drops as the temperature rises; j2_plasticity drives it (local edges), so the temperature reaches
/// the return map only through drives().
template <typename Traits>
class thermal_hardening final : public nm::material_base<thermal_hardening<Traits>, Traits> {
public:
  using base = nm::material_base<thermal_hardening<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit thermal_hardening(Args &&...args)
      : base(std::forward<Args>(args)...), m_H(base::template add_output<value_type>("hardening_stress")),
        m_dH(base::template add_output<value_type>("hardening_modulus")),
        m_K(base::template get_parameter<value_type>("K")), m_s(base::template get_parameter<value_type>("softening")),
        m_T_ref(base::template get_parameter<value_type>("reference_temperature")),
        m_kappa(base::template add_input<value_type>("j2", "equivalent_plastic_strain", nm::EdgeKind::Local)),
        m_T(base::template add_input<value_type>("temperature", "state")) {
    base::add_callback(&thermal_hardening::compute, m_H, m_dH).reads(m_kappa, m_T);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<value_type>(para, "K");
    nm::required<value_type>(para, "softening");
    nm::required<value_type>(para, "reference_temperature");
    return para;
  }

  void compute() {
    m_H = m_K * m_kappa.get() - m_s * (m_T.get() - m_T_ref);
    m_dH = m_K;
  }

private:
  value_type &m_H, &m_dH;
  value_type const &m_K, &m_s, &m_T_ref;
  nm::input_property<value_type, nm::property_traits> const &m_kappa, &m_T;
};

constexpr T K{166.67e3}, G{76.92e3}, sigma_0{250.0}, H{2000.0}, softening{0.5}, T_ref{20.0};

/// The model and a driver over it: bind a strain and a temperature, evaluate, accept or reject.
struct model {
  ctx_type ctx;
  std::optional<nm::material_point<policy>> point; ///< engine-owned history
  tensor2 eps_n;     ///< accepted strain
  T T_n{T_ref};      ///< accepted temperature

  static std::vector<std::string> const &created() {
    static std::vector<std::string> const order{"strain", "temperature", "solver", "hardening", "j2"};
    return order;
  }

  explicit model(std::vector<std::string> const &order = created(), bool fold = true) {
    for (auto const &name : order) {
      param_type p;
      p.insert<std::string>("name", name);
      if (name == "strain") {
        ctx.create<nm::external_strain_source<policy>>(p);
      } else if (name == "temperature") {
        ctx.create<nm::external_scalar_source<policy>>(p);
      } else if (name == "solver") {
        ctx.create<nm::local_newton<policy>>(p);
      } else if (name == "hardening") {
        p.insert<T>("K", H);
        p.insert<T>("softening", softening);
        p.insert<T>("reference_temperature", T_ref);
        ctx.create<thermal_hardening<policy>>(p);
      } else {
        p.insert<std::string>("hardening_source", "hardening");
        p.insert<std::string>("strain_source", "strain");
        p.insert<std::string>("solver_source", "solver");
        p.insert<T>("K", K);
        p.insert<T>("G", G);
        p.insert<T>("sigma_0", sigma_0);
        ctx.create<nm::j2_plasticity<policy>>(p);
      }
    }
    ctx.finalize();
    ctx.fold_constants(fold);
    param_type inputs;
    inputs.insert<std::vector<std::string>>("inputs", {"strain", "temperature"});
    point.emplace(ctx, inputs);
  }

  /// A trial evaluation from the accepted state: strain eps and temperature temp at the end of the step.
  tensor2 const &evaluate(tensor2 const &eps, T temp) {
    point->reset();
    point->input<nm::external_strain_source<policy>>("strain").bind(eps_n, eps);
    point->input<nm::external_scalar_source<policy>>("temperature").bind(T_n, temp);
    point->evaluate();
    return stress();
  }
  void accept(tensor2 const &eps, T temp) {
    point->accept();
    eps_n = eps;
    T_n = temp;
  }
  void reject() { point->reset(); }
  /// evaluate() and accept().
  void step(tensor2 const &eps, T temp) {
    evaluate(eps, temp);
    accept(eps, temp);
  }

  tensor2 const &stress() const { return point->output<tensor2>("j2", "stress"); }
  tensor4 const &tangent() const { return point->output<tensor4>("j2", "tangent"); }
  tensor2 const &plastic_strain() const { return point->output<tensor2>("j2", "plastic_strain"); }
  T kappa() const { return point->output<T>("j2", "equivalent_plastic_strain"); }
};

/// Uniaxial-like strain path, a shear part to make the flow direction rotate.
tensor2 strain_at(T e) {
  tensor2 eps;
  eps(0, 0) = e;
  eps(1, 1) = eps(2, 2) = -0.3 * e;
  eps(0, 1) = eps(1, 0) = 0.4 * e;
  return eps;
}

T rel_diff(tensor2 const &a, tensor2 const &b) { return tmech::norm(tensor2{a - b}) / (1 + tmech::norm(a)); }
T rel_diff(tensor4 const &a, tensor4 const &b) { return tmech::norm(tensor4{a - b}) / (1 + tmech::norm(a)); }

constexpr T yield_strain{sigma_0 / (2 * G)}; // order of magnitude of the first yield

TEST(ProvingCase, TrialEvaluationsAreRepeatable) {
  model m;
  m.step(strain_at(2 * yield_strain), 80.0); // plastic, accepted
  auto const kappa_n{m.kappa()};
  ASSERT_GT(kappa_n, 0) << "the accepted step should be plastic";

  auto const trial_a{strain_at(3 * yield_strain)}, trial_b{strain_at(5 * yield_strain)};
  tensor2 const sigma_a{m.evaluate(trial_a, 90.0)};
  tensor4 const C_a{m.tangent()};
  for (int i{0}; i < 5; ++i) { // as a host's Newton iterations would
    m.evaluate(trial_b, 120.0);
    EXPECT_LT(rel_diff(m.evaluate(trial_a, 90.0), sigma_a), 1e-14) << "iteration " << i;
    EXPECT_LT(rel_diff(m.tangent(), C_a), 1e-14) << "iteration " << i;
  }
  m.reject();
  EXPECT_EQ(m.kappa(), kappa_n) << "trials must not advance the accepted history";
}

TEST(ProvingCase, UnloadingIsElastic) {
  model m;
  auto const eps_loaded{strain_at(4 * yield_strain)};
  m.step(eps_loaded, T_ref);
  tensor2 const sigma_loaded{m.stress()}, eps_p{m.plastic_strain()};
  ASSERT_GT(m.kappa(), 0);

  auto const eps_unloaded{strain_at(3.5 * yield_strain)};
  m.step(eps_unloaded, T_ref);
  auto const C_e{nm::plasticity_detail::make_isotropic_tangent<T, 3>(K, G)};
  EXPECT_LT(rel_diff(m.stress(), tensor2{sigma_loaded + tmech::dcontract(C_e, tensor2{eps_unloaded - eps_loaded})}), 1e-12);
  EXPECT_LT(rel_diff(m.plastic_strain(), eps_p), 1e-14);
  EXPECT_LT(rel_diff(m.tangent(), C_e), 1e-12);
}

TEST(ProvingCase, ARejectedStepLeavesTheAcceptedStateIntact) {
  model reference, rejecting;
  for (auto *m : {&reference, &rejecting})
    m->step(strain_at(2 * yield_strain), 60.0);

  rejecting.evaluate(strain_at(40 * yield_strain), 300.0); // far too large: the host cuts the step
  rejecting.reject();

  for (auto *m : {&reference, &rejecting})
    m->step(strain_at(3 * yield_strain), 70.0);
  EXPECT_LT(rel_diff(rejecting.stress(), reference.stress()), 1e-14);
  EXPECT_LT(rel_diff(rejecting.plastic_strain(), reference.plastic_strain()), 1e-14);
  EXPECT_EQ(rejecting.kappa(), reference.kappa());
}

TEST(ProvingCase, ARestartContinuesAsTheUninterruptedRun) {
  model uninterrupted, before;
  for (auto *m : {&uninterrupted, &before})
    for (int k{1}; k <= 3; ++k)
      m->step(strain_at(k * yield_strain), T_ref + 10 * k);
  std::stringstream checkpoint;
  before.point->save(checkpoint);

  model restarted;
  restarted.point->load(checkpoint);
  restarted.eps_n = before.eps_n; // the host restores its own state
  restarted.T_n = before.T_n;
  for (int k{4}; k <= 6; ++k)
    for (auto *m : {&uninterrupted, &restarted}) {
      m->step(strain_at(k * yield_strain), T_ref + 10 * k);
      if (m == &restarted) {
        EXPECT_LT(rel_diff(restarted.stress(), uninterrupted.stress()), 1e-14) << "step " << k;
        EXPECT_EQ(restarted.kappa(), uninterrupted.kappa()) << "step " << k;
      }
    }
}

TEST(ProvingCase, TheTangentIsTheDerivativeOfTheStress) {
  model m;
  m.step(strain_at(2 * yield_strain), 50.0);
  auto const eps{strain_at(3 * yield_strain)};
  m.evaluate(eps, 50.0);
  tensor4 const C{m.tangent()};

  T const h{1e-7};
  for (auto const &[a, b] : std::vector<std::pair<std::size_t, std::size_t>>{{0, 0}, {1, 1}, {0, 1}, {1, 2}}) {
    tensor2 D; // a symmetric direction
    D(a, b) = D(b, a) = 1;
    tensor2 const plus{m.evaluate(tensor2{eps + h * D}, 50.0)};
    tensor2 const minus{m.evaluate(tensor2{eps - h * D}, 50.0)};
    tensor2 const numerical{(plus - minus) / (2 * h)};
    EXPECT_LT(rel_diff(numerical, tensor2{tmech::dcontract(C, D)}), 1e-5) << "direction (" << a << "," << b << ")";
  }
}

TEST(ProvingCase, AHotterStepYieldsAtALowerStress) {
  model cold, hot;
  cold.step(strain_at(4 * yield_strain), T_ref);
  hot.step(strain_at(4 * yield_strain), T_ref + 200);
  auto const eq{[](tensor2 const &s) {
    tensor2 const dev{s - tmech::trace(s) / 3 * tensor2{tmech::eye<T, 3, 2>()}};
    return std::sqrt(1.5 * tmech::dcontract(dev, dev));
  }};
  EXPECT_NEAR(eq(cold.stress()), sigma_0 + H * cold.kappa(), 1e-6 * sigma_0);
  EXPECT_NEAR(eq(hot.stress()), sigma_0 + H * hot.kappa() - softening * 200, 1e-6 * sigma_0);
  EXPECT_GT(hot.kappa(), cold.kappa());
}

/// Six steps loading, unloading and reloading at a rising temperature.
std::vector<tensor2> history_of(model &m) {
  std::vector<tensor2> stresses;
  for (T const k : {1.0, 2.0, 4.0, 3.0, 3.5, 6.0}) {
    m.step(strain_at(k * yield_strain), T_ref + 15 * k);
    stresses.push_back(m.stress());
  }
  return stresses;
}

void expect_same(std::vector<tensor2> const &a, std::vector<tensor2> const &b) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i{0}; i < a.size(); ++i)
    EXPECT_LT(rel_diff(a[i], b[i]), 1e-14) << "step " << i;
}

TEST(ProvingCase, TheCreationOrderChangesNoResult) {
  model forward, backward{{"j2", "hardening", "solver", "temperature", "strain"}};
  expect_same(history_of(forward), history_of(backward));
}

TEST(ProvingCase, FoldingConstantsChangesNoResult) {
  model folded, unfolded{model::created(), false};
  expect_same(history_of(folded), history_of(unfolded));
}

} // namespace
