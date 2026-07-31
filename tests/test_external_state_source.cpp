#include <gtest/gtest.h>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/autocatalytic_reaction.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/scalar_stepper.h"
#include "numsim-materials/materials/small_strain_plasticity.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/umat/external_state_source.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;

constexpr T K = 166.67;
constexpr T G = 76.92;
constexpr T sigma_0 = 50.0;
constexpr T H_mod = 1000.0;
constexpr T d_eps = 0.05;

/// Everything downstream of the strain producer. Identical in both contexts —
/// only the producer differs, which is the whole point of the comparison.
void add_j2_chain(ctx_type& ctx) {
  param_type p;

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  ctx.create<nm::linear_elasticity<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "solver");
  ctx.create<nm::backward_euler<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "hardening");
  p.insert<std::string>("source", "j2");
  p.insert<T>("K", H_mod);
  ctx.create<nm::linear_isotropic_hardening<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "j2");
  p.insert<std::string>("elastic_source", "elastic");
  p.insert<std::string>("hardening_source", "hardening");
  p.insert<std::string>("strain_source", "stepper");
  p.insert<std::string>("solver_source", "solver");
  p.insert<T>("G", G);
  p.insert<T>("sigma_0", sigma_0);
  ctx.create<nm::j2_plasticity<policy>>(p);
}

tensor2 uniaxial(T value) {
  tensor2 e;
  e.fill(0.0);
  e(0, 0) = value;
  return e;
}

// ---------------------------------------------------------------------------
// Equivalence with the self-driving steppers
// ---------------------------------------------------------------------------

/// Swapping tensor_component_stepper for external_strain_source, driven along
/// the very path the stepper would have generated, must reproduce the
/// trajectory exactly — not approximately. Any difference means the history
/// wiring or the value the consumer binds to has changed.
TEST(ExternalStrainSource, ReproducesStepperTrajectoryExactly) {
  ctx_type ref;
  {
    param_type p;
    p.insert<std::string>("name", "stepper");
    p.insert<T>("increment", d_eps);
    p.insert<std::vector<std::size_t>>("indices", {0, 0});
    ref.create<nm::tensor_component_stepper<2, policy>>(p);
  }
  add_j2_chain(ref);
  ref.finalize();

  ctx_type ext;
  param_type ep;
  ep.insert<std::string>("name", "stepper");
  auto& src = ext.create<nm::external_strain_source<policy>>(ep);
  add_j2_chain(ext);
  ext.finalize();

  // Drive the external source with the reference stepper's OWN strain values
  // rather than recomputing them as step * d_eps. The stepper accumulates
  // (`+= d_eps`), and repeated addition is not bit-identical to a single
  // multiply — feeding a recomputed path would inject a 1-ulp input difference
  // and turn an exact-equality test into a tolerance argument. Binding the
  // identical input means any output difference is a genuine wiring difference.
  tensor2 prev = uniaxial(0.0);

  for (int step = 0; step < 30; ++step) {
    ref.update();

    const tensor2 cur = ref.get<tensor2>("stepper", "strain");
    src.bind(prev, cur);
    ext.update();
    prev = cur;

    const auto& sig_ref = ref.get<tensor2>("j2", "stress");
    const auto& sig_ext = ext.get<tensor2>("j2", "stress");
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        EXPECT_DOUBLE_EQ(sig_ext(i, j), sig_ref(i, j))
            << "step " << step << " component (" << i << "," << j << ")";

    EXPECT_DOUBLE_EQ(ext.get<T>("j2", "equivalent_plastic_strain"),
                     ref.get<T>("j2", "equivalent_plastic_strain"))
        << "step " << step;

    ref.commit();
    ext.commit();
  }
}

/// The reason "strain" must stay a history property: a consumer binding via
/// add_input_history needs old/new. autocatalytic_reaction does exactly that
/// for time, forming dt = new - old, so driving it from an external scalar
/// source must reproduce the scalar_stepper result exactly.
TEST(ExternalScalarSource, ReproducesScalarStepperCuringExactly) {
  constexpr T dt = 10.0;
  constexpr T temp = 353.0;

  auto add_curing_chain = [](ctx_type& ctx) {
    param_type p;
    p.clear();
    p.insert<std::string>("name", "solver");
    p.insert<std::string>("function", "curing");
    ctx.create<nm::backward_euler<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "curing");
    p.insert<T>("A", T{1e6});
    p.insert<T>("E", T{50000});
    p.insert<T>("n", T{1.2});
    p.insert<T>("m", T{0.8});
    p.insert<std::string>("temperature_name", "temperature");
    p.insert<std::string>("timer_name", "time");
    p.insert<std::string>("solver_name", "solver");
    ctx.create<nm::autocatalytic_reaction<policy>>(p);
  };

  ctx_type ref;
  {
    param_type p;
    p.clear();
    p.insert<std::string>("name", "temperature");
    p.insert<T>("increment", T{0});
    ref.create<nm::scalar_stepper<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "time");
    p.insert<T>("increment", dt);
    ref.create<nm::scalar_stepper<policy>>(p);
  }
  add_curing_chain(ref);
  ref.finalize();
  ref.get_mutable<T>("temperature", "state") = temp;

  ctx_type ext;
  nm::external_scalar_source<policy>* temp_src = nullptr;
  nm::external_scalar_source<policy>* time_src = nullptr;
  {
    param_type p;
    p.clear();
    p.insert<std::string>("name", "temperature");
    temp_src = &ext.create<nm::external_scalar_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "time");
    time_src = &ext.create<nm::external_scalar_source<policy>>(p);
  }
  add_curing_chain(ext);
  ext.finalize();

  for (int step = 0; step < 20; ++step) {
    ref.update();

    temp_src->bind(temp, temp);
    time_src->bind(dt * step, dt * (step + 1));
    ext.update();

    EXPECT_DOUBLE_EQ(ext.get<T>("curing", "current_state"),
                     ref.get<T>("curing", "current_state"))
        << "step " << step;

    ref.commit();
    ext.commit();
  }

  EXPECT_GT(ext.get<T>("curing", "current_state"), 0.99);
}

// ---------------------------------------------------------------------------
// Buffer binding
// ---------------------------------------------------------------------------

/// The raw-buffer overload must apply the engineering-shear convention, and the
/// old/new split must land on the right sides.
TEST(ExternalStrainSource, BindsFromCanonicalBuffersWithEngineeringShear) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "stepper");
  auto& src = ctx.create<nm::external_strain_source<policy>>(p);
  ctx.finalize();

  // STRAN and STRAN + DSTRAN, Abaqus/Standard order, engineering shear.
  const T stran[6] = {0.001, 0.002, 0.003, 0.008, 0.010, 0.012};
  T total[6];
  const T dstran[6] = {0.0005, 0.0, 0.0, 0.002, 0.0, 0.0};
  for (std::size_t i = 0; i < 6; ++i) total[i] = stran[i] + dstran[i];

  src.bind(stran, total);

  const auto& old_eps = src.strain().old_value();
  const auto& new_eps = src.strain().new_value();

  EXPECT_DOUBLE_EQ(old_eps(0, 0), 0.001);
  EXPECT_DOUBLE_EQ(old_eps(1, 1), 0.002);
  EXPECT_DOUBLE_EQ(old_eps(2, 2), 0.003);
  EXPECT_DOUBLE_EQ(old_eps(0, 1), 0.004);  // gamma_12 / 2
  EXPECT_DOUBLE_EQ(old_eps(0, 2), 0.005);
  EXPECT_DOUBLE_EQ(old_eps(1, 2), 0.006);

  EXPECT_DOUBLE_EQ(new_eps(0, 0), 0.0015);
  EXPECT_DOUBLE_EQ(new_eps(0, 1), 0.005);

  // The increment a rate-dependent material would recover.
  EXPECT_DOUBLE_EQ(new_eps(0, 0) - old_eps(0, 0), 0.0005);
  EXPECT_DOUBLE_EQ(new_eps(0, 1) - old_eps(0, 1), 0.001);
}

/// A plain input_property consumer binds new_value(); this is what lets every
/// existing strain consumer work against a history-valued "strain" unchanged.
TEST(ExternalStrainSource, PlainConsumerSeesNewValue) {
  ctx_type ctx;
  param_type p;
  p.clear();
  p.insert<std::string>("name", "stepper");
  auto& src = ctx.create<nm::external_strain_source<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "stepper");
  p.insert<T>("K", K);
  p.insert<T>("G", G);
  ctx.create<nm::linear_elasticity<policy>>(p);
  ctx.finalize();

  src.bind(uniaxial(0.001), uniaxial(0.002));
  ctx.update();

  // linear_elasticity uses add_input<tensor2>, so it must see the NEW strain.
  const auto& sig = ctx.get<tensor2>("elastic", "stress");
  const T expected = (K + 4.0 * G / 3.0) * 0.002;
  EXPECT_NEAR(sig(0, 0), expected, 1e-12);
}

}  // namespace
