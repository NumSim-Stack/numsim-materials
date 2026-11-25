#include <gtest/gtest.h>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/autocatalytic_reaction.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/linear_isotropic_hardening.h"
#include "numsim-materials/materials/scalar_stepper.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/solvers/backward_euler.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/umat/tensor_conversion.h"
#include "numsim-materials/solvers/local_newton.h"
#include "numsim-materials/materials/j2_plasticity.h"
#include "test_support.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using numsim::materials::test::add;

constexpr T K = 166.67;
constexpr T G = 76.92;
constexpr T sigma_0 = 50.0;
constexpr T H_mod = 1000.0;
constexpr T d_eps = 0.05;

/// Everything downstream of the strain producer. Identical in both contexts —
/// only the producer differs, which is the whole point of the comparison.
void add_j2_chain(ctx_type& ctx) {

  add<nm::linear_elasticity<policy>>(ctx, "elastic", {{"strain_producer_name", "stepper"}, {"K", K}, {"G", G}});
  add<nm::local_newton<policy>>(ctx, "solver");
  add<nm::linear_isotropic_hardening<policy>>(ctx, "hardening", {{"source", "j2"}, {"K", H_mod}});
  add<nm::j2_plasticity<policy>>(ctx, "j2",
      {{"K", K}, {"hardening_source", "hardening"}, {"strain_source", "stepper"}, {"solver_source", "solver"}, {"G", G},
       {"sigma_0", sigma_0}});
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
    add<nm::tensor_component_stepper<2, policy>>(ref, "stepper",
        {{"increment", d_eps}, {"indices", std::vector<std::size_t>{0, 0}}});
  }
  add_j2_chain(ref);
  ref.finalize();

  ctx_type ext;
  auto& src = add<nm::external_strain_source<policy>>(ext, "stepper");
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
    add<nm::backward_euler<policy>>(ctx, "solver", {{"function", "curing"}});
    add<nm::autocatalytic_reaction<policy>>(ctx, "curing",
        {{"A", T{1e6}}, {"E", T{50000}}, {"n", T{1.2}}, {"m", T{0.8}}, {"temperature_name", "temperature"},
         {"timer_name", "time"}, {"solver_name", "solver"}});
  };

  ctx_type ref;
  {
    add<nm::scalar_stepper<policy>>(ref, "temperature", {{"increment", T{0}}});
    add<nm::scalar_stepper<policy>>(ref, "time", {{"increment", dt}});
  }
  add_curing_chain(ref);
  ref.finalize();
  ref.get_mutable<T>("temperature", "state") = temp;

  ctx_type ext;
  nm::external_scalar_source<policy>* temp_src = nullptr;
  nm::external_scalar_source<policy>* time_src = nullptr;
  {
    temp_src = &add<nm::external_scalar_source<policy>>(ext, "temperature");
    time_src = &add<nm::external_scalar_source<policy>>(ext, "time");
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
  auto& src = add<nm::external_strain_source<policy>>(ctx, "stepper");
  ctx.finalize();

  // STRAN and STRAN + DSTRAN, Abaqus/Standard order, engineering shear.
  const T stran[6] = {0.001, 0.002, 0.003, 0.008, 0.010, 0.012};
  T total[6];
  const T dstran[6] = {0.0005, 0.0, 0.0, 0.002, 0.0, 0.0};
  for (std::size_t i = 0; i < 6; ++i) total[i] = stran[i] + dstran[i];

  src.bind(u::strain_from_buffer<T>(stran), u::strain_from_buffer<T>(total));

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
  auto& src = add<nm::external_strain_source<policy>>(ctx, "stepper");
  add<nm::linear_elasticity<policy>>(ctx, "elastic", {{"strain_producer_name", "stepper"}, {"K", K}, {"G", G}});
  ctx.finalize();

  src.bind(uniaxial(0.001), uniaxial(0.002));
  ctx.update();

  // linear_elasticity uses add_input<tensor2>, so it must see the NEW strain.
  const auto& sig = ctx.get<tensor2>("elastic", "stress");
  const T expected = (K + 4.0 * G / 3.0) * 0.002;
  EXPECT_NEAR(sig(0, 0), expected, 1e-12);
}

}  // namespace
