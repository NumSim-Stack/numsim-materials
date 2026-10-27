#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/driver/material_point.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/statev_map.h"
#include "test_fixture_materials.h"

// The material point core (#184 phase 2): input variables, output variables and history management, with the
// history owned either by the host (load/store) or by the engine (accept/reset).

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using tensor2 = tmech::tensor<T, 3, 2>;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using point_type = nm::material_point<policy>;

/// "count" (history) counts the accepted steps, "last" (history) is the strain, "stress" = 2 strain + last_n;
/// throws when |strain| exceeds "limit".
template <typename Traits>
class memory final : public nm::material_base<memory<Traits>, Traits> {
public:
  using base = nm::material_base<memory<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit memory(Args &&...args)
      : base(std::forward<Args>(args)...), m_stress(base::template add_output<tensor2>("stress")),
        m_count(base::template add_history_output<value_type>("count")),
        m_last(base::template add_history_output<tensor2>("last")),
        m_limit(base::template get_parameter<value_type>("limit")),
        m_strain(base::template add_input<tensor2>("strain", "strain", nm::EdgeKind::Global)) {
    base::add_callback(&memory::compute, m_stress, m_count, m_last).reads(m_strain);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<value_type>(para, "limit", value_type{1e30});
    return para;
  }

  void compute() {
    if (tmech::norm(m_strain.get()) > m_limit)
      throw std::runtime_error("memory: above the limit");
    m_count.new_value() = m_count.old_value() + 1;
    m_last.new_value() = m_strain.get();
    m_stress = 2 * m_strain.get() + m_last.old_value();
  }

private:
  tensor2 &m_stress;
  nm::history_property<value_type> &m_count;
  nm::history_property<tensor2> &m_last;
  value_type const &m_limit;
  nm::input_property<tensor2, nm::property_traits> const &m_strain;
};

/// "x" relaxes towards 2 by one halving per update; "residual" is the last change.
template <typename Traits>
class relaxer final : public nm::material_base<relaxer<Traits>, Traits> {
public:
  using base = nm::material_base<relaxer<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit relaxer(Args &&...args)
      : base(std::forward<Args>(args)...), m_x(base::template add_output<value_type>("x")),
        m_residual(base::template add_output<value_type>("residual")) {
    base::add_callback(&relaxer::compute, m_x, m_residual).reads().stateful(); // moves from its last value
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() {
    value_type const next{(m_x + 2) / 2};
    m_residual = std::abs(next - m_x);
    m_x = next;
  }

private:
  value_type &m_x;
  value_type &m_residual;
};

/// strain (external) and "mem", plus an int_counter "tag" with `tagged`.
struct setup {
  ctx_type ctx;

  explicit setup(T limit = 1e30, bool tagged = false) {
    param_type p;
    p.insert<std::string>("name", "strain");
    ctx.create<nm::external_strain_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "mem");
    p.insert<T>("limit", limit);
    ctx.create<memory<policy>>(p);
    if (tagged) {
      p.clear();
      p.insert<std::string>("name", "tag");
      ctx.create<nm::test::int_counter<policy>>(p);
    }
    ctx.finalize();
  }
};

tensor2 strain_at(T e) {
  tensor2 eps;
  eps(0, 0) = e;
  eps(0, 1) = eps(1, 0) = 0.5 * e;
  return eps;
}

/// Point parameters with the strain source as the only input.
param_type strain_input() {
  param_type p;
  p.insert<std::vector<std::string>>("inputs", {"strain"});
  return p;
}

/// Point parameters iterating to the residual `source`.
param_type residual(std::string const &source) {
  param_type p;
  p.insert<std::string>("residual_source", source);
  return p;
}

/// One engine-owned step from `from` to `to`.
tensor2 trial(point_type &point, T from, T to) {
  point.reset();
  point.input<nm::external_strain_source<policy>>("strain").bind(strain_at(from), strain_at(to));
  point.evaluate();
  return point.output<tensor2>("mem", "stress");
}

TEST(MaterialPoint, ResolvesInputsAndOutputs) {
  setup s;
  point_type point{s.ctx, strain_input()};
  auto const stress{trial(point, 0, 1)};
  EXPECT_EQ(stress, tensor2{2 * strain_at(1)});
  EXPECT_EQ(&point.output<tensor2>("mem::stress"), &point.output<tensor2>("mem", "stress"));
  EXPECT_EQ(point.history<T>("mem", "count").new_value(), 1);
}

TEST(MaterialPoint, RejectsUnknownOrMistypedNames) {
  setup s;
  auto unknown{strain_input()};
  unknown.insert<std::vector<std::string>>("inputs", {"nothing"});
  EXPECT_THROW((point_type{s.ctx, unknown}), nm::umat::fatal_error);
  point_type point{s.ctx, strain_input()};
  EXPECT_THROW(point.input<nm::external_strain_source<policy>>("mem"), nm::umat::fatal_error) << "not an input";
  EXPECT_THROW(point.input<nm::external_scalar_source<policy>>("strain"), nm::umat::fatal_error) << "another source type";
  EXPECT_THROW(point.output<tensor2>("mem", "nothing"), nm::umat::fatal_error);
  EXPECT_THROW(point.output<T>("mem", "stress"), nm::umat::fatal_error);
  EXPECT_THROW(point.history<T>("mem", "stress"), nm::umat::fatal_error) << "not a history";
}

TEST(MaterialPoint, EngineOwnedHistoryAcceptsAndResets) {
  setup s;
  point_type point{s.ctx, strain_input()};
  trial(point, 0, 1);
  point.accept();
  auto const first{trial(point, 1, 2)};
  EXPECT_EQ(first, tensor2{2 * strain_at(2) + strain_at(1)});
  EXPECT_EQ(trial(point, 1, 2), first) << "a repeated trial after reset() is the same";
  trial(point, 1, 5); // rejected
  EXPECT_EQ(trial(point, 1, 2), first) << "a rejected trial leaves no trace";
  point.accept();
  EXPECT_EQ(point.history<T>("mem", "count").new_value(), 2);
}

TEST(MaterialPoint, TheAcceptedLoadIsTheOldValueOfItsSource) {
  setup s;
  point_type point{s.ctx, strain_input()};
  auto const &source{point.input<nm::external_strain_source<policy>>("strain")};
  trial(point, 0, 1);
  point.accept();
  EXPECT_EQ(source.strain().old_value(), strain_at(1)) << "a driver keeps no copy of the accepted load";
  trial(point, 1, 3);
  point.reset();
  EXPECT_EQ(source.strain().new_value(), strain_at(1)) << "reset() returns the load to the accepted one";
}

TEST(MaterialPoint, AFailedEvaluationRecoversAfterReset) {
  setup s{1.5};
  point_type point{s.ctx, strain_input()};
  EXPECT_THROW(trial(point, 0, 2), std::runtime_error);
  EXPECT_NO_THROW(trial(point, 0, 1));
  point.accept();
  EXPECT_EQ(point.history<T>("mem", "count").new_value(), 1);
}

TEST(MaterialPoint, HostOwnedHistoryMatchesEngineOwned) {
  setup engine, host;
  point_type a{engine.ctx, strain_input()};
  point_type b{host.ctx, strain_input()};
  nm::umat::statev_map<policy> const reference{host.ctx, {{"strain", "strain"}}};
  ASSERT_EQ(b.history_width(), reference.nstatv()) << "the bound input's history stays with the host";
  EXPECT_EQ(b.describe_history(), reference.describe());

  std::vector<T> statev(b.history_width());
  for (int step{1}; step <= 3; ++step) {
    auto const expected{trial(a, step - 1, step)};
    a.accept();
    b.reset();
    b.load_history(statev);
    b.input<nm::external_strain_source<policy>>("strain").bind(strain_at(step - 1), strain_at(step));
    b.evaluate();
    EXPECT_EQ(b.output<tensor2>("mem", "stress"), expected) << "step " << step;
    b.store_history(statev);
  }
  EXPECT_EQ(statev[0], 3) << "count";
}

TEST(MaterialPoint, HostBufferShorterThanTheHistoryIsFatal) {
  setup s;
  point_type point{s.ctx, strain_input()};
  std::vector<T> statev(point.history_width() - 1);
  EXPECT_THROW(point.load_history(statev), nm::umat::fatal_error);
  EXPECT_THROW(point.store_history(statev), nm::umat::fatal_error);
}

TEST(MaterialPoint, HostOwnedKeepsFurtherExclusionsOut) {
  setup s;
  auto parameters{strain_input()};
  parameters.insert<std::vector<std::string>>("host_owned", {"mem::count"});
  point_type point{s.ctx, parameters};
  EXPECT_EQ(point.history_width(), nm::umat::canonical_width);
}

TEST(MaterialPoint, ParametersHaveDefaultsAndAreChecked) {
  setup s;
  point_type plain{s.ctx, param_type{}};
  EXPECT_FALSE(plain.has_live_props());
  EXPECT_THROW(plain.input<nm::external_strain_source<policy>>("strain"), nm::umat::fatal_error) << "no inputs by default";
  auto malformed{strain_input()};
  malformed.insert<std::vector<std::string>>("host_owned", {"count"});
  EXPECT_ANY_THROW((point_type{s.ctx, malformed})) << "host_owned names material::property";
  auto mistyped{param_type{}};
  mistyped.insert<std::string>("inputs", "strain");
  EXPECT_ANY_THROW((point_type{s.ctx, mistyped})) << "inputs is a list";
}

TEST(MaterialPoint, EngineOwnedHistoryNeedsNoHostLayout) {
  setup s{1e30, true};
  point_type point{s.ctx, strain_input()};
  trial(point, 0, 1);
  point.accept();
  trial(point, 1, 2);
  point.accept();
  EXPECT_EQ(point.history<int>("tag", "count").new_value(), 2);
  EXPECT_THROW(static_cast<void>(point.history_width()), nm::umat::fatal_error) << "an int has no STATEV layout";
}

TEST(MaterialPoint, CheckpointsThroughStreams) {
  setup s, restarted;
  point_type point{s.ctx, strain_input()};
  trial(point, 0, 1);
  point.accept();
  std::stringstream checkpoint;
  point.save(checkpoint);
  auto const expected{trial(point, 1, 2)};

  point_type copy{restarted.ctx, strain_input()};
  copy.load(checkpoint);
  EXPECT_EQ(trial(copy, 1, 2), expected);
  EXPECT_EQ(copy.history<T>("mem", "count").new_value(), 2);
}

TEST(MaterialPoint, IteratesToTheResidual) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "relax");
  ctx.create<relaxer<policy>>(p);
  ctx.finalize();
  auto parameters{residual("relax::residual")};
  parameters.insert<T>("residual_tolerance", 1e-8);
  point_type point{ctx, parameters};
  point.evaluate();
  EXPECT_LE(point.output<T>("relax", "residual"), 1e-8);
  EXPECT_NEAR(point.output<T>("relax", "x"), 2, 1e-7);

  ctx_type short_ctx;
  short_ctx.create<relaxer<policy>>(p);
  short_ctx.finalize();
  parameters.insert<int>("max_passes", 3);
  point_type hurried{short_ctx, parameters};
  EXPECT_THROW(hurried.evaluate(), nm::umat::cutback_error) << "the host may cut the step back";
}

TEST(MaterialPoint, ResidualCountsByMagnitudeAndNaNNeverConverges) {
  for (T const value : {T{-1}, std::numeric_limits<T>::quiet_NaN()}) {
    ctx_type ctx;
    param_type p;
    p.insert<std::string>("name", "stuck");
    p.insert<T>("value", value);
    ctx.create<nm::constant_scalar<policy>>(p);
    ctx.finalize();
    auto parameters{residual("stuck::value")};
    parameters.insert<int>("max_passes", 3);
    point_type point{ctx, parameters};
    EXPECT_THROW(point.evaluate(), nm::umat::cutback_error) << "residual " << value;
  }
}

TEST(MaterialPoint, InvalidResidualSettingsAreFatal) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "relax");
  ctx.create<relaxer<policy>>(p);
  ctx.finalize();
  for (T const tolerance : {T{-1}, std::numeric_limits<T>::quiet_NaN()}) {
    auto parameters{residual("relax::residual")};
    parameters.insert<T>("residual_tolerance", tolerance);
    EXPECT_THROW((point_type{ctx, parameters}), nm::umat::fatal_error);
  }
  auto parameters{residual("relax::residual")};
  parameters.insert<int>("max_passes", 0);
  EXPECT_THROW((point_type{ctx, parameters}), nm::umat::fatal_error);
}

} // namespace
