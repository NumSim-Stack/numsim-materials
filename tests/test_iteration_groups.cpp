// Iteration groups: a fixed point closed over local edges runs inside update() (#214).
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "numsim-materials/core/errors.h"
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/driver/material_point.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/postprocessing/property_recorder.h"
#include "numsim-materials/solvers/system.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

/// "target" = goal::value; "calls" counts its runs (stateful: never folded, never in a group).
template <typename Traits>
class probe final : public nm::material_base<probe<Traits>, Traits> {
public:
  using base = nm::material_base<probe<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit probe(Args &&...args)
      : base(std::forward<Args>(args)...), m_target(base::template add_output<T>("target")),
        m_calls(base::template add_output<T>("calls")), m_goal(base::template add_input<T>("goal", "value")) {
    base::add_callback(&probe::compute, m_target, m_calls).reads(m_goal).stateful();
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() {
    m_target = m_goal.get();
    m_calls += 1;
  }

private:
  T &m_target;
  T &m_calls;
  nm::input_property<T, nm::property_traits> const &m_goal;
};

/// "used" = "<source>::x" of the previous pass, over a local edge (as a phase reads its tangent).
template <typename Traits>
class guess final : public nm::material_base<guess<Traits>, Traits> {
public:
  using base = nm::material_base<guess<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit guess(Args &&...args)
      : base(std::forward<Args>(args)...), m_used(base::template add_output<T>("used")),
        m_x(base::template add_input<T>(base::template get_parameter<std::string>("source"), "x", nm::EdgeKind::Local)) {
    base::add_callback(&guess::copy, m_used).reads(m_x);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<std::string>(para, "source");
    return para;
  }

  void copy() { m_used = m_x.get(); }

private:
  T &m_used;
  nm::input_property<T, nm::property_traits> const &m_x;
};

/// x = (used + target) / 2 and residual = |x - used|: every pass halves the distance to the target. It
/// declares converges(residual), and with "twice" a second residual "check" over the same callbacks.
template <typename Traits>
class relaxer final : public nm::material_base<relaxer<Traits>, Traits> {
public:
  using base = nm::material_base<relaxer<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit relaxer(Args &&...args)
      : base(std::forward<Args>(args)...), m_x(base::template add_output<T>("x")),
        m_residual(base::template add_output<T>("residual")), m_check(base::template add_output<T>("check")),
        m_used(base::template add_input<T>(base::template get_parameter<std::string>("guess"), "used")),
        m_target(target(base::template get_parameter<std::string>("target_source"))) {
    auto handle{base::add_callback(&relaxer::compute, m_x, m_residual, m_check).reads(m_used, m_target)};
    if (base::template get_parameter<bool>("stateful"))
      handle.stateful();
    auto const &tolerance{base::template get_parameter<T>("tolerance")};
    auto const &max_passes{base::template get_parameter<int>("max_passes")};
    if (base::template get_parameter<bool>("converges"))
      base::converges(m_residual, tolerance, max_passes);
    if (base::template get_parameter<bool>("twice"))
      base::converges(m_check, tolerance, max_passes);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<std::string>(para, "guess", std::string{"guess"});
    nm::optional<std::string>(para, "target_source", std::string{"probe::target"});
    nm::optional<T>(para, "tolerance", T{1e-8});
    nm::optional<int>(para, "max_passes", 100);
    nm::optional<bool>(para, "stateful", false);
    nm::optional<bool>(para, "twice", false);
    nm::optional<bool>(para, "converges", true);
    return para;
  }

  void compute() {
    m_x = (m_used.get() + m_target.get()) / 2;
    m_residual = std::abs(m_x - m_used.get());
    m_check = m_residual;
  }

private:
  T &m_x;
  T &m_residual;
  T &m_check;
  nm::input_property<T, nm::property_traits> const &m_used;
  nm::input_property<T, nm::property_traits> const &m_target;

  nm::input_property<T, nm::property_traits> const &target(std::string const &source) {
    auto const src{nm::connection_source::parse(source)};
    return base::template add_input<T>(src.material, src.property);
  }
};

/// "residual" = 1, declared converging with no local edge behind it.
template <typename Traits>
class lonely final : public nm::material_base<lonely<Traits>, Traits> {
public:
  using base = nm::material_base<lonely<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit lonely(Args &&...args)
      : base(std::forward<Args>(args)...), m_residual(base::template add_output<T>("residual")) {
    base::add_callback(&lonely::compute, m_residual).reads();
    base::converges(m_residual, base::template get_parameter<T>("tolerance"), base::template get_parameter<int>("max_passes"));
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<T>(para, "tolerance", T{1e-8});
    nm::optional<int>(para, "max_passes", 10);
    return para;
  }

  void compute() { m_residual = 1; }

private:
  T &m_residual;
};

void constant(ctx_type &ctx, std::string const &name, T value) {
  param_type p;
  p.insert<std::string>("name", name);
  p.insert<T>("value", value);
  ctx.create<nm::constant_scalar<policy>>(p);
}

/// goal -> probe -> [guess -> relax] (the group, closed by guess reading relax::x over a local edge), and with
/// `recorded` a property_recorder of relax::x after it. `relax` adds parameters of the relaxer.
std::unique_ptr<ctx_type> build(param_type relax = {}, bool recorded = false, bool finalize = true) {
  auto owner{std::make_unique<ctx_type>()};
  auto &ctx{*owner};
  constant(ctx, "goal", 2);
  param_type p;
  p.insert<std::string>("name", "probe");
  ctx.create<probe<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "guess");
  p.insert<std::string>("source", "relax");
  ctx.create<guess<policy>>(p);
  relax.insert<std::string>("name", "relax");
  ctx.create<relaxer<policy>>(relax);
  if (recorded) {
    p.clear();
    p.insert<std::string>("name", "rec");
    p.insert<std::vector<std::string>>("scalar_sources", {"relax::x"});
    ctx.create<nm::property_recorder<policy>>(p);
  }
  if (finalize)
    ctx.finalize();
  return owner;
}

/// The message of the exception of type E that `f` throws; empty when it throws none.
template <typename E, typename F>
std::string thrown(F &&f) {
  try {
    f();
  } catch (E const &e) {
    return e.what();
  }
  return {};
}

TEST(IterationGroups, UpdateIteratesToTheResidual) {
  auto ctx{build()};
  ctx->update();
  EXPECT_LE(ctx->get<T>("relax", "residual"), 1e-8);
  EXPECT_NEAR(ctx->get<T>("relax", "x"), 2, 1e-7);
}

TEST(IterationGroups, ACallbackBeforeTheGroupRunsOncePerUpdate) {
  auto ctx{build()};
  for (int n{1}; n <= 3; ++n) {
    ctx->update();
    EXPECT_EQ(ctx->get<T>("probe", "calls"), n) << "the group took about 27 passes, the probe ran once";
  }
}

TEST(IterationGroups, ARecorderAfterTheGroupWritesOneRowPerUpdate) {
  auto ctx{build({}, true)};
  for (int n{0}; n < 3; ++n)
    ctx->update();
  auto const &rec{dynamic_cast<nm::property_recorder<policy> &>(*ctx->find("rec"))};
  ASSERT_EQ(rec.buffer().rows(), 3u);
  EXPECT_NEAR(rec.buffer().at(0, 0), 2, 1e-7) << "the row holds the converged value";
}

// residual_source is deprecated: the material point finds the residual converged after the one update
TEST(IterationGroups, TheMaterialPointResidualSourceStillWorks) {
  auto ctx{build({}, true)};
  param_type p;
  p.insert<std::string>("residual_source", "relax::residual");
  p.insert<T>("residual_tolerance", 1e-8);
  nm::material_point<policy> point{*ctx, p};
  for (int n{0}; n < 3; ++n)
    point.evaluate();
  EXPECT_EQ(dynamic_cast<nm::property_recorder<policy> &>(*ctx->find("rec")).buffer().rows(), 3u);
  EXPECT_EQ(ctx->get<T>("probe", "calls"), 3);
}

TEST(IterationGroups, TooFewPassesThrowACutbackNamingTheResidual) {
  param_type relax;
  relax.insert<int>("max_passes", 3);
  auto ctx{build(relax)};
  auto const message{thrown<nm::cutback_error>([&] { ctx->update(); })};
  EXPECT_NE(message.find("relax::residual"), std::string::npos) << message;
  EXPECT_NE(message.find("3 passes"), std::string::npos) << message;
  EXPECT_THROW(ctx->update(), std::logic_error) << "the context is failed until revert()";
  ctx->revert();
  dynamic_cast<relaxer<policy> &>(*ctx->find("relax")).set_parameter<int>("max_passes", 100);
  EXPECT_NO_THROW(ctx->update()) << "the parameter is read on every update";
}

TEST(IterationGroups, ANaNResidualNeverConverges) {
  auto ctx{build()};
  dynamic_cast<nm::constant_scalar<policy> &>(*ctx->find("goal")).set_parameter<T>("value", std::numeric_limits<T>::quiet_NaN());
  auto const message{thrown<nm::cutback_error>([&] { ctx->update(); })};
  EXPECT_NE(message.find("in 1 pass "), std::string::npos) << "stops at the first pass: " << message;
}

TEST(IterationGroups, UpdatePropertyOnAMemberRunsItsUpstreamOnce) {
  auto ctx{build()};
  ctx->update_property("relax", "residual");
  EXPECT_EQ(ctx->get<T>("relax", "x"), 1) << "one pass from x = 0: no iteration";
  EXPECT_EQ(ctx->get<T>("relax", "residual"), 1);
}

TEST(IterationGroups, ConstantsUpstreamStayFoldedAndFollowSetParameter) {
  auto ctx{build()};
  ctx->update();
  EXPECT_TRUE(ctx->is_constant("goal", "value"));
  dynamic_cast<nm::constant_scalar<policy> &>(*ctx->find("goal")).set_parameter<T>("value", 4);
  ctx->update();
  EXPECT_NEAR(ctx->get<T>("relax", "x"), 4, 1e-7);
  EXPECT_LE(ctx->get<T>("relax", "residual"), 1e-8);
}

TEST(IterationGroups, AGroupDownstreamOfAnotherRunsAfterIt) {
  auto ctx{build({}, false, false)};
  param_type p;
  p.insert<std::string>("name", "guess2");
  p.insert<std::string>("source", "relax2");
  ctx->create<guess<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "relax2");
  p.insert<std::string>("guess", "guess2");
  p.insert<std::string>("target_source", "relax::x"); // the converged x of the first group
  ctx->create<relaxer<policy>>(p);
  ctx->finalize();
  ctx->update();
  EXPECT_LE(ctx->get<T>("relax", "residual"), 1e-8);
  EXPECT_LE(ctx->get<T>("relax2", "residual"), 1e-8);
  EXPECT_NEAR(ctx->get<T>("relax2", "x"), 2, 1e-7);
}

TEST(IterationGroups, ConvergesWithoutALocalEdgeBehindItThrowsAtFinalize) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "alone");
  ctx.create<lonely<policy>>(p);
  auto const message{thrown<std::logic_error>([&] { ctx.finalize(); })};
  EXPECT_NE(message.find("alone::residual"), std::string::npos) << message;
  EXPECT_NE(message.find("local edge"), std::string::npos) << message;
}

TEST(IterationGroups, AStatefulCallbackInsideAGroupThrowsAtFinalize) {
  param_type relax;
  relax.insert<bool>("stateful", true);
  auto ctx{build(relax, false, false)};
  auto const message{thrown<std::logic_error>([&] { ctx->finalize(); })};
  EXPECT_NE(message.find("relax::x"), std::string::npos) << message;
  EXPECT_NE(message.find("stateful"), std::string::npos) << message;
}

TEST(IterationGroups, OverlappingGroupsThrowAtFinalize) {
  param_type relax;
  relax.insert<bool>("twice", true);
  auto ctx{build(relax, false, false)};
  auto const message{thrown<std::logic_error>([&] { ctx->finalize(); })};
  EXPECT_NE(message.find("relax::residual"), std::string::npos) << message;
  EXPECT_NE(message.find("relax::check"), std::string::npos) << message;
}

TEST(IterationGroups, InvalidSettingsThrowAtFinalize) {
  for (auto const &[key, value] : {std::pair{"tolerance", -1.0}, std::pair{"tolerance", std::numeric_limits<T>::quiet_NaN()}}) {
    param_type relax;
    relax.insert<T>(key, value);
    auto ctx{build(relax, false, false)};
    EXPECT_NE(thrown<std::invalid_argument>([&] { ctx->finalize(); }).find("relax::residual"), std::string::npos) << value;
  }
  param_type relax;
  relax.insert<int>("max_passes", 0);
  auto ctx{build(relax, false, false)};
  EXPECT_NE(thrown<std::invalid_argument>([&] { ctx->finalize(); }).find("relax::residual"), std::string::npos);
}

// converges() is shorthand for a picard system without unknowns (#218)
TEST(IterationGroups, APicardSystemWithoutUnknownsIsTheGroup) {
  auto const with_system{[](int max_iter) {
    param_type relax;
    relax.insert<bool>("converges", false);
    auto ctx{build(relax, false, false)};
    param_type p;
    p.insert<std::string>("name", "fix");
    p.insert<std::string>("method", "picard");
    p.insert<std::vector<nm::system_equation>>("equations", {{"relax::residual", ""}});
    p.insert<T>("tolerance", 1e-8);
    p.insert<int>("max_iter", max_iter);
    ctx->create<nm::residual_system<policy>>(p);
    ctx->finalize();
    return ctx;
  }};
  auto ctx{with_system(99)};
  auto reference{build()};
  for (int n{1}; n <= 3; ++n) {
    ctx->update();
    reference->update();
    EXPECT_EQ(ctx->get<T>("relax", "x"), reference->get<T>("relax", "x"));
    EXPECT_EQ(ctx->get<T>("probe", "calls"), n);
    if (n == 1) {
      EXPECT_GT(dynamic_cast<nm::residual_system<policy> &>(*ctx->find("fix")).iterations(), 10) << "the passes of the first update";
    }
  }
  auto short_of_it{with_system(2)};
  EXPECT_NE(thrown<nm::cutback_error>([&] { short_of_it->update(); }).find("'fix'"), std::string::npos);
}

} // namespace
