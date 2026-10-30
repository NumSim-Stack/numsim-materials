// Iteration groups: a fixed point closed over local edges runs inside update() (#214).
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <ranges>
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

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

/// The names of the callbacks in the order they ran.
std::vector<std::string> g_log;

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
    g_log.push_back(base::name());
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
        m_x(base::template add_input<T>(base::template get_parameter<std::string>("source"),
                                        base::template get_parameter<std::string>("property"), nm::EdgeKind::Local)) {
    base::add_callback(&guess::copy, m_used).reads(m_x);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<std::string>(para, "source");
    nm::optional<std::string>(para, "property", std::string{"x"});
    return para;
  }

  void copy() {
    g_log.push_back(base::name());
    m_used = m_x.get();
  }

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
    return para;
  }

  void compute() {
    g_log.push_back(base::name());
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

/// "x" = "<source>" (its output), a callback that only passes a value on.
template <typename Traits>
class follow final : public nm::material_base<follow<Traits>, Traits> {
public:
  using base = nm::material_base<follow<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit follow(Args &&...args)
      : base(std::forward<Args>(args)...), m_x(base::template add_output<T>("x")), m_in(source(base::template get_parameter<std::string>("source"))) {
    base::add_callback(&follow::copy, m_x).reads(m_in);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<std::string>(para, "source");
    return para;
  }

  void copy() { m_x = m_in.get(); }

private:
  T &m_x;
  nm::input_property<T, nm::property_traits> const &m_in;

  nm::input_property<T, nm::property_traits> const &source(std::string const &s) {
    auto const src{nm::connection_source::parse(s)};
    return base::template add_input<T>(src.material, src.property);
  }
};

/// Two halves of a solver (as j2_plasticity and its hardening law): each reads the other over a local edge, and
/// each counts its runs (stateful: never folded, and an error inside a group).
template <typename Traits>
class half final : public nm::material_base<half<Traits>, Traits> {
public:
  using base = nm::material_base<half<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit half(Args &&...args)
      : base(std::forward<Args>(args)...), m_out(base::template add_output<T>("out")), m_calls(base::template add_output<T>("calls")),
        m_in(base::template add_input<T>(base::template get_parameter<std::string>("other"), "out", nm::EdgeKind::Local)) {
    base::add_callback(&half::compute, m_out, m_calls).reads(m_in).stateful();
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<std::string>(para, "other");
    return para;
  }

  void compute() {
    m_out = m_in.get() + 1;
    m_calls += 1;
  }

  // a deleted overload keeps the pointers the declaration stores from dangling
  template <typename Self = half>
  static constexpr bool accepts_references = requires(Self &s, T &r, T const &tol, int const &n) { s.converges(r, tol, n); };
  template <typename Self = half>
  static constexpr bool accepts_literals = requires(Self &s, T &r) { s.converges(r, 1e-10, 100); };
  template <typename Self = half>
  static constexpr bool accepts_literal_tolerance = requires(Self &s, T &r, int const &n) { s.converges(r, 1e-10, n); };
  template <typename Self = half>
  static constexpr bool accepts_literal_passes = requires(Self &s, T &r, T const &tol) { s.converges(r, tol, 100); };
  template <typename Self = half>
  static constexpr bool declares_literals = requires(Self &s, T &r) { s.declare_converges(r, 1e-10, 100); };
  template <typename Self = half>
  static constexpr bool declares_references = requires(Self &s, T &r, T const &tol, int const &n) { s.declare_converges(r, tol, n); };

private:
  T &m_out;
  T &m_calls;
  nm::input_property<T, nm::property_traits> const &m_in;
};

static_assert(half<policy>::accepts_references<> && half<policy>::declares_references<>);
static_assert(!half<policy>::accepts_literals<> && !half<policy>::accepts_literal_tolerance<> && !half<policy>::accepts_literal_passes<>,
              "converges() keeps pointers to its arguments: a temporary would dangle");
static_assert(!half<policy>::declares_literals<>);

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

TEST(IterationGroups, AGroupDownstreamOfAnotherRunsAfterItInTheCallLog) {
  auto ctx{build({}, false, false)};
  param_type p;
  p.insert<std::string>("name", "guess2");
  p.insert<std::string>("source", "relax2");
  ctx->create<guess<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "relax2");
  p.insert<std::string>("guess", "guess2");
  p.insert<std::string>("target_source", "relax::x");
  ctx->create<relaxer<policy>>(p);
  ctx->finalize();
  g_log.clear();
  ctx->update();
  auto const first{[&](std::string const &name) { return std::ranges::find(g_log, name) - g_log.begin(); }};
  auto const last{[&](std::string const &name) { return std::ranges::find_last(g_log, name).begin() - g_log.begin(); }};
  EXPECT_EQ(first("probe"), 0);
  EXPECT_EQ(std::ranges::count(g_log, "probe"), 1);
  EXPECT_LT(last("relax"), first("guess2")) << "the second group starts after the first has converged";
  EXPECT_LT(last("guess"), first("guess2"));
  EXPECT_GT(std::ranges::count(g_log, "relax2"), 1) << "and iterates";
  EXPECT_EQ(g_log.back(), "relax2");
}

TEST(IterationGroups, GroupsThatReadEachOtherThrowAtFinalize) {
  param_type relax;
  relax.insert<std::string>("target_source", "guess2::used"); // each group reads the front of the other
  auto ctx{build(relax, false, false)};
  param_type p;
  p.insert<std::string>("name", "guess2");
  p.insert<std::string>("source", "relax2");
  ctx->create<guess<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "relax2");
  p.insert<std::string>("guess", "guess2");
  p.insert<std::string>("target_source", "guess::used");
  ctx->create<relaxer<policy>>(p);
  auto const message{thrown<std::logic_error>([&] { ctx->finalize(); })};
  EXPECT_NE(message.find("relax::residual"), std::string::npos) << message;
  EXPECT_NE(message.find("relax2::residual"), std::string::npos) << message;
  EXPECT_NE(message.find("read each other"), std::string::npos) << message;
}

TEST(IterationGroups, ALocalEdgeWhoseProducerIsOutsideTheGroupThrowsAtFinalize) {
  ctx_type ctx; // guess reads probe::target of the previous pass, but probe is not downstream of guess
  constant(ctx, "goal", 2);
  param_type p;
  p.insert<std::string>("name", "probe");
  ctx.create<probe<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "guess");
  p.insert<std::string>("source", "probe");
  p.insert<std::string>("property", "target");
  ctx.create<guess<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "relax");
  ctx.create<relaxer<policy>>(p);
  auto const message{thrown<std::logic_error>([&] { ctx.finalize(); })};
  EXPECT_NE(message.find("relax::residual"), std::string::npos) << message;
  EXPECT_NE(message.find("closed local loop"), std::string::npos) << message;
}

TEST(IterationGroups, AResidualInTheMiddleOfTheLoopThrowsAtFinalize) {
  ctx_type ctx; // guess -> relax -> follow -> (local) guess: the producer of the edge is behind the residual
  constant(ctx, "goal", 2);
  param_type p;
  p.insert<std::string>("name", "probe");
  ctx.create<probe<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "guess");
  p.insert<std::string>("source", "follow");
  ctx.create<guess<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "relax");
  ctx.create<relaxer<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "follow");
  p.insert<std::string>("source", "relax::x");
  ctx.create<follow<policy>>(p);
  auto const message{thrown<std::logic_error>([&] { ctx.finalize(); })};
  EXPECT_NE(message.find("relax::residual"), std::string::npos) << message;
  EXPECT_NE(message.find("closed local loop"), std::string::npos) << message;
}

TEST(IterationGroups, ANestedLoopThrowsAtFinalize) {
  ctx_type ctx; // the inner loop (gi, ri) feeds the outer one (go, ro), which feeds the inner one back through ro::x
  constant(ctx, "goal", 2);
  param_type p;
  p.insert<std::string>("name", "probe");
  ctx.create<probe<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "go");
  p.insert<std::string>("source", "ro");
  ctx.create<guess<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "gi");
  p.insert<std::string>("source", "ri");
  ctx.create<guess<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "ri");
  p.insert<std::string>("guess", "gi");
  p.insert<std::string>("target_source", "go::used");
  ctx.create<relaxer<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "ro");
  p.insert<std::string>("guess", "go");
  p.insert<std::string>("target_source", "ri::x");
  ctx.create<relaxer<policy>>(p);
  auto const message{thrown<std::logic_error>([&] { ctx.finalize(); })};
  EXPECT_NE(message.find("ri::residual"), std::string::npos) << message;
  EXPECT_NE(message.find("ro::residual"), std::string::npos) << message;
  EXPECT_NE(message.find("nested"), std::string::npos) << message;
}

TEST(IterationGroups, ALoopThroughTheGroupOfAnotherResidualThrowsAtFinalize) {
  param_type relax; // relax2 -> (local) guess3 -> relax (target_source) -> relax2: the loop runs through relax's group
  relax.insert<std::string>("target_source", "guess3::used");
  auto ctx{build(relax, false, false)};
  param_type p;
  p.insert<std::string>("name", "guess3");
  p.insert<std::string>("source", "relax2");
  ctx->create<guess<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "relax2");
  p.insert<std::string>("guess", "guess3");
  p.insert<std::string>("target_source", "relax::x");
  ctx->create<relaxer<policy>>(p);
  auto const message{thrown<std::logic_error>([&] { ctx->finalize(); })};
  EXPECT_NE(message.find("relax2::residual"), std::string::npos) << message;
  EXPECT_NE(message.find("relax::residual"), std::string::npos) << message;
  EXPECT_NE(message.find("nested"), std::string::npos) << message;
}

TEST(IterationGroups, ASolverWithLocalEdgesUpstreamOfAResidualRunsOncePerUpdate) {
  ctx_type ctx; // a and b read each other over local edges, as j2_plasticity and its hardening law do
  constant(ctx, "goal", 2);
  param_type p;
  p.insert<std::string>("name", "a");
  p.insert<std::string>("other", "b");
  ctx.create<half<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "b");
  p.insert<std::string>("other", "a");
  ctx.create<half<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "probe");
  ctx.create<probe<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "guess");
  p.insert<std::string>("source", "relax");
  ctx.create<guess<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "relax");
  p.insert<std::string>("target_source", "b::out");
  ctx.create<relaxer<policy>>(p);
  ASSERT_NO_THROW(ctx.finalize()) << "neither is a member of the group: both are stateful";
  for (int n{1}; n <= 3; ++n) {
    ctx.update();
    EXPECT_EQ(ctx.get<T>("a", "calls"), n);
    EXPECT_EQ(ctx.get<T>("b", "calls"), n);
    EXPECT_LE(ctx.get<T>("relax", "residual"), 1e-8);
  }
}

TEST(IterationGroups, AnInfiniteResidualIsNotConvergedByAnInfiniteTolerance) {
  param_type relax;
  relax.insert<T>("tolerance", std::numeric_limits<T>::infinity());
  auto ctx{build(relax)};
  dynamic_cast<nm::constant_scalar<policy> &>(*ctx->find("goal")).set_parameter<T>("value", std::numeric_limits<T>::infinity());
  auto const message{thrown<nm::cutback_error>([&] { ctx->update(); })};
  EXPECT_NE(message.find("in 1 pass "), std::string::npos) << "stops at the first pass: " << message;
}

TEST(IterationGroups, SettingsChangedAfterFinalizeAreCheckedAtEveryUpdate) {
  auto ctx{build()};
  auto &relax{dynamic_cast<relaxer<policy> &>(*ctx->find("relax"))};
  relax.set_parameter<T>("tolerance", -1);
  EXPECT_NE(thrown<std::invalid_argument>([&] { ctx->update(); }).find("relax::residual"), std::string::npos);
  ctx->revert();
  relax.set_parameter<T>("tolerance", 1e-8);
  relax.set_parameter<int>("max_passes", 0);
  EXPECT_NE(thrown<std::invalid_argument>([&] { ctx->update(); }).find("relax::residual"), std::string::npos);
  ctx->revert();
  relax.set_parameter<int>("max_passes", 100);
  EXPECT_NO_THROW(ctx->update());
}

TEST(IterationGroups, ConvergesWithoutALocalEdgeBehindItThrowsAtFinalize) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "alone");
  ctx.create<lonely<policy>>(p);
  auto const message{thrown<std::logic_error>([&] { ctx.finalize(); })};
  EXPECT_NE(message.find("alone::residual"), std::string::npos) << message;
  EXPECT_NE(message.find("closed local loop"), std::string::npos) << message;
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

} // namespace
