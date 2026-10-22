/// Constant properties: an output declared a function of parameters and of reads that are constant too is
/// computed once, not on every update; a write through set_parameter() or get_mutable() recomputes it.
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <string>
#include <tmech/tmech.h>
#include <vector>

#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/umat/external_state_source.h"

#define EXPECT_THAT_MESSAGE(statement, exception, text)                                                       \
  try {                                                                                                     \
    statement;                                                                                              \
    ADD_FAILURE() << #statement " did not throw";                                                           \
  } catch (exception const &e) {                                                                            \
    EXPECT_NE(std::string{e.what()}.find(text), std::string::npos) << e.what();                             \
  }

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

/// "value" = factor * source (factor alone without a source), counting its callback. With "constant" it
/// declares "value" constant; with "extra" it also writes an undeclared output from the same callback;
/// "other" adds an input "value" does not use, "narrow" declares the read of "source" only; "hidden" adds the
/// call count (state the declaration hides); "bogus" declares a parameter, "foreign" a read it does not own;
/// "late" declares from its callback, after finalize().
template <typename Traits>
class counter final : public nm::material_base<counter<Traits>, Traits> {
public:
  using base = nm::material_base<counter<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit counter(Args &&...args)
      : base(std::forward<Args>(args)...), m_value(base::template add_output<value_type>("value", &counter::compute)),
        m_factor(base::template get_parameter<value_type>("factor")) {
    auto const &source{base::template get_parameter<std::string>("source")};
    if (!source.empty())
      m_source = &base::template add_input<value_type>(source, base::template get_parameter<std::string>("property"),
                                                       base::template get_parameter<bool>("local") ? nm::EdgeKind::Local
                                                                                                   : nm::EdgeKind::Global);
    if (base::template get_parameter<bool>("extra"))
      m_extra = &base::template add_output<value_type>("extra");
    auto const &other{base::template get_parameter<std::string>("other")};
    if (!other.empty())
      base::template add_input<value_type>(other, "state", nm::EdgeKind::Global);
    if (base::template get_parameter<bool>("bogus"))
      base::constant_output(m_factor);
    if (base::template get_parameter<bool>("foreign"))
      base::constant_output(m_value, {&m_foreign});
    if (base::template get_parameter<bool>("constant")) {
      if (base::template get_parameter<bool>("narrow"))
        base::constant_output(m_value, {m_source});
      else
        base::constant_output(m_value);
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<value_type>(para, "factor", value_type{1});
    nm::optional<std::string>(para, "source", std::string{});
    nm::optional<std::string>(para, "property", std::string{"value"});
    nm::optional<bool>(para, "local", false);
    nm::optional<bool>(para, "constant", true);
    nm::optional<bool>(para, "extra", false);
    nm::optional<std::string>(para, "other", std::string{});
    nm::optional<bool>(para, "narrow", false);
    nm::optional<bool>(para, "hidden", false);
    nm::optional<bool>(para, "bogus", false);
    nm::optional<bool>(para, "foreign", false);
    nm::optional<bool>(para, "late", false);
    return para;
  }

  void compute() {
    if (base::template get_parameter<bool>("late"))
      base::constant_output(m_value);
    ++calls;
    m_value = m_factor * (m_source ? m_source->get() : value_type{1}) + (m_hidden ? calls : 0);
    if (m_extra)
      *m_extra = m_value;
  }

  int calls{0};

private:
  value_type &m_value;
  value_type const &m_factor;
  nm::input_property<value_type, nm::property_traits> const *m_source{nullptr};
  value_type *m_extra{nullptr};
  bool const &m_hidden{base::template get_parameter<bool>("hidden")};
  nm::input_property<value_type, nm::property_traits> m_foreign{"elsewhere", "value", nm::EdgeKind::Global};
};

/// Two callbacks: "slow" = factor * source (declared constant), also writing "slow_extra"; "fast" = slow * x
/// for the varying "x". "attribute" names "slow_extra" written by "slow"; "extra_constant" declares it constant;
/// "orphan" names a writer without a callback; "loose" orders "plain", which has none; "rewrite" names a second
/// writer of "old".
template <typename Traits>
class two_callbacks final : public nm::material_base<two_callbacks<Traits>, Traits> {
public:
  using base = nm::material_base<two_callbacks<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit two_callbacks(Args &&...args)
      : base(std::forward<Args>(args)...), m_slow(base::template add_output<value_type>("slow", &two_callbacks::compute_slow)),
        m_slow_extra(base::template add_output<value_type>("slow_extra")),
        m_fast(base::template add_output<value_type>("fast", &two_callbacks::compute_fast)),
        m_plain(base::template add_output<value_type>("plain")),
        m_old(base::template add_history_output<value_type>("old")),
        m_factor(base::template get_parameter<value_type>("factor")),
        m_source(base::template add_input<value_type>(base::template get_parameter<std::string>("source"), "value",
                                                      nm::EdgeKind::Global)),
        m_x(base::template add_input<value_type>(base::template get_parameter<std::string>("x"), "state", nm::EdgeKind::Global)) {
    base::constant_output(m_slow, {&m_source});
    base::constant_output(m_plain, {}); // set here, written by no callback
    base::written_by(m_old, m_fast);
    if (base::template get_parameter<bool>("extra_constant"))
      base::constant_output(m_slow_extra, {&m_source});
    if (base::template get_parameter<bool>("attribute"))
      base::written_by(m_slow_extra, m_slow);
    if (base::template get_parameter<bool>("orphan"))
      base::written_by(m_slow_extra, m_plain);
    base::runs_after(m_fast, m_slow);
    if (base::template get_parameter<bool>("cycle"))
      base::runs_after(m_slow, m_fast);
    if (base::template get_parameter<bool>("loose"))
      base::runs_after(m_plain, m_slow);
    if (base::template get_parameter<bool>("rewrite"))
      base::written_by(m_old, m_slow);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<value_type>(para, "factor", value_type{2});
    nm::required<std::string>(para, "source");
    nm::required<std::string>(para, "x");
    nm::optional<bool>(para, "attribute", true);
    nm::optional<bool>(para, "extra_constant", true);
    nm::optional<bool>(para, "orphan", false);
    nm::optional<bool>(para, "cycle", false);
    nm::optional<bool>(para, "loose", false);
    nm::optional<bool>(para, "rewrite", false);
    return para;
  }

  void compute_slow() {
    ++slow_calls;
    m_slow = m_factor * m_source.get();
    m_slow_extra = m_slow + (base::template get_parameter<bool>("extra_constant") ? 0 : m_x.get());
  }
  void compute_fast() {
    ++fast_calls;
    m_fast = m_slow * m_x.get();
  }

  int slow_calls{0}, fast_calls{0};

private:
  value_type &m_slow, &m_slow_extra, &m_fast, &m_plain;
  nm::history_property<value_type> &m_old;
  value_type const &m_factor;
  nm::input_property<value_type, nm::property_traits> const &m_source, &m_x;
};

/// One callback filling "first", "second" and the optional "third" (a null pointer otherwise) from "source";
/// "constant" declares the group, "taken" gives "first" its own callback before, "empty" adds a group of none,
/// "again" puts "second" into a second group.
template <typename Traits>
class grouped final : public nm::material_base<grouped<Traits>, Traits> {
public:
  using base = nm::material_base<grouped<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit grouped(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_first(base::template add_output<value_type>(
            "first", base::template get_parameter<bool>("taken") ? &grouped::compute : nullptr)),
        m_second({&base::template add_output<value_type>("second")}),
        m_source(base::template add_input<value_type>(base::template get_parameter<std::string>("source"), "value",
                                                      nm::EdgeKind::Global)) {
    if (base::template get_parameter<bool>("third"))
      m_third = &base::template add_output<value_type>("third");
    if (base::template get_parameter<bool>("empty"))
      base::add_callback(&grouped::compute);
    auto group{base::add_callback(&grouped::compute, m_first, m_second, m_third)};
    if (base::template get_parameter<bool>("constant"))
      group.reads(&m_source, decltype(&m_source){});
    if (base::template get_parameter<bool>("again"))
      base::add_callback(&grouped::compute, m_second);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<std::string>(para, "source");
    nm::optional<bool>(para, "third", false);
    nm::optional<bool>(para, "constant", true);
    nm::optional<bool>(para, "taken", false);
    nm::optional<bool>(para, "empty", false);
    nm::optional<bool>(para, "again", false);
    return para;
  }

  void compute() {
    ++calls;
    m_first = m_source.get();
    *m_second[0] = 2 * m_source.get();
    if (m_third)
      *m_third = 3 * m_source.get();
  }

  int calls{0};

  /// Whether an output of type U can be declared constant, alone or in a group.
  template <typename U>
  static constexpr bool declarable{requires(grouped &m, U const &u) { m.constant_output(u); }};
  template <typename U>
  static constexpr bool groupable{requires(grouped &m, U const &u) { m.add_callback(&grouped::compute, u); }};
  template <typename U>
  static constexpr bool group_declarable{
      requires(grouped &m, U const &u) { m.add_callback(&grouped::compute, u).constant(); }};

private:
  value_type &m_first;
  std::vector<value_type *> m_second;
  value_type *m_third{nullptr};
  nm::input_property<value_type, nm::property_traits> const &m_source;
};

/// Neither arithmetic, nor a tensor, nor comparable with ==.
struct opaque {
  int bits;
};

counter<policy> &make(ctx_type &ctx, std::string const &name, param_type p = {}) {
  p.insert<std::string>("name", name);
  return ctx.create<counter<policy>>(p);
}

param_type reading(std::string const &source, std::string const &property = "value") {
  param_type p;
  p.insert<std::string>("source", source);
  p.insert<std::string>("property", property);
  return p;
}

T value_of(ctx_type &ctx, std::string const &name) { return ctx.get<T>(name, "value"); }

} // namespace

TEST(ConstantProperties, AConstantChainIsComputedOnce) {
  ctx_type ctx;
  ctx.verify_constants(false); // counts the calls
  param_type p;
  p.insert<T>("factor", 2.0);
  auto &a{make(ctx, "a", p)};
  auto p_b{reading("a")};
  p_b.insert<T>("factor", 3.0);
  auto &b{make(ctx, "b", p_b)};
  ctx.finalize();
  for (int i{0}; i < 5; ++i)
    ctx.update();
  EXPECT_EQ(a.calls, 1);
  EXPECT_EQ(b.calls, 1);
  EXPECT_EQ(value_of(ctx, "b"), 6.0);
}

TEST(ConstantProperties, AReadOfAVaryingPropertyKeepsTheOutputInTheLoop) {
  ctx_type ctx;
  ctx.verify_constants(false); // counts the calls
  param_type p;
  p.insert<std::string>("name", "x");
  auto &x{ctx.create<nm::external_scalar_source<policy>>(p)};
  auto &b{make(ctx, "b", reading("x", "state"))};
  ctx.finalize();
  for (int i{0}; i < 5; ++i) {
    x.bind(0.0, static_cast<T>(i));
    ctx.update();
    EXPECT_EQ(value_of(ctx, "b"), static_cast<T>(i));
  }
  EXPECT_EQ(b.calls, 5);
}

TEST(ConstantProperties, AnUndeclaredOutputIsNeverConstant) {
  ctx_type ctx;
  ctx.verify_constants(false); // counts the calls
  param_type p;
  p.insert<bool>("constant", false);
  auto &a{make(ctx, "a", p)};
  auto &b{make(ctx, "b", reading("a"))};
  ctx.finalize();
  for (int i{0}; i < 3; ++i)
    ctx.update();
  EXPECT_EQ(a.calls, 3);
  EXPECT_EQ(b.calls, 3); // its read varies
}

TEST(ConstantProperties, ALocalEdgeIsNeverConstant) {
  ctx_type ctx;
  ctx.verify_constants(false); // counts the calls
  make(ctx, "a");
  auto p{reading("a")};
  p.insert<bool>("local", true);
  auto &b{make(ctx, "b", p)};
  ctx.finalize();
  for (int i{0}; i < 3; ++i)
    ctx.update();
  EXPECT_EQ(b.calls, 3);
}

TEST(ConstantProperties, ACallbackRunsWhileAnOutputItWritesVaries) {
  ctx_type ctx;
  ctx.verify_constants(false); // counts the calls
  param_type p;
  p.insert<bool>("extra", true); // "extra" is written by the same callback but not declared
  auto &a{make(ctx, "a", p)};
  ctx.finalize();
  for (int i{0}; i < 3; ++i)
    ctx.update();
  EXPECT_EQ(a.calls, 3);
}

TEST(ConstantProperties, SetParameterRecomputesTheConstants) {
  ctx_type ctx;
  ctx.verify_constants(false); // counts the calls
  auto &a{make(ctx, "a")};
  auto &b{make(ctx, "b", reading("a"))};
  ctx.finalize();
  ctx.update();
  ctx.update();
  a.set_parameter<T>("factor", 4.0);
  ctx.update();
  EXPECT_EQ(value_of(ctx, "b"), 4.0);
  EXPECT_EQ(a.calls, 2);
  EXPECT_EQ(b.calls, 2);
  ctx.update();
  EXPECT_EQ(a.calls, 2);
}

TEST(ConstantProperties, AMutableWriteRecomputesTheConstants) {
  ctx_type ctx;
  ctx.verify_constants(false); // counts the calls
  auto &a{make(ctx, "a")};
  auto &b{make(ctx, "b", reading("a"))};
  ctx.finalize();
  ctx.update();
  ctx.get_mutable<T>("a", "value") = 7.0; // bypasses the graph
  ctx.update();
  EXPECT_EQ(b.calls, 2);
  EXPECT_EQ(a.calls, 2); // recomputed as well: the write is not trusted over the producer
}

TEST(ConstantProperties, UpdatePropertyComputesAConstantNotYetEvaluated) {
  ctx_type ctx;
  ctx.verify_constants(false); // counts the calls
  auto &a{make(ctx, "a")};
  auto p{reading("a")};
  p.insert<T>("factor", 5.0);
  auto &b{make(ctx, "b", p)};
  ctx.finalize();
  ctx.update_property("b", "value");
  EXPECT_EQ(value_of(ctx, "b"), 5.0);
  ctx.update();
  ctx.update_property("b", "value");
  EXPECT_EQ(a.calls, 1);
  EXPECT_EQ(b.calls, 1);
}

TEST(ConstantProperties, AllInputsAreReadByDefault) {
  ctx_type ctx;
  ctx.verify_constants(false);
  param_type p;
  p.insert<std::string>("name", "x");
  auto &x{ctx.create<nm::external_scalar_source<policy>>(p)};
  make(ctx, "a");
  auto p_b{reading("a")};
  p_b.insert<std::string>("other", "x"); // not used by "value", but an input
  auto &b{make(ctx, "b", p_b)};
  auto p_c{p_b};
  p_c.insert<std::string>("name", "c");
  p_c.insert<bool>("narrow", true);
  auto &c{ctx.create<counter<policy>>(p_c)};
  ctx.finalize();
  for (int i{0}; i < 3; ++i) {
    x.bind(0.0, static_cast<T>(i));
    ctx.update();
  }
  EXPECT_EQ(b.calls, 3);
  EXPECT_EQ(c.calls, 1);
}

TEST(ConstantProperties, OnlyAnOutputCanBeDeclared) {
  ctx_type ctx;
  param_type p;
  p.insert<bool>("bogus", true);
  EXPECT_THROW(make(ctx, "a", p), std::logic_error);
}

TEST(ConstantProperties, OnlyAnInputCanBeADeclaredRead) {
  ctx_type ctx;
  param_type p;
  p.insert<bool>("foreign", true);
  EXPECT_THROW(make(ctx, "a", p), std::logic_error);
}

TEST(ConstantProperties, VerificationCatchesHiddenState) {
  ctx_type ctx;
  ctx.verify_constants(true);
  param_type p;
  p.insert<bool>("hidden", true);
  make(ctx, "a", p);
  ctx.finalize();
  ctx.update();
  EXPECT_THROW(ctx.update(), std::logic_error);
}

TEST(ConstantProperties, VerificationPassesATrueConstant) {
  ctx_type ctx;
  ctx.verify_constants(true);
  auto &a{make(ctx, "a")};
  make(ctx, "b", reading("a"));
  ctx.finalize();
  for (int i{0}; i < 3; ++i)
    ctx.update();
  EXPECT_EQ(a.calls, 3); // the first run and two verifying reruns
  EXPECT_EQ(value_of(ctx, "b"), 1.0);
}

namespace {

struct two_callbacks_setup {
  ctx_type ctx;
  nm::external_scalar_source<policy> *x{nullptr};
  two_callbacks<policy> *m{nullptr};

  explicit two_callbacks_setup(param_type p = {}) {
    ctx.verify_constants(false); // counts the calls
    param_type q;
    q.insert<std::string>("name", "x");
    x = &ctx.create<nm::external_scalar_source<policy>>(q);
    make(ctx, "a");
    p.insert<std::string>("name", "m");
    p.insert<std::string>("source", "a");
    p.insert<std::string>("x", "x");
    m = &ctx.create<two_callbacks<policy>>(p);
  }

  void run(int n) {
    for (int i{1}; i <= n; ++i) {
      x->bind(0.0, static_cast<T>(i));
      ctx.update();
      EXPECT_EQ(ctx.get<T>("m", "fast"), 2.0 * i);
    }
  }
};

} // namespace

TEST(ConstantProperties, CallbacksOfOneMaterialAreSkippedOneByOne) {
  two_callbacks_setup s;
  s.ctx.finalize();
  s.run(4);
  EXPECT_EQ(s.m->slow_calls, 1);
  EXPECT_EQ(s.m->fast_calls, 4);
}

TEST(ConstantProperties, AnOutputWithoutAWriterKeepsTheMaterialsCallbacksRunning) {
  param_type p;
  p.insert<bool>("attribute", false);
  p.insert<bool>("extra_constant", false);
  two_callbacks_setup s{p};
  s.ctx.finalize();
  s.run(3);
  EXPECT_EQ(s.m->slow_calls, 3); // "slow_extra" may be written by any callback
}

TEST(ConstantProperties, AVaryingWrittenOutputKeepsItsWriterRunning) {
  param_type p;
  p.insert<bool>("extra_constant", false);
  two_callbacks_setup s{p};
  s.ctx.finalize();
  s.run(3);
  EXPECT_EQ(s.m->slow_calls, 3);
  EXPECT_EQ(s.ctx.get<T>("m", "slow_extra"), 2.0 + 3.0);
}

TEST(ConstantProperties, CallbacksAndTheOutputsTheyWriteAreOrdered) {
  two_callbacks_setup s;
  s.ctx.finalize();
  auto const &order{s.ctx.property_execution_order()};
  auto const at{[&](std::string const &name) {
    return std::ranges::find(order, s.ctx.find_property("m", name)) - order.begin();
  }};
  EXPECT_LT(at("slow"), at("fast"));
  EXPECT_LT(at("slow"), at("slow_extra")); // after its writer
  EXPECT_LT(at("fast"), at("old"));
  EXPECT_LT(at("slow"), at("plain")); // no writer declared: after every callback of the material
  EXPECT_LT(at("fast"), at("plain"));
}

TEST(ConstantProperties, ACycleOfRunsAfterIsRejected) {
  param_type p;
  p.insert<bool>("cycle", true);
  two_callbacks_setup s{p};
  EXPECT_THAT_MESSAGE(s.ctx.finalize(), std::runtime_error, "runs_after");
}

TEST(ConstantProperties, AWriterNeedsACallback) {
  param_type p;
  p.insert<bool>("orphan", true);
  EXPECT_THROW(two_callbacks_setup{p}, std::logic_error);
}

TEST(ConstantProperties, AHistoryOutputCannotBeConstant) {
  static_assert(!grouped<policy>::declarable<nm::history_property<T>>);
  static_assert(!grouped<policy>::group_declarable<nm::history_property<T>>);
}

namespace {

grouped<policy> &make_grouped(ctx_type &ctx, param_type p) {
  p.insert<std::string>("name", "g");
  return ctx.create<grouped<policy>>(p);
}

} // namespace

TEST(ConstantProperties, ACallbackFillsAGroupOfOutputs) {
  ctx_type ctx;
  ctx.verify_constants(false); // counts the calls
  make(ctx, "a");
  param_type p;
  p.insert<std::string>("source", "a");
  p.insert<bool>("third", true);
  auto &g{make_grouped(ctx, p)};
  ctx.finalize();
  auto const &order{ctx.property_execution_order()};
  auto const at{[&](std::string const &name) { return std::ranges::find(order, ctx.find_property("g", name)) - order.begin(); }};
  EXPECT_LT(at("first"), at("second"));
  EXPECT_LT(at("first"), at("third"));
  for (int i{0}; i < 3; ++i)
    ctx.update();
  EXPECT_EQ(g.calls, 1); // a constant group runs once
  EXPECT_EQ(ctx.get<T>("g", "third"), 3.0);
}

TEST(ConstantProperties, ANullOutputIsLeftOutOfTheGroup) {
  ctx_type ctx;
  make(ctx, "a");
  param_type p;
  p.insert<std::string>("source", "a");
  make_grouped(ctx, p); // "third" is a null pointer
  ctx.finalize();
  ctx.update();
  EXPECT_EQ(ctx.find_property("g", "third"), nullptr);
  EXPECT_EQ(ctx.get<T>("g", "second"), 2.0);
}

TEST(ConstantProperties, AnOutputWithACallbackCannotJoinAGroup) {
  ctx_type ctx;
  make(ctx, "a");
  param_type p;
  p.insert<std::string>("source", "a");
  p.insert<bool>("taken", true);
  EXPECT_THROW(make_grouped(ctx, p), std::logic_error);
}

TEST(ConstantProperties, AGroupNeedsAnOutput) {
  ctx_type ctx;
  make(ctx, "a");
  param_type p;
  p.insert<std::string>("source", "a");
  p.insert<bool>("empty", true);
  EXPECT_THROW(make_grouped(ctx, p), std::logic_error);
}

// debug builds compare a rerun constant with the stored value: a type that cannot be compared is a compile error
TEST(ConstantProperties, OnlyComparableTypesCanBeConstant) {
  using g = grouped<policy>;
  static_assert(g::declarable<T> && g::declarable<tmech::tensor<T, 3, 2>> && g::declarable<tmech::tensor<T, 3, 4>>);
  static_assert(g::declarable<std::string>);
  static_assert(!g::declarable<opaque>);
  static_assert(g::group_declarable<T *> && g::group_declarable<std::vector<tmech::tensor<T, 3, 4> *>>);
  static_assert(!g::group_declarable<opaque> && !g::group_declarable<opaque *>);
  static_assert(!g::group_declarable<std::vector<opaque *>>);
  static_assert(g::groupable<opaque> && g::groupable<std::vector<opaque *>>); // a group that is not constant
}

TEST(ConstantProperties, VerificationToleranceIsRelative) {
  using nm::detail::same_value;
  EXPECT_TRUE(same_value(T{0}, T{0}));
  EXPECT_TRUE(same_value(T{1e5}, T{1e5} * (1 + 1e-12)));
  EXPECT_TRUE(same_value(T{1e-5}, T{1e-5} * (1 + 1e-12)));
  EXPECT_FALSE(same_value(T{1e-5}, T{1e-5} * (1 + 1e-3))); // a compliance in 1/MPa
  tmech::tensor<T, 3, 4> S, S_changed;
  S(0, 0, 0, 0) = S_changed(0, 0, 0, 0) = 1e-5;
  S_changed(0, 0, 0, 0) *= 1 + 1e-3;
  EXPECT_FALSE(same_value(S, S_changed));
}

TEST(ConstantProperties, VerificationReportsANonFiniteConstant) {
  ctx_type ctx;
  ctx.verify_constants(true);
  param_type p;
  p.insert<T>("factor", std::numeric_limits<T>::quiet_NaN());
  make(ctx, "a", p);
  ctx.finalize();
  ctx.update();
  EXPECT_THAT_MESSAGE(ctx.update(), std::logic_error, "not finite");
}

// ties are broken by creation order: the order does not depend on hashing
TEST(ConstantProperties, IndependentPropertiesRunInCreationOrder) {
  ctx_type ctx;
  std::vector<std::string> const names{"d", "b", "e", "a", "c", "g", "f"};
  for (auto const &name : names)
    make(ctx, name);
  ctx.finalize();
  std::vector<std::string> owners;
  for (auto const *prop : ctx.property_execution_order())
    owners.push_back(prop->traits().id.owner);
  EXPECT_EQ(owners, names);
}

TEST(ConstantProperties, OnlyCallbacksCanBeOrdered) {
  param_type p;
  p.insert<bool>("loose", true);
  EXPECT_THROW(two_callbacks_setup{p}, std::logic_error);
}

TEST(ConstantProperties, AnOutputHasOneWriter) {
  {
    param_type p;
    p.insert<bool>("rewrite", true);
    EXPECT_THROW(two_callbacks_setup{p}, std::logic_error);
  }
  ctx_type ctx;
  make(ctx, "a");
  param_type p;
  p.insert<std::string>("source", "a");
  p.insert<bool>("again", true);
  EXPECT_THROW(make_grouped(ctx, p), std::logic_error);
}

TEST(ConstantProperties, NoDeclarationAfterFinalize) {
  ctx_type ctx;
  param_type p;
  p.insert<bool>("late", true);
  make(ctx, "a", p);
  ctx.finalize();
  EXPECT_THROW(ctx.update(), std::logic_error);
}

TEST(ConstantProperties, UpdatePropertyAfterSetParameterRecomputesOnce) {
  ctx_type ctx;
  ctx.verify_constants(false); // counts the calls
  auto &a{make(ctx, "a")};
  auto &b{make(ctx, "b", reading("a"))};
  ctx.finalize();
  ctx.update();
  a.set_parameter<T>("factor", 4.0);
  ctx.update_property("b", "value");
  EXPECT_EQ(value_of(ctx, "b"), 4.0);
  ctx.update();
  ctx.update();
  EXPECT_EQ(a.calls, 2);
  EXPECT_EQ(b.calls, 2);
}

TEST(ConstantProperties, AConstantComputedByUpdatePropertyIsVerified) {
  ctx_type ctx;
  ctx.verify_constants(true);
  param_type p;
  p.insert<bool>("hidden", true);
  make(ctx, "a", p);
  ctx.finalize();
  ctx.update_property("a", "value");
  EXPECT_THROW(ctx.update(), std::logic_error);
}

namespace {

/// A thermoelastic bar declaring its reads per callback: "stress" and "energy" = E (strain - thermal_strain),
/// "stiffness" = E, "thermal_strain" = alpha (temperature - 20), "thermal_stress" = stiffness * thermal_strain.
/// "stress" is created first, so only its reads order it after "stiffness" and "thermal_strain". "foreign"
/// reads a member that is no output, "self" a callback its own output, "twice" declares a read set twice,
/// "unbound" reads an output nothing writes, "sloppy" reads the strain in "thermal_strain" undeclared.
template <typename Traits>
class bar final : public nm::material_base<bar<Traits>, Traits> {
public:
  using base = nm::material_base<bar<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit bar(Args &&...args)
      : base(std::forward<Args>(args)...), m_E(base::template get_parameter<value_type>("E")),
        m_alpha(base::template get_parameter<value_type>("alpha")),
        m_strain(base::template add_input<value_type>("strain", "state", nm::EdgeKind::Global)),
        m_temperature(base::template add_input<value_type>(base::template get_parameter<std::string>("temperature"),
                                                           base::template get_parameter<std::string>("property"),
                                                           nm::EdgeKind::Global)),
        m_stress(base::template add_output<value_type>("stress")), m_energy(base::template add_output<value_type>("energy")),
        m_stiffness(base::template add_output<value_type>("stiffness")),
        m_thermal(base::template add_output<value_type>("thermal_strain")),
        m_thermal_stress(base::template add_output<value_type>("thermal_stress")),
        m_plain(base::template add_output<value_type>("plain")) {
    auto const option{[this](char const *key) { return base::template get_parameter<bool>(key); }};
    base::constant_output(m_plain, {}); // set here, written by no callback
    base::add_callback(&bar::compute_stress, m_stress, m_energy).reads(m_strain, m_stiffness, &m_thermal);
    base::add_callback(&bar::compute_stiffness, m_stiffness).reads(option("self") ? &m_stiffness : nullptr);
    auto thermal{base::add_callback(&bar::compute_thermal, m_thermal)};
    thermal.reads(std::vector<nm::input_wire_base const *>{&m_temperature, nullptr});
    if (option("twice"))
      thermal.reads(m_temperature);
    auto thermal_stress{base::add_callback(&bar::compute_thermal_stress, m_thermal_stress)};
    if (option("foreign"))
      thermal_stress.reads(m_E);
    else if (option("unbound"))
      thermal_stress.reads(m_plain);
    else
      thermal_stress.reads(m_stiffness, m_thermal);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<value_type>(para, "E", value_type{200});
    nm::optional<value_type>(para, "alpha", value_type{1e-2});
    nm::optional<std::string>(para, "temperature", std::string{"temperature"});
    nm::optional<std::string>(para, "property", std::string{"value"});
    for (auto const *key : {"foreign", "self", "twice", "unbound", "sloppy"})
      nm::optional<bool>(para, key, false);
    return para;
  }

  void compute_stress() {
    ++stress_calls;
    m_stress = m_stiffness * (m_strain.get() - m_thermal);
    m_energy = m_stress * m_stress / (2 * m_stiffness);
  }
  void compute_stiffness() {
    ++stiffness_calls;
    m_stiffness = m_E;
  }
  void compute_thermal() {
    ++thermal_calls;
    m_thermal = m_alpha * (m_temperature.get() - value_type{20}) +
                (base::template get_parameter<bool>("sloppy") ? m_strain.get() : value_type{0});
  }
  void compute_thermal_stress() {
    ++thermal_stress_calls;
    m_thermal_stress = m_stiffness * m_thermal;
  }

  int stress_calls{0}, stiffness_calls{0}, thermal_calls{0}, thermal_stress_calls{0};

private:
  value_type const &m_E, &m_alpha;
  nm::input_property<value_type, nm::property_traits> const &m_strain, &m_temperature;
  value_type &m_stress, &m_energy, &m_stiffness, &m_thermal, &m_thermal_stress, &m_plain;
};

/// "temperature" (constant 70 or the varying "heat"), "strain" (varying) and "bar".
struct bar_setup {
  ctx_type ctx;
  nm::external_scalar_source<policy> *strain{nullptr}, *heat{nullptr};
  bar<policy> *b{nullptr};

  explicit bar_setup(param_type p = {}, bool verify = false) {
    ctx.verify_constants(verify);
    param_type s;
    s.insert<std::string>("name", "temperature");
    s.insert<T>("factor", 70.0);
    ctx.create<counter<policy>>(s);
    s.clear();
    s.insert<std::string>("name", "heat");
    heat = &ctx.create<nm::external_scalar_source<policy>>(s);
    s.clear();
    s.insert<std::string>("name", "strain");
    strain = &ctx.create<nm::external_scalar_source<policy>>(s);
    p.insert<std::string>("name", "bar");
    b = &ctx.create<bar<policy>>(p);
    ctx.finalize();
  }
  void step(T eps, T temperature = 70.0) {
    strain->bind(0.0, eps);
    heat->bind(0.0, temperature);
    ctx.update();
  }
  T get(std::string const &property) { return ctx.get<T>("bar", property); }
};

param_type heated() {
  param_type p;
  p.insert<std::string>("temperature", "heat");
  p.insert<std::string>("property", "state");
  return p;
}

param_type with(char const *key) {
  param_type p;
  p.insert<bool>(key, true);
  return p;
}

} // namespace

TEST(DeclaredReads, ConstancyIsInferredFromTheReads) {
  bar_setup s;
  for (int i{1}; i <= 5; ++i)
    s.step(0.1 * i);
  EXPECT_EQ(s.b->stiffness_calls, 1);
  EXPECT_EQ(s.b->thermal_calls, 1);
  EXPECT_EQ(s.b->thermal_stress_calls, 1); // reads constant own outputs only
  EXPECT_EQ(s.b->stress_calls, 5);
  EXPECT_DOUBLE_EQ(s.get("stress"), 200 * 0.5 - 200 * 0.5);
  EXPECT_DOUBLE_EQ(s.get("thermal_stress"), 100.0);
}

TEST(DeclaredReads, AVaryingReadKeepsTheCallbackAndItsReadersRunning) {
  bar_setup s{heated()};
  for (int i{1}; i <= 5; ++i)
    s.step(0.1, 20.0 + i);
  EXPECT_EQ(s.b->stiffness_calls, 1);
  EXPECT_EQ(s.b->thermal_calls, 5);
  EXPECT_EQ(s.b->thermal_stress_calls, 5); // reads the varying thermal strain
  EXPECT_DOUBLE_EQ(s.get("thermal_stress"), 200 * 0.05);
}

TEST(DeclaredReads, ReadingAnOwnOutputOrdersTheCallbacks) {
  bar_setup s;
  s.step(0.75); // "stress" was created first: right on the first update only when it runs last
  EXPECT_DOUBLE_EQ(s.get("stress"), 200 * (0.75 - 0.5));
  EXPECT_DOUBLE_EQ(s.get("energy"), 50.0 * 50.0 / 400);
}

TEST(DeclaredReads, OnlyOutputsOfTheMaterialCanBeRead) {
  EXPECT_THAT_MESSAGE(bar_setup{with("foreign")}, std::logic_error, "not one of its outputs");
}

TEST(DeclaredReads, ACallbackCannotReadItsOwnOutput) {
  EXPECT_THAT_MESSAGE(bar_setup{with("self")}, std::logic_error, "its own output");
}

TEST(DeclaredReads, ReadsAreDeclaredOnce) {
  EXPECT_THAT_MESSAGE(bar_setup{with("twice")}, std::logic_error, "twice");
}

TEST(DeclaredReads, AReadOutputNeedsAWriter) {
  EXPECT_THAT_MESSAGE(bar_setup{with("unbound")}, std::logic_error, "no callback");
}

TEST(DeclaredReads, VerificationCatchesAnUndeclaredRead) {
  bar_setup s{with("sloppy"), true};
  s.ctx.check_reads(false); // the verification alone; test_read_check covers the check
  s.step(0.1);
  EXPECT_THAT_MESSAGE(s.step(0.2), std::logic_error, "thermal_strain");
}

namespace {

/// Two independent callbacks with declared reads: "a" = 2 x, "b" = 3 y.
template <typename Traits>
class pair final : public nm::material_base<pair<Traits>, Traits> {
public:
  using base = nm::material_base<pair<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit pair(Args &&...args)
      : base(std::forward<Args>(args)...), m_a(base::template add_output<value_type>("a")),
        m_b(base::template add_output<value_type>("b")),
        m_x(base::template add_input<value_type>(base::template get_parameter<std::string>("x"), "value")),
        m_y(base::template add_input<value_type>(base::template get_parameter<std::string>("y"), "value")) {
    base::add_callback(&pair::compute_a, m_a).reads(m_x);
    base::add_callback(&pair::compute_b, m_b).reads(m_y);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<std::string>(para, "x");
    nm::required<std::string>(para, "y");
    return para;
  }

  void compute_a() { m_a = 2 * m_x.get(); }
  void compute_b() { m_b = 3 * m_y.get(); }

private:
  value_type &m_a, &m_b;
  nm::input_property<value_type, nm::property_traits> const &m_x, &m_y;
};

pair<policy> &make_pair(ctx_type &ctx, std::string const &x, std::string const &y) {
  param_type p;
  p.insert<std::string>("name", "pair");
  p.insert<std::string>("x", x);
  p.insert<std::string>("y", y);
  return ctx.create<pair<policy>>(p);
}

} // namespace

TEST(CallbackEdges, UpdatingOneOutputRunsOnlyWhatItsCallbackReads) {
  ctx_type ctx;
  ctx.verify_constants(false); // counts the calls
  auto &x{make(ctx, "x")};
  auto &y{make(ctx, "y")};
  make_pair(ctx, "x", "y");
  ctx.finalize();
  ctx.update_property("pair", "a");
  EXPECT_EQ(x.calls, 1);
  EXPECT_EQ(y.calls, 0);
  EXPECT_EQ(ctx.get<T>("pair", "a"), 2.0);
}

TEST(CallbackEdges, CallbacksOfOneMaterialFormNoFalseCycle) {
  ctx_type ctx;
  make(ctx, "y");
  make_pair(ctx, "n", "y"); // "a" reads n, n reads "b": no cycle, "b" does not read n
  make(ctx, "n", reading("pair", "b"));
  ASSERT_NO_THROW(ctx.finalize());
  ctx.update();
  EXPECT_EQ(ctx.get<T>("pair", "a"), 2.0 * 3.0);
}

namespace {
/// "count" = the call count of the counter named by "target", reached through a material_ref, not an input.
template <typename Traits>
class referrer final : public nm::material_base<referrer<Traits>, Traits> {
public:
  using base = nm::material_base<referrer<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit referrer(Args &&...args)
      : base(std::forward<Args>(args)...), m_count(base::template add_output<value_type>("count")),
        m_target(base::template add_material_ref<counter<Traits>>(base::template get_parameter<std::string>("target"))) {
    base::add_callback(&referrer::compute, m_count).reads();
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<std::string>(para, "target");
    return para;
  }

  void compute() { m_count = m_target.get().calls; }

private:
  value_type &m_count;
  nm::material_ref<counter<Traits>, Traits> &m_target;
};

} // namespace

TEST(HiddenDependencies, AMaterialReachedByReferenceKeepsItsReaderVarying) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "r");
  p.insert<std::string>("target", "b");
  ctx.create<referrer<policy>>(p); // created first: only an edge runs it after b's source
  p.clear();
  p.insert<std::string>("name", "x");
  auto &x{ctx.create<nm::external_scalar_source<policy>>(p)};
  make(ctx, "b", reading("x", "state"));
  ctx.finalize();
  x.bind(0.0, 1.0);
  ctx.update();
  EXPECT_FALSE(ctx.is_constant("r", "count"));
}
