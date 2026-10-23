#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/umat/external_state_source.h"

// A callback with declared reads throws when it reads an input it did not declare (#170).

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

/// "out" = a + b; declares the read of "a", and of "b" too unless "forget_b".
template <typename Traits>
class reader final : public nm::material_base<reader<Traits>, Traits> {
public:
  using base = nm::material_base<reader<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit reader(Args &&...args)
      : base(std::forward<Args>(args)...), m_out(base::template add_output<value_type>("out")),
        m_a(base::template add_input<value_type>("a", "state")), m_b(base::template add_input<value_type>("b", "state")) {
    auto handle{base::add_callback(&reader::compute, m_out)};
    if (base::template get_parameter<bool>("forget_b"))
      handle.reads(m_a);
    else
      handle.reads(m_a, m_b);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<bool>(para, "forget_b", false);
    return para;
  }

  void compute() { m_out = m_a.get() + m_b.get(); }

private:
  value_type &m_out;
  nm::input_property<value_type, nm::property_traits> const &m_a, &m_b;
};

/// Reads "b::state" through a history input: old_value() unless "new_value"; declares it unless "forget".
template <typename Traits>
class history_reader final : public nm::material_base<history_reader<Traits>, Traits> {
public:
  using base = nm::material_base<history_reader<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit history_reader(Args &&...args)
      : base(std::forward<Args>(args)...), m_out(base::template add_output<value_type>("out")),
        m_b(base::template add_input_history<value_type>({"b", "state"})) {
    auto handle{base::add_callback(&history_reader::compute, m_out)};
    if (base::template get_parameter<bool>("forget"))
      handle.reads();
    else
      handle.reads(m_b);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<bool>(para, "forget", false);
    nm::optional<bool>(para, "new_value", false);
    return para;
  }

  void compute() { m_out = base::template get_parameter<bool>("new_value") ? m_b.new_value() : m_b.old_value(); }

private:
  value_type &m_out;
  nm::input_history<value_type, nm::property_traits> const &m_b;
};

/// Reruns "reader::out" over a local edge, as a solver reruns its function material.
template <typename Traits>
class rerunner final : public nm::material_base<rerunner<Traits>, Traits> {
public:
  using base = nm::material_base<rerunner<Traits>, Traits>;
  using value_type = typename base::value_type;

  template <typename... Args>
  explicit rerunner(Args &&...args)
      : base(std::forward<Args>(args)...), m_out(base::template add_output<value_type>("out")),
        m_in(base::template add_input<value_type>("reader", "out", nm::EdgeKind::Local)) {
    base::add_callback(&rerunner::compute, m_out).reads(m_in);
  }

  void compute() {
    m_in.update_source();
    m_out = m_in.get();
  }

private:
  value_type &m_out;
  nm::input_property<value_type, nm::property_traits> const &m_in;
};

/// "late_input": reads "a::state" through an input added after constant(); "null_callback" binds none.
template <typename Traits>
class misdeclared final : public nm::material_base<misdeclared<Traits>, Traits> {
public:
  using base = nm::material_base<misdeclared<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit misdeclared(Args &&...args) : base(std::forward<Args>(args)...), m_out(base::template add_output<value_type>("out")) {
    if (base::template get_parameter<std::string>("mistake") == "late_input") {
      base::add_callback(&misdeclared::compute, m_out).constant();
      m_late = &base::template add_input<value_type>("a", "state");
    } else {
      base::add_callback(static_cast<void (misdeclared::*)()>(nullptr), m_out);
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<std::string>(para, "mistake");
    return para;
  }

  void compute() { m_out = m_late ? m_late->get() : 1.0; }

private:
  value_type &m_out;
  nm::input_property<value_type, nm::property_traits> const *m_late{nullptr};
};

struct setup {
  ctx_type ctx;
  explicit setup(bool forget_b, bool with_rerunner = false) {
    for (auto const *name : {"a", "b"}) {
      param_type p;
      p.insert<std::string>("name", name);
      ctx.create<nm::external_scalar_source<policy>>(p).bind(0.0, 1.0);
    }
    param_type p;
    p.insert<std::string>("name", "reader");
    p.insert<bool>("forget_b", forget_b);
    ctx.create<reader<policy>>(p);
    if (with_rerunner) {
      p.clear();
      p.insert<std::string>("name", "rerunner");
      ctx.create<rerunner<policy>>(p);
    }
    ctx.finalize();
  }
};

// The checks compiled out (NUMSIM_MATERIALS_DEBUG_CHECKS=0): nothing to test.
#define SKIP_WITHOUT_CHECKS()                                                                                 \
  if (!NUMSIM_MATERIALS_DEBUG_CHECKS)                                                                         \
  GTEST_SKIP() << "NUMSIM_MATERIALS_DEBUG_CHECKS is off"

TEST(ReadCheck, AnUndeclaredInputThrowsOnTheFirstRun) {
  SKIP_WITHOUT_CHECKS();
  setup s{true};
  try {
    s.ctx.update();
    FAIL() << "no throw";
  } catch (std::logic_error const &e) {
    std::string const what{e.what()};
    EXPECT_NE(what.find("reader::out"), std::string::npos) << what;
    EXPECT_NE(what.find("b::state"), std::string::npos) << what;
  }
}

TEST(ReadCheck, DeclaredInputsPass) {
  setup s{false};
  ASSERT_NO_THROW(s.ctx.update());
  EXPECT_EQ(s.ctx.get<T>("reader", "out"), 2.0);
}

TEST(ReadCheck, TheCheckCanBeSwitchedOff) {
  setup s{true};
  s.ctx.check_reads(false);
  EXPECT_NO_THROW(s.ctx.update());
}

TEST(ReadCheck, AThrowRestoresTheOuterScope) {
  SKIP_WITHOUT_CHECKS();
  setup s{true};
  EXPECT_THROW(s.ctx.update(), std::logic_error);
  EXPECT_EQ(nm::detail::current_reads, nullptr);
}

TEST(ReadCheck, ARerunCallbackIsCheckedAgainstItsOwnDeclaration) {
  SKIP_WITHOUT_CHECKS();
  setup s{true, true};
  try {
    s.ctx.update_property("rerunner", "out"); // the local edge leaves the reader out of the upstream
    FAIL() << "no throw";
  } catch (std::logic_error const &e) {
    std::string const what{e.what()};
    EXPECT_NE(what.find("'reader::out' reads 'b::state'"), std::string::npos) << what;
  }
}

TEST(ReadCheck, ARerunCallbackMayReadWhatItsCallerDidNotDeclare) {
  setup s{false, true};
  ASSERT_NO_THROW(s.ctx.update_property("rerunner", "out"));
  EXPECT_EQ(s.ctx.get<T>("rerunner", "out"), 2.0);
}

void history_case(bool forget, bool new_value) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "b");
  ctx.create<nm::external_scalar_source<policy>>(p).bind(0.0, 1.0);
  p.clear();
  p.insert<std::string>("name", "h");
  p.insert<bool>("forget", forget);
  p.insert<bool>("new_value", new_value);
  ctx.create<history_reader<policy>>(p);
  ctx.finalize();
  ctx.update();
}

TEST(ReadCheck, HistoryReadsAreChecked) {
  SKIP_WITHOUT_CHECKS();
  for (bool const new_value : {false, true}) {
    EXPECT_THROW(history_case(true, new_value), std::logic_error) << "new_value " << new_value;
    EXPECT_NO_THROW(history_case(false, new_value)) << "new_value " << new_value;
  }
}

void create_misdeclared(std::string const &mistake, ctx_type &ctx) {
  param_type p;
  p.insert<std::string>("name", "m");
  p.insert<std::string>("mistake", mistake);
  ctx.create<misdeclared<policy>>(p);
}

TEST(Declarations, ConstantReadsTheInputsOfFinalize) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "a");
  auto &a{ctx.create<nm::external_scalar_source<policy>>(p)};
  create_misdeclared("late_input", ctx);
  ctx.finalize();
  a.bind(0.0, 3.0);
  ASSERT_NO_THROW(ctx.update());
  EXPECT_EQ(ctx.get<T>("m", "out"), 3.0);
  EXPECT_FALSE(ctx.is_constant("m", "out"));
}

TEST(Declarations, ANullCallbackThrows) {
  ctx_type ctx;
  EXPECT_THROW(create_misdeclared("null_callback", ctx), std::logic_error);
}

} // namespace
