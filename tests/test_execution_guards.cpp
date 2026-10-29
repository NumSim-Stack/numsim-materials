#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/driver/external_state_source.h"

// Rules of execution (#187): no update() from inside a callback, a nested update_property() leaves the running
// update intact, and a context whose update threw is partly updated until revert().

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

/// "value" = x::state (or 1 with "constant"), counting its runs; throws above "limit".
template <typename Traits>
class tally final : public nm::material_base<tally<Traits>, Traits> {
public:
  using base = nm::material_base<tally<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit tally(Args &&...args)
      : base(std::forward<Args>(args)...), m_value(base::template add_output<value_type>("value")),
        m_limit(base::template get_parameter<value_type>("limit")) {
    if (base::template get_parameter<bool>("constant")) {
      base::add_callback(&tally::compute, m_value).reads();
    } else {
      m_x = &base::template add_input<value_type>("x", "state");
      base::add_callback(&tally::compute, m_value).reads(m_x).stateful();
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<bool>(para, "constant", false);
    nm::optional<value_type>(para, "limit", value_type{1e30});
    return para;
  }

  void compute() {
    ++runs;
    m_value = m_x ? m_x->get() : value_type{1};
    if (m_value > m_limit)
      throw std::runtime_error("tally: above the limit");
  }

  int runs{0};

private:
  value_type &m_value;
  value_type const &m_limit;
  nm::input_property<value_type, nm::property_traits> const *m_x{nullptr};
};

/// Calls back into its context from its callback: update() ("update"), or a perturbation through get_mutable()
/// and update_property() of "probe::value", as numerical_diff_checker does ("reevaluate").
template <typename Traits>
class caller final : public nm::material_base<caller<Traits>, Traits> {
public:
  using base = nm::material_base<caller<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit caller(Args &&...args)
      : base(std::forward<Args>(args)...), m_out(base::template add_output<value_type>("out")),
        m_ctx(base::template get_parameter<ctx_type *>("context")), m_mode(base::template get_parameter<std::string>("mode")) {
    base::add_callback(&caller::compute, m_out).reads().stateful();
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<ctx_type *>(para, "context");
    nm::required<std::string>(para, "mode");
    return para;
  }

  void compute() {
    if (m_mode == "update") {
      m_ctx->update();
    } else {
      auto &x{m_ctx->template get_mutable<value_type>("x", "state")};
      auto const kept{x};
      x = kept + 1;
      m_ctx->update_property("probe", "value");
      m_out = m_ctx->template get<value_type>("probe", "value");
      x = kept;
      m_ctx->update_property("probe", "value");
    }
  }

private:
  value_type &m_out;
  ctx_type *const &m_ctx;
  std::string const &m_mode;
};

/// x (external), a constant "k" (retired after the first update, so the running loop is shorter than the full
/// one), "probe" reading x, the caller, then "late" reading x: created in that order.
struct setup {
  ctx_type ctx;
  nm::external_scalar_source<policy> *x{nullptr};
  tally<policy> *k{nullptr}, *probe{nullptr}, *late{nullptr};

  explicit setup(std::string const &mode, T limit = 1e30) {
    param_type p;
    p.insert<std::string>("name", "x");
    x = &ctx.create<nm::external_scalar_source<policy>>(p);
    k = &make("k", true, 1e30);
    probe = &make("probe", false, limit);
    p.clear();
    p.insert<std::string>("name", "caller");
    p.insert<ctx_type *>("context", &ctx);
    p.insert<std::string>("mode", mode);
    if (mode != "none")
      ctx.create<caller<policy>>(p);
    late = &make("late", false, 1e30);
    ctx.verify_constants(false); // counts the runs
    ctx.finalize();
  }

  tally<policy> &make(std::string const &name, bool constant, T limit) {
    param_type p;
    p.insert<std::string>("name", name);
    p.insert<bool>("constant", constant);
    p.insert<T>("limit", limit);
    return ctx.create<tally<policy>>(p);
  }
};

TEST(ExecutionGuards, UpdateFromACallbackThrows) {
  setup s{"update"};
  s.x->bind(0.0, 1.0);
  try {
    s.ctx.update();
    FAIL() << "no throw";
  } catch (std::logic_error const &e) {
    EXPECT_NE(std::string{e.what()}.find("from a callback"), std::string::npos) << e.what();
  }
}

TEST(ExecutionGuards, ANestedReevaluationLeavesTheRunningUpdateIntact) {
  setup s{"reevaluate"};
  for (int i{1}; i <= 3; ++i) {
    s.x->bind(0.0, 1.0 * i);
    s.ctx.update();
    EXPECT_EQ(s.late->runs, i) << "every callback after the nested one runs once per update";
    EXPECT_EQ(s.ctx.get<T>("caller", "out"), 1.0 * i + 1);
    EXPECT_EQ(s.ctx.get<T>("probe", "value"), 1.0 * i) << "restored after the perturbation";
  }
}

TEST(ExecutionGuards, AFailedUpdateMustBeReverted) {
  setup s{"none", 1.5};
  s.x->bind(0.0, 2.0);
  EXPECT_THROW(s.ctx.update(), std::runtime_error);
  try {
    s.ctx.update();
    FAIL() << "no throw";
  } catch (std::logic_error const &e) {
    EXPECT_NE(std::string{e.what()}.find("revert()"), std::string::npos) << e.what();
  }
  EXPECT_THROW(s.ctx.commit(), std::logic_error);
  s.ctx.revert();
  s.x->bind(0.0, 1.0);
  ASSERT_NO_THROW(s.ctx.update());
  EXPECT_EQ(s.ctx.get<T>("late", "value"), 1.0);
}

TEST(ExecutionGuards, AFailedSelectiveUpdateMustBeRevertedToo) {
  setup s{"none", 1.5};
  s.x->bind(0.0, 2.0);
  EXPECT_THROW(s.ctx.update_property("probe", "value"), std::runtime_error);
  EXPECT_THROW(s.ctx.update(), std::logic_error);
  s.ctx.revert();
  s.x->bind(0.0, 1.0);
  EXPECT_NO_THROW(s.ctx.update());
}

} // namespace
