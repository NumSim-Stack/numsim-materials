/// Constant properties: an output declared a function of parameters and of reads that are constant too is
/// computed once, not on every update; a write through set_parameter() or get_mutable() recomputes it.
#include <gtest/gtest.h>
#include <string>

#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/umat/external_state_source.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

/// "value" = factor * source (factor alone without a source), counting its callback. With "constant" it
/// declares "value" constant; with "extra" it also writes an undeclared output from the same callback.
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
    if (base::template get_parameter<bool>("constant")) {
      if (m_source)
        base::constant_output("value", {m_source});
      else
        base::constant_output("value");
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
    return para;
  }

  void compute() {
    ++calls;
    m_value = m_factor * (m_source ? m_source->get() : value_type{1});
    if (m_extra)
      *m_extra = m_value;
  }

  int calls{0};

private:
  value_type &m_value;
  value_type const &m_factor;
  nm::input_property<value_type, nm::property_traits> const *m_source{nullptr};
  value_type *m_extra{nullptr};
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
