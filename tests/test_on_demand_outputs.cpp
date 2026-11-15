#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/driver/external_state_source.h"

// An output declared on_demand() is filled only while a node reads it: the callback asks is_read().

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

/// "value" = 2 x and, on demand, "extra" = x^2 (counting its evaluations). "on_demand" names the output declared
/// on demand: "extra", "value" (the first output, rejected) or "other" (an output of another callback, rejected).
template <typename Traits>
class producer final : public nm::material_base<producer<Traits>, Traits> {
public:
  using base = nm::material_base<producer<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit producer(Args &&...args)
      : base(std::forward<Args>(args)...), m_value(base::template add_output<value_type>("value")),
        m_extra(base::template add_output<value_type>("extra")), m_other(base::template add_output<value_type>("other")),
        m_x(base::template add_input<value_type>("x", "state")) {
    auto const &which{base::template get_parameter<std::string>("on_demand")};
    auto handle{base::add_callback(&producer::compute, m_value, m_extra)};
    handle.reads(m_x);
    base::add_callback(&producer::compute_other, m_other).reads(m_x);
    handle.on_demand(which == "extra" ? m_extra : which == "value" ? m_value : m_other);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<std::string>(para, "on_demand", std::string{"extra"});
    return para;
  }

  void compute() {
    m_value = 2 * m_x.get();
    if (base::is_read(m_extra)) {
      ++extra_runs;
      m_extra = m_x.get() * m_x.get();
    }
  }
  void compute_other() { m_other = m_x.get(); }

  bool value_read() const { return base::is_read(m_value); }

  int extra_runs{0};

private:
  value_type &m_value, &m_extra, &m_other;
  nm::input_property<value_type, nm::property_traits> const &m_x;
};

/// "out" = source::extra + 1.
template <typename Traits>
class reader final : public nm::material_base<reader<Traits>, Traits> {
public:
  using base = nm::material_base<reader<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit reader(Args &&...args)
      : base(std::forward<Args>(args)...), m_out(base::template add_output<value_type>("out")),
        m_extra(base::template add_input<value_type>("source", "extra")) {
    base::add_callback(&reader::compute, m_out).reads(m_extra);
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() { m_out = m_extra.get() + 1; }

private:
  value_type &m_out;
  nm::input_property<value_type, nm::property_traits> const &m_extra;
};

param_type named(std::string const &name) {
  param_type p;
  p.insert<std::string>("name", name);
  return p;
}

struct setup {
  ctx_type ctx;
  nm::external_scalar_source<policy> *x{nullptr};
  producer<policy> *source{nullptr};

  explicit setup(bool with_reader) {
    auto px{named("x")};
    x = &ctx.create<nm::external_scalar_source<policy>>(px);
    auto ps{named("source")};
    source = &ctx.create<producer<policy>>(ps);
    if (with_reader) {
      auto pr{named("reader")};
      ctx.create<reader<policy>>(pr);
    }
    ctx.finalize();
    x->bind(0.0, 3.0);
    ctx.update();
  }
};

TEST(OnDemandOutputs, AnOutputNoNodeReadsIsNotComputed) {
  setup s{false};
  EXPECT_EQ(s.source->extra_runs, 0);
  EXPECT_EQ(s.ctx.get<T>("source", "value"), 6.0);
  EXPECT_TRUE(s.source->value_read()) << "an output not declared on demand counts as read";
  try {
    s.ctx.get<T>("source", "extra");
    FAIL() << "no throw";
  } catch (std::logic_error const &e) {
    EXPECT_NE(std::string{e.what()}.find("on demand"), std::string::npos) << e.what();
  }
}

TEST(OnDemandOutputs, AnOutputANodeReadsIsComputed) {
  setup s{true};
  EXPECT_EQ(s.source->extra_runs, 1);
  EXPECT_EQ(s.ctx.get<T>("source", "extra"), 9.0);
  EXPECT_EQ(s.ctx.get<T>("reader", "out"), 10.0);
}

TEST(OnDemandOutputs, OnlyAFurtherOutputOfTheCallbackIsOnDemand) {
  for (std::string const which : {"value", "other"}) {
    ctx_type ctx;
    auto px{named("x")};
    ctx.create<nm::external_scalar_source<policy>>(px);
    auto ps{named("source")};
    ps.insert<std::string>("on_demand", which);
    EXPECT_THROW(ctx.create<producer<policy>>(ps), std::logic_error) << which;
  }
}

} // namespace
