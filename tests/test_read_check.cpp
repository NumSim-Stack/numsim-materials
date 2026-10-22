#undef NDEBUG // the read check is compiled in debug builds only: this file checks it in every configuration
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

struct setup {
  ctx_type ctx;
  explicit setup(bool forget_b) {
    for (auto const *name : {"a", "b"}) {
      param_type p;
      p.insert<std::string>("name", name);
      ctx.create<nm::external_scalar_source<policy>>(p).bind(0.0, 1.0);
    }
    param_type p;
    p.insert<std::string>("name", "reader");
    p.insert<bool>("forget_b", forget_b);
    ctx.create<reader<policy>>(p);
    ctx.finalize();
  }
};

TEST(ReadCheck, AnUndeclaredInputThrowsOnTheFirstRun) {
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

} // namespace
