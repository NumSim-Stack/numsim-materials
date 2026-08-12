#include <gtest/gtest.h>
#include <array>
#include <string>
#include <numsim-core/parameter_handler.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"

namespace {

namespace nm = numsim::materials;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

/// A parameter large enough that std::any cannot use its small buffer. Storing
/// one is what separates a correct write API from one that merely appears to
/// work with scalars.
struct big_parameter {
  std::array<double, 32> data{};
};

/// Binds references to its parameters exactly as every real material does, and
/// exposes them so a test can observe what a bound reference sees after a write.
template <typename Traits>
class probe_material final
    : public nm::material_base<probe_material<Traits>, Traits> {
public:
  using base = nm::material_base<probe_material<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit probe_material(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_k(base::template get_parameter<value_type>("K")),
        m_big(base::template get_parameter<big_parameter>("big")) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<value_type>("K").template add<nm::is_required>();
    para.template insert<big_parameter>("big").template add<nm::is_required>();
    return para;
  }

  const value_type& bound_k() const noexcept { return m_k; }
  const big_parameter& bound_big() const noexcept { return m_big; }

private:
  const value_type& m_k;
  const big_parameter& m_big;
};

probe_material<policy>& build(ctx_type& ctx) {
  param_type p;
  p.insert<std::string>("name", "probe");
  p.insert<T>("K", 100.0);
  p.insert<big_parameter>("big", big_parameter{});
  auto& m = ctx.create<probe_material<policy>>(p);
  ctx.finalize();
  return m;
}

// ---------------------------------------------------------------------------

/// A material binds `const T&` into the parameter handler at construction, so
/// the only useful write is one those references observe.
TEST(SetParameter, BoundReferenceSeesTheNewValue) {
  ctx_type ctx;
  auto& m = build(ctx);

  EXPECT_DOUBLE_EQ(m.bound_k(), 100.0);
  m.template set_parameter<T>("K", 250.0);
  EXPECT_DOUBLE_EQ(m.bound_k(), 250.0);
}

/// The address must not move, for any type. This is the reason set_parameter
/// assigns through the non-const get<T>() instead of calling insert():
/// insert() goes through insert_or_assign, which replaces the whole std::any.
/// For a value past std::any's small buffer that relocates the object and every
/// bound reference dangles — while a `double` would keep the same address by
/// accident and pass a test written only against scalars.
TEST(SetParameter, WriteDoesNotRelocateEvenForALargeType) {
  ctx_type ctx;
  auto& m = build(ctx);

  const void* addr_before = static_cast<const void*>(&m.bound_big());
  big_parameter v;
  v.data[0] = 7.0;
  v.data[31] = 9.0;
  m.template set_parameter<big_parameter>("big", v);

  EXPECT_EQ(static_cast<const void*>(&m.bound_big()), addr_before)
      << "the bound reference must remain valid";
  EXPECT_DOUBLE_EQ(m.bound_big().data[0], 7.0);
  EXPECT_DOUBLE_EQ(m.bound_big().data[31], 9.0);
}

/// The same guarantee, stated directly against parameter_handler so the reason
/// is documented where the mechanism lives rather than only in a material.
TEST(SetParameter, InsertRelocatesALargeValueButAssignmentDoesNot) {
  numsim_core::parameter_handler<> h;
  h.insert<big_parameter>("big", big_parameter{});
  const void* bound = static_cast<const void*>(&h.get<big_parameter>("big"));

  h.insert<big_parameter>("big", big_parameter{});
  const void* after_insert =
      static_cast<const void*>(&h.get<big_parameter>("big"));

  h.get<big_parameter>("big").data[0] = 1.0;
  const void* after_assign =
      static_cast<const void*>(&h.get<big_parameter>("big"));

  EXPECT_NE(after_insert, bound)
      << "insert() is expected to relocate; if this ever stops being true the "
         "rationale for set_parameter's implementation needs revisiting";
  EXPECT_EQ(after_assign, after_insert)
      << "assignment through get<T>() must never relocate";
}

TEST(SetParameter, ThrowsForAnUnknownKey) {
  ctx_type ctx;
  auto& m = build(ctx);
  EXPECT_THROW(m.template set_parameter<T>("nosuchkey", 1.0),
               std::invalid_argument);
}

/// Writing does not disturb neighbouring parameters.
TEST(SetParameter, LeavesOtherParametersAlone) {
  ctx_type ctx;
  auto& m = build(ctx);

  big_parameter v;
  v.data[5] = 3.0;
  m.template set_parameter<big_parameter>("big", v);
  m.template set_parameter<T>("K", 42.0);

  EXPECT_DOUBLE_EQ(m.bound_k(), 42.0);
  EXPECT_DOUBLE_EQ(m.bound_big().data[5], 3.0);
  EXPECT_EQ(m.name(), "probe");
}

}  // namespace
