#include <gtest/gtest.h>
#include <array>
#include <string>
#include <type_traits>
#include <numsim-core/parameter_handler.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/umat/external_state_source.h"
#include <tmech/tmech.h>

namespace {

namespace nm = numsim::materials;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;   // named: commas break the gtest macros

/// Too large for std::any's small buffer — a scalar would hide the bug.
struct big_parameter {
  std::array<double, 32> data{};
};

/// Binds parameters by reference as real materials do, and exposes them.
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

/// The only useful write is one the material's bound reference observes.
TEST(SetParameter, BoundReferenceSeesTheNewValue) {
  ctx_type ctx;
  auto& m = build(ctx);

  EXPECT_DOUBLE_EQ(m.bound_k(), 100.0);
  m.template set_parameter<T>("K", 250.0);
  EXPECT_DOUBLE_EQ(m.bound_k(), 250.0);
}

/// The address must not move, for any type — which is why set_parameter
/// assigns rather than calling insert().
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

/// The same guarantee against parameter_handler directly, so the rationale
/// lives next to the mechanism.
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
      << "insert() is expected to relocate; if that changes, revisit why "
         "set_parameter assigns instead";
  EXPECT_EQ(after_assign, after_insert)
      << "assignment through get<T>() must never relocate";
}

TEST(SetParameter, ThrowsForAnUnknownKey) {
  ctx_type ctx;
  auto& m = build(ctx);
  EXPECT_THROW(m.template set_parameter<T>("nosuchkey", 1.0),
               std::invalid_argument);
}

/// No collateral damage to neighbouring parameters.
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


// ---------------------------------------------------------------------------
// Against SHIPPED materials
//
// The probe above binds every parameter by reference and derives nothing, which
// is the one shape this API handles cleanly. These pin what happens with real
// materials, where it does not.
// ---------------------------------------------------------------------------

/// linear_elasticity computes its tangent in the constructor and registers
/// "tangent" with no callback, so a write to K lands in the parameter and
/// changes nothing. Pinned, not endorsed: this is the limitation the API
/// documents, and the reason isotropic_tangent's "recompute" exists.
TEST(SetParameterShippedMaterials, WriteLandsButDerivedStateGoesStale) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "strain_in");
  auto& src = ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "strain_in");
  p.insert<T>("K", 100.0);
  p.insert<T>("G", 40.0);
  auto& el = ctx.create<nm::linear_elasticity<policy>>(p);
  ctx.finalize();

  tensor2 eps;
  eps.fill(0.0);
  eps(0, 0) = 0.001;
  src.bind(eps, eps);
  ctx.update();
  const T before = ctx.get<tensor2>("elastic", "stress")(0, 0);
  EXPECT_NEAR(before, (100.0 + 4.0 * 40.0 / 3.0) * 0.001, 1e-12);

  el.template set_parameter<T>("K", 200.0);

  // The write is real...
  EXPECT_DOUBLE_EQ(el.template get_parameter<T>("K"), 200.0);
  // ... and has no effect, because the tangent was built once.
  ctx.update();
  EXPECT_DOUBLE_EQ(ctx.get<tensor2>("elastic", "stress")(0, 0),
                   before);
}

/// Each material holds its own COPY of the handler, so a write reaches one
/// material and not the caller's handler or any sibling built from it.
TEST(SetParameterShippedMaterials, WriteIsLocalToTheMaterial) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "probe");
  p.insert<T>("K", 100.0);
  p.insert<big_parameter>("big", big_parameter{});
  auto& m = ctx.create<probe_material<policy>>(p);
  ctx.finalize();

  m.template set_parameter<T>("K", 777.0);

  EXPECT_DOUBLE_EQ(m.bound_k(), 777.0);
  EXPECT_DOUBLE_EQ(p.get<T>("K"), 100.0)
      << "the caller's handler is a separate copy";
}

/// A wiring parameter is consumed once, at construction, to build the input.
/// Writing it afterwards cannot re-wire anything.
TEST(SetParameterShippedMaterials, WritingAWiringKeyDoesNotRewire) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "strain_in");
  auto& src = ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "other");
  ctx.create<nm::external_strain_source<policy>>(p);
  p.clear();
  p.insert<std::string>("name", "elastic");
  p.insert<std::string>("strain_producer_name", "strain_in");
  p.insert<T>("K", 100.0);
  p.insert<T>("G", 40.0);
  auto& el = ctx.create<nm::linear_elasticity<policy>>(p);
  ctx.finalize();

  el.template set_parameter<std::string>("strain_producer_name", "other");

  tensor2 eps;
  eps.fill(0.0);
  eps(0, 0) = 0.002;
  src.bind(eps, eps);   // the ORIGINAL source
  ctx.update();
  // Still reading strain_in, so the stress follows it despite the write.
  EXPECT_NEAR(ctx.get<tensor2>("elastic", "stress")(0, 0),
              (100.0 + 4.0 * 40.0 / 3.0) * 0.002, 1e-12);
}

// ---------------------------------------------------------------------------
// Guards
// ---------------------------------------------------------------------------

/// "name" is cached in m_name and used as the registry key, so writing it would
/// desynchronise the parameter from the material's identity.
TEST(SetParameter, RejectsWritingTheIdentityKey) {
  ctx_type ctx;
  auto& m = build(ctx);
  EXPECT_THROW(m.template set_parameter<std::string>("name", "renamed"),
               std::invalid_argument);
  EXPECT_EQ(m.name(), "probe");
}

/// A type mismatch must name the key. Without the translation it surfaces as a
/// bare "bad any_cast" that is neither invalid_argument nor runtime_error, so a
/// UMAT boundary catching those would miss it entirely.
TEST(SetParameter, TypeMismatchThrowsADiagnosticNotBadAnyCast) {
  ctx_type ctx;
  auto& m = build(ctx);
  try {
    m.template set_parameter<float>("K", 1.0f);   // "K" is stored as double
    FAIL() << "expected a diagnostic";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("K"), std::string::npos) << e.what();
  }
  EXPECT_DOUBLE_EQ(m.bound_k(), 100.0) << "the value must be unchanged";
}

/// T is not deduced, so a wrong-typed literal is a compile error rather than a
/// run-time throw.
template <typename M, typename = void>
struct deduces_t : std::false_type {};
template <typename M>
struct deduces_t<M, std::void_t<decltype(std::declval<M&>().set_parameter(
                        std::declval<std::string const&>(), 250))>>
    : std::true_type {};

TEST(SetParameter, TypeIsNotDeduced) {
  static_assert(!deduces_t<probe_material<policy>>::value,
                "set_parameter must not deduce T from the value");
  ctx_type ctx;
  auto& m = build(ctx);
  m.template set_parameter<T>("K", 250);   // int converts to the named double
  EXPECT_DOUBLE_EQ(m.bound_k(), 250.0);
}

}  // namespace
