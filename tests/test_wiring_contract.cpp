/// Two ways a material could reach for something that was not there yet.
///
/// material_ref::get() was guarded only by an assert, and Release is
/// -O3 -DNDEBUG, so a use-before-wire was a null dereference in exactly the
/// builds that run real analyses. And material_interface used to carry a
/// vestigial virtual update(), so the obvious way to write a material --
/// `void update() override` -- compiled, ran nothing, and left the property at
/// its initial value with no diagnostic.
#include <gtest/gtest.h>
#include <type_traits>

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/material_ref.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/solvers/local_newton.h"

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;
using stepper2 = nm::tensor_component_stepper<2, policy>;

// --- material_ref::get() before wiring (#59) ---

/// A material that holds a reference to a solver and hands out access to it,
/// so the test can reach get() at a chosen moment rather than only from
/// inside a graph-driven callback.
template <typename Traits>
class ref_holder final : public nm::material_base<ref_holder<Traits>, Traits> {
public:
  using base = nm::material_base<ref_holder<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit ref_holder(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_out(base::template add_output<value_type>("value",
                                                    &ref_holder::compute)),
        m_target(base::template add_material_ref<nm::local_newton<Traits>>(
            base::template get_parameter<std::string>("solver_source"))) {}

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("solver_source")
        .template add<numsim_core::is_required>();
    return para;
  }

  void compute() { m_out = value_type{1}; }

  /// Reaches the reference directly, which is what a constructor resolving
  /// too eagerly or a hand-invoked callback would do.
  void touch_target() { (void)m_target.get(); }

  /// get() is overloaded on constness and a material can reach either. A test
  /// that exercised only the mutable one left the const overload unguarded --
  /// removing its check passed the whole suite.
  void touch_target_const() const {
    const auto& ref = m_target;
    (void)ref.get();
  }

private:
  value_type& m_out;
  nm::material_ref<nm::local_newton<Traits>, Traits>& m_target;
};

TEST(WiringContract, AMaterialRefReadBeforeFinalizeThrows) {
  ctx_type ctx;
  param_type p;

  p.insert<std::string>("name", "solver");
  p.insert<T>("tolerance", T{1e-10});
  p.insert<int>("max_iter", 50);
  ctx.create<nm::local_newton<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "holder");
  p.insert<std::string>("solver_source", "solver");
  auto& holder = ctx.create<ref_holder<policy>>(p);

  // Wiring happens in finalize(), which has not run. Before the guard this
  // dereferenced nullptr in Release. Both overloads, because a material can
  // reach either and each needs its own check.
  EXPECT_THROW(holder.touch_target(), std::logic_error);
  EXPECT_THROW(holder.touch_target_const(), std::logic_error);
}

TEST(WiringContract, TheUnwiredRefErrorNamesTheTarget) {
  ctx_type ctx;
  param_type p;

  p.insert<std::string>("name", "the_solver");
  p.insert<T>("tolerance", T{1e-10});
  p.insert<int>("max_iter", 50);
  ctx.create<nm::local_newton<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "holder");
  p.insert<std::string>("solver_source", "the_solver");
  auto& holder = ctx.create<ref_holder<policy>>(p);

  try {
    holder.touch_target();
    FAIL() << "an unwired material_ref was dereferenced";
  } catch (const std::logic_error& e) {
    EXPECT_NE(std::string(e.what()).find("the_solver"), std::string::npos)
        << "the message does not name the target: " << e.what();
  }
}

/// The guard must not fire once wiring has happened -- otherwise it would
/// pass the test above while breaking every model.
TEST(WiringContract, AMaterialRefReadAfterFinalizeSucceeds) {
  ctx_type ctx;
  param_type p;

  p.insert<std::string>("name", "solver");
  p.insert<T>("tolerance", T{1e-10});
  p.insert<int>("max_iter", 50);
  ctx.create<nm::local_newton<policy>>(p);

  p.clear();
  p.insert<std::string>("name", "holder");
  p.insert<std::string>("solver_source", "solver");
  auto& holder = ctx.create<ref_holder<policy>>(p);

  ctx.finalize();
  EXPECT_NO_THROW(holder.touch_target());
  EXPECT_NO_THROW(holder.touch_target_const());
}

// --- material_interface must carry no update() to override (#62) ---

template <typename M, typename = void>
struct has_update : std::false_type {};

template <typename M>
struct has_update<M, std::void_t<decltype(std::declval<M&>().update())>>
    : std::true_type {};

/// A regression guard, because the failure this prevents is a COMPILE error in
/// someone else's material and nothing here would otherwise notice the virtual
/// coming back. If material_interface grows an update() again, a newcomer's
/// `void update() override` compiles once more and silently does nothing.
TEST(WiringContract, MaterialInterfaceHasNoUpdateToOverride) {
  static_assert(!has_update<nm::material_interface<policy>>::value,
                "material_interface declares update() again. That is the "
                "vestigial virtual removed for #62: with it back, writing "
                "`void update() override` in a material compiles, runs "
                "nothing, and leaves the property at its initial value.");
  SUCCEED();
}

/// The materials that DO define update() keep it -- it is their callback,
/// passed to add_output as a member pointer. Removing the virtual must not
/// have removed the mechanism.
TEST(WiringContract, AMaterialsOwnUpdateStillDrivesItsProperty) {
  static_assert(has_update<stepper2>::value,
                "tensor_component_stepper lost its update()");

  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "stepper");
  p.insert<T>("increment", T{0.25});
  p.insert<std::vector<std::size_t>>("indices", {0, 0});
  ctx.create<stepper2>(p);
  ctx.finalize();
  ctx.update();

  const auto& strain = ctx.get<tmech::tensor<T, 3, 2>>("stepper", "strain");
  EXPECT_DOUBLE_EQ(strain(0, 0), 0.25)
      << "the callback bound through add_output did not run";
}

} // namespace
