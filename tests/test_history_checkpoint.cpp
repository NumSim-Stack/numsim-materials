#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/driver/material_point.h"
#include "numsim-materials/umat/errors.h"
#include "numsim-materials/umat/external_state_source.h"
#include "test_fixture_materials.h"

// History checkpoints within a step (#215): restore() returns to the checkpoint after any number of commits, and
// clears a failed update like revert().

namespace {

namespace nm = numsim::materials;
using policy = nm::material_policy_default;
using T = policy::value_type;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using ctx_type = nm::material_context<policy>;
using param_type = policy::ParameterHandler;

/// A rank-4 history "moment" summing strain ⊗ strain over the steps; throws a cutback_error above "limit".
template <typename Traits>
class moments final : public nm::material_base<moments<Traits>, Traits> {
public:
  using base = nm::material_base<moments<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit moments(Args &&...args)
      : base(std::forward<Args>(args)...), m_moment(base::template add_history_output<tensor4>("moment")),
        m_limit(base::template get_parameter<value_type>("limit")),
        m_strain(base::template add_input<tensor2>("strain", "strain", nm::EdgeKind::Global)) {
    base::add_callback(&moments::compute, m_moment).reads(m_strain);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::optional<value_type>(para, "limit", value_type{1e30});
    return para;
  }

  void compute() {
    if (tmech::norm(m_strain.get()) > m_limit)
      throw nm::umat::cutback_error("moments: above the limit");
    m_moment.new_value() = m_moment.old_value() + tmech::otimes(m_strain.get(), m_strain.get());
  }

private:
  nm::history_property<tensor4> &m_moment;
  value_type const &m_limit;
  nm::input_property<tensor2, nm::property_traits> const &m_strain;
};

/// strain (external), "tag" (int history) and "mom" (rank-4 history).
struct setup {
  ctx_type ctx;

  explicit setup(T limit = 1e30) {
    param_type p;
    p.insert<std::string>("name", "strain");
    ctx.create<nm::external_strain_source<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "tag");
    ctx.create<nm::test::int_counter<policy>>(p);
    p.clear();
    p.insert<std::string>("name", "mom");
    p.insert<T>("limit", limit);
    ctx.create<moments<policy>>(p);
    ctx.finalize();
  }

  /// One update to the strain e (11) and e/2 (12), from the accepted strain.
  void update(T e) {
    auto &source{dynamic_cast<nm::external_strain_source<policy> &>(*ctx.find("strain"))};
    tensor2 eps;
    eps(0, 0) = e;
    eps(0, 1) = eps(1, 0) = 0.5 * e;
    source.bind(tensor2{source.strain().old_value()}, eps);
    ctx.update();
  }

  /// Old and new value of every history property, byte for byte.
  std::string bytes() const {
    std::ostringstream out;
    ctx.save_state(out);
    return std::move(out).str();
  }

  int count() { return ctx.get<int>("tag", "count"); }
};

TEST(HistoryCheckpoint, RestoreUndoesTwoCommittedStepsBitwise) {
  setup s;
  s.update(1e-3);
  s.ctx.commit();
  s.update(2e-3); // a trial: old and new differ
  auto const cp{s.ctx.checkpoint()};
  auto const before{s.bytes()};
  for (T const e : {3e-3, 4e-3}) {
    s.update(e);
    s.ctx.commit();
  }
  ASSERT_EQ(s.count(), 3); // the trial at 2e-3 was never committed
  ASSERT_NE(s.bytes(), before);
  s.ctx.restore(cp);
  EXPECT_EQ(s.bytes(), before);
  EXPECT_EQ(s.count(), 2);
  auto const *count{dynamic_cast<nm::history_property<int> const *>(s.ctx.find_property("tag", "count"))};
  ASSERT_NE(count, nullptr);
  EXPECT_EQ(count->old_value(), 1);
}

TEST(HistoryCheckpoint, AContinuationAfterRestoreMatchesTheUninterruptedRun) {
  setup restored, uninterrupted;
  restored.update(1e-3);
  restored.ctx.commit();
  auto const cp{restored.ctx.checkpoint()};
  restored.update(5e-3);
  restored.ctx.commit();
  restored.ctx.restore(cp);
  uninterrupted.update(1e-3);
  uninterrupted.ctx.commit();
  for (T const e : {2e-3, 3e-3}) {
    restored.update(e);
    restored.ctx.commit();
    uninterrupted.update(e);
    uninterrupted.ctx.commit();
  }
  EXPECT_EQ(restored.bytes(), uninterrupted.bytes());
}

TEST(HistoryCheckpoint, RestoreAfterAFailedUpdateLeavesAContextThatUpdatesAgain) {
  setup s{0.01};
  s.update(1e-3);
  s.ctx.commit();
  auto const cp{s.ctx.checkpoint()};
  auto const before{s.bytes()};
  s.update(2e-3);
  s.ctx.commit(); // an accepted substep
  EXPECT_THROW(s.update(1.0), nm::umat::cutback_error);
  EXPECT_THROW(s.ctx.commit(), std::logic_error) << "partly updated";
  EXPECT_THROW(static_cast<void>(s.ctx.checkpoint()), std::logic_error) << "a partly updated state is no checkpoint";
  s.ctx.restore(cp);
  EXPECT_EQ(s.bytes(), before);
  EXPECT_NO_THROW(s.update(2e-3));
  s.ctx.commit();
  EXPECT_EQ(s.count(), 2);
}

TEST(HistoryCheckpoint, RestoringAnotherContextsCheckpointThrows) {
  setup a, b;
  a.update(1e-3);
  auto const cp{a.ctx.checkpoint()};
  auto const before{b.bytes()};
  EXPECT_THROW(b.ctx.restore(cp), std::logic_error);
  EXPECT_EQ(b.bytes(), before);
  EXPECT_THROW(b.ctx.restore(ctx_type::checkpoint_type{}), std::logic_error) << "an empty checkpoint has no owner";
}

TEST(HistoryCheckpoint, NeedsAFinalizedContext) {
  ctx_type ctx;
  EXPECT_THROW(static_cast<void>(ctx.checkpoint()), std::logic_error);
}

TEST(HistoryCheckpoint, TheMaterialPointForwardsCheckpointAndRestore) {
  setup s;
  param_type p;
  p.insert<std::vector<std::string>>("inputs", {"strain"});
  nm::material_point<policy> point{s.ctx, p};
  s.update(1e-3);
  point.accept();
  auto const cp{point.checkpoint()};
  auto const before{s.bytes()};
  s.update(2e-3);
  point.accept();
  point.restore(cp);
  EXPECT_EQ(s.bytes(), before);
}

} // namespace
