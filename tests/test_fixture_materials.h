#ifndef NUMSIM_MATERIALS_TESTS_TEST_FIXTURE_MATERIALS_H
#define NUMSIM_MATERIALS_TESTS_TEST_FIXTURE_MATERIALS_H

#include "numsim-materials/core/material_base.h"

/// Test-only materials shared by several test files.
namespace numsim::materials::test {

/// An int history ("count", +1 per update): a history no STATEV buffer can hold.
template <typename Traits>
class int_counter final : public material_base<int_counter<Traits>, Traits> {
public:
  using base = material_base<int_counter<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit int_counter(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_count(base::template add_history_output<int>("count", &int_counter::compute)) {}

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() { m_count.new_value() = m_count.old_value() + 1; }

private:
  history_property<int> &m_count;
};

} // namespace numsim::materials::test

#endif // NUMSIM_MATERIALS_TESTS_TEST_FIXTURE_MATERIALS_H
