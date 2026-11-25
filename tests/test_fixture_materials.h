#ifndef NUMSIM_MATERIALS_TESTS_TEST_FIXTURE_MATERIALS_H
#define NUMSIM_MATERIALS_TESTS_TEST_FIXTURE_MATERIALS_H

#include <tmech/tmech.h>
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
        m_count(base::template add_history_output<int>("count")) {
    base::add_callback(&int_counter::compute, m_count).reads();
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void compute() { m_count.new_value() = m_count.old_value() + 1; }

private:
  history_property<int> &m_count;
};

/// A history tensor "F" declared unsymmetric_output(), set by bind(): an unsymmetric seed with no driver behind it.
template <typename Traits>
class unsymmetric_source final : public material_base<unsymmetric_source<Traits>, Traits> {
public:
  using base = material_base<unsymmetric_source<Traits>, Traits>;
  using input_parameter_controller = typename base::input_parameter_controller;
  using tensor2 = tmech::tensor<typename Traits::value_type, Traits::Dim, 2>;

  template <typename... Args>
  explicit unsymmetric_source(Args &&...args) : base(std::forward<Args>(args)...), m_F(base::template add_history_output<tensor2>("F")) {
    base::unsymmetric_output(m_F);
  }

  static input_parameter_controller parameters() { return input_parameter_controller{base::parameters()}; }

  void bind(tensor2 const &F) { m_F.new_value() = F; }

private:
  history_property<tensor2> &m_F;
};

} // namespace numsim::materials::test

#endif // NUMSIM_MATERIALS_TESTS_TEST_FIXTURE_MATERIALS_H
