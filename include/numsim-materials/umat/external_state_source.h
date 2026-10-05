#ifndef EXTERNAL_STATE_SOURCE_H
#define EXTERNAL_STATE_SOURCE_H

#include <limits>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/umat/tensor_conversion.h"

/// Host-driven replacements for the self-driving steppers.
///
/// `tensor_component_stepper` and `scalar_stepper` invent their own load path.
/// Under a UMAT the host supplies it instead, so these two materials publish the
/// SAME properties with the SAME history semantics, but take their values from
/// the caller.
///
/// Three properties of the design are load-bearing:
///
///  1. The outputs are HISTORY properties, exactly as the steppers' are. The
///     old/new pair is what `input_history` consumers bind to, and it is where
///     DSTRAN (= new - old) and dt (= new - old) come from. Publishing plain
///     properties instead would break every `add_input_history` consumer, since
///     `input_history::wire()` requires a genuine history property.
///
///  2. Values are written by `bind()`, not by an update callback. The material
///     therefore has no callback at all, which means the values are already in
///     place before `ctx.update()` runs and graph ordering cannot affect them.
///
///  3. Consequently `input_property::update_source()` on these is a no-op —
///     which is exactly right. Host-supplied data must stay frozen while a
///     solver iterates internally. Contrast `tensor_component_stepper::update`,
///     whose `+=` is documented as unsafe to re-run within a step.
namespace numsim::materials {

/// Host-driven strain source. Drop-in for `tensor_component_stepper<2, Traits>`
/// from a consumer's point of view: same "strain" property, same history type.
template <typename Traits>
class external_strain_source final
    : public material_base<external_strain_source<Traits>, Traits> {
public:
  using base = material_base<external_strain_source<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;

  static_assert(Dim == 3,
                "external_strain_source is 3D only: every element family runs "
                "the same Dim=3 material and 2D is handled by widening the "
                "host vector (see umat/tensor_conversion.h), never by a 2D "
                "policy");

  template <typename... Args>
  explicit external_strain_source(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_strain(base::template add_history_output<tensor2>("strain")) {}

  static input_parameter_controller parameters() { return base::parameters(); }

  /// Bind from canonical 6-slot buffers — Abaqus/Standard order with
  /// ENGINEERING shear, i.e. STRAN and STRAN+DSTRAN after widening.
  ///
  /// Widening is the caller's job, not this material's: the element family and
  /// (for plane stress) the iterated out-of-plane strain both live in the
  /// evaluator, so keeping element_case out of here means it is known in
  /// exactly one place.
  void bind(const value_type* old6, const value_type* new6) {
    m_strain.old_value() = umat::strain_from_buffer<value_type>(old6);
    m_strain.new_value() = umat::strain_from_buffer<value_type>(new6);
  }

  /// Bind directly from tensors, for C++ drivers and tests that never touch a
  /// raw buffer.
  void bind(const tensor2& old_strain, const tensor2& new_strain) {
    m_strain.old_value() = old_strain;
    m_strain.new_value() = new_strain;
  }

  const history_property<tensor2>& strain() const noexcept { return m_strain; }

private:
  history_property<tensor2>& m_strain;
};

/// Host-driven deformation-gradient source for finite-strain models: the
/// "deformation_gradient" history property, F at t_n and t_{n+1}, written by
/// bind() like the strain of external_strain_source. With it, bind() also
/// publishes "inverse" (F^-1) and "determinant" (J = det F) of the new F, so
/// consumers share them instead of each inverting F. They are plain
/// properties of the current state, not history, and add nothing to STATEV.
/// For det F = 0 the inverse is NaN; consumers check J. Consumers (the
/// hyperelastic models) publish the first Piola-Kirchhoff stress as "stress"
/// and dP/dF as "tangent".
template <typename Traits>
class external_deformation_gradient_source final
    : public material_base<external_deformation_gradient_source<Traits>, Traits> {
public:
  using base = material_base<external_deformation_gradient_source<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;

  static_assert(Dim == 3, "external_deformation_gradient_source is 3D only");

  template <typename... Args>
  explicit external_deformation_gradient_source(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_F(base::template add_history_output<tensor2>("deformation_gradient")),
        m_F_inv(base::template add_output<tensor2>("inverse")),
        m_J(base::template add_output<value_type>("determinant")) {
    m_F.old_value() = tmech::eye<value_type, Dim, 2>();
    m_F.new_value() = tmech::eye<value_type, Dim, 2>();
    derive();
  }

  static input_parameter_controller parameters() { return base::parameters(); }

  void bind(const tensor2& old_F, const tensor2& new_F) {
    m_F.old_value() = old_F;
    m_F.new_value() = new_F;
    derive();
  }

  /// Bind from two 9-slot buffers in row-major order (F_11 F_12 F_13 F_21 ...).
  void bind(const value_type* old9, const value_type* new9) {
    m_F.old_value() = tmech::adaptor<const value_type, Dim, 2, tmech::full<Dim>>{old9};
    m_F.new_value() = tmech::adaptor<const value_type, Dim, 2, tmech::full<Dim>>{new9};
    derive();
  }

  const history_property<tensor2>& deformation_gradient() const noexcept { return m_F; }
  const tensor2& inverse() const noexcept { return m_F_inv; }
  value_type determinant() const noexcept { return m_J; }

private:
  /// Inverse and determinant of the new F.
  void derive() {
    const auto& F{m_F.new_value()};
    m_J = tmech::det(F);
    if (m_J != value_type(0))
      m_F_inv = tmech::inv(F);
    else
      m_F_inv = std::numeric_limits<value_type>::quiet_NaN() * tmech::eye<value_type, Dim, 2>();
  }

  history_property<tensor2>& m_F;
  tensor2& m_F_inv;
  value_type& m_J;
};

/// Host-driven scalar source — time, temperature, or any other externally
/// prescribed scalar. Drop-in for `scalar_stepper`: same "state" property, same
/// history type, so consumers such as `autocatalytic_reaction` (which forms
/// dt = new - old) work against it unchanged.
template <typename Traits>
class external_scalar_source final
    : public material_base<external_scalar_source<Traits>, Traits> {
public:
  using base = material_base<external_scalar_source<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit external_scalar_source(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_state(base::template add_history_output<value_type>("state")) {}

  static input_parameter_controller parameters() { return base::parameters(); }

  /// Bind the value at t_n and t_{n+1}. For time under a UMAT that is
  /// TIME(2) and TIME(2) + DTIME, so consumers recover dt as new - old.
  void bind(value_type old_value, value_type new_value) {
    m_state.old_value() = old_value;
    m_state.new_value() = new_value;
  }

  const history_property<value_type>& state() const noexcept { return m_state; }

private:
  history_property<value_type>& m_state;
};

}  // namespace numsim::materials

#endif  // EXTERNAL_STATE_SOURCE_H
