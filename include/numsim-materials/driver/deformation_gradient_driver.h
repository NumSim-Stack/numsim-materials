#ifndef NUMSIM_MATERIALS_DRIVER_DEFORMATION_GRADIENT_DRIVER_H
#define NUMSIM_MATERIALS_DRIVER_DEFORMATION_GRADIENT_DRIVER_H

#include <algorithm>
#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/driver/material_point.h"
#include "numsim-materials/umat/errors.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/driver/load_control.h"

/// One material point at finite strain under mixed F / P control: Newton on the free components of F.
///
/// The history lives in the graph, as for material_point_driver: step() leaves the converged trial, accept()
/// makes it the start of the next step, reject() returns to the accepted state. The accepted F is the old
/// value of the deformation gradient source.
namespace numsim::materials {

template <typename Traits>
class deformation_gradient_driver {
public:
  using value_type = typename Traits::value_type;
  using parameter_handler = typename Traits::ParameterHandler;
  using input_parameter_controller = typename Traits::InputParameterController;
  using tensor2 = tmech::tensor<value_type, 3, 2>;
  using tensor4 = tmech::tensor<value_type, 3, 4>;
  using control = deformation_control<value_type>;

  struct step_result {
    tensor2 F{};
    tensor2 stress{};  ///< first Piola-Kirchhoff
    tensor4 tangent{}; ///< dP/dF
    std::vector<value_type> residuals{}; ///< |M:(P - P*)| before each correction
    bool converged{false};
    value_type time{0}, dt{0};
  };

  static input_parameter_controller parameters() {
    input_parameter_controller para;
    required<std::string>(para, "deformation_gradient_source");    // external_deformation_gradient_source
    required<std::string>(para, "stress_source");                  // material publishing "stress" (P)
    optional<std::string>(para, "tangent_source", std::string{});  // publishing "tangent" (dP/dF); empty: stress_source
    optional<std::string>(para, "time_source", std::string{});     // external_scalar_source; empty: none
    optional<value_type>(para, "tolerance", value_type{1e-10});    // on |M:(P - P*)| relative to |P|
    optional<int>(para, "max_iter", 25);
    optional<value_type>(para, "fd_step", value_type{1e-7});       // perturbation of F in tangent_error()
    // passed to the material point (same defaults): a fixed point of the graph, e.g. "scs::residual"
    optional<std::string>(para, "residual_source", std::string{});
    optional<value_type>(para, "residual_tolerance", value_type{1e-10});
    optional<int>(para, "max_passes", 100);
    return para;
  }

  deformation_gradient_driver(material_context<Traits> &ctx, parameter_handler parameters)
      : m_parameters(checked(std::move(parameters))), m_point(ctx, point_parameters()),
        m_F(m_point.template input<external_deformation_gradient_source<Traits>>(text("deformation_gradient_source"))),
        m_stress(m_point.template output<tensor2>(text("stress_source"), "stress")),
        m_tangent(m_point.template output<tensor4>(
            text("tangent_source").empty() ? text("stress_source") : text("tangent_source"), "tangent")),
        m_tolerance(m_parameters.template get<value_type>("tolerance")),
        m_max_iter(m_parameters.template get<int>("max_iter")),
        m_fd_step(m_parameters.template get<value_type>("fd_step")) {
    if (m_max_iter < 0 || !(m_tolerance >= 0) || !(m_fd_step > 0))
      throw umat::fatal_error("deformation_gradient_driver: max_iter and tolerance must be >= 0, fd_step > 0");
    if (!text("time_source").empty())
      m_time = &m_point.template input<external_scalar_source<Traits>>(text("time_source"));
  }

  /// Prescribed components of F from `F`, stress-controlled ones from `P`, from the accepted F.
  /// Leaves the converged trial in the graph: accept() or reject() next.
  step_result step(tensor2 const &F, tensor2 const &P, control const &c, value_type time, value_type dt) {
    auto const &M{c.stress_projector()};
    tensor4 const N{c.deformation_projector()};
    tensor2 const start{accepted_F()};
    step_result res{.F = tmech::dcontract(N, F) + tmech::dcontract(M, start), .time = time, .dt = dt};
    for (int it{0}; it <= m_max_iter; ++it) {
      evaluate(res.F, time, dt);
      res.stress = m_stress;
      res.tangent = m_tangent;
      tensor2 const r{tmech::dcontract(M, tensor2{res.stress - P})};
      res.residuals.push_back(tmech::norm(r));
      // as for small strain: the step's own stress scale when P passes through zero
      auto const scale{std::max({tmech::norm(res.stress), tmech::norm(tensor2{tmech::dcontract(M, P)}),
                                 tmech::norm(res.tangent) * tmech::norm(tensor2{res.F - start})})};
      if (res.residuals.back() <= m_tolerance * scale) {
        res.converged = true;
        break;
      }
      if (it == m_max_iter)
        break; // the result stays the evaluated trial
      tensor4 const A{tmech::dcontract(tensor4{tmech::dcontract(M, res.tangent)}, M) + N};
      res.F = tmech::eval(res.F - tmech::dcontract(tensor4{tmech::invf(A)}, r));
    }
    return res;
  }

  /// The trial of step() becomes the start of the next step.
  void accept() { m_point.accept(); }

  /// Back to the accepted state.
  void reject() { m_point.reset(); }

  /// Relative error of dP/dF of the trial `res` against central differences, before accept(). The trial is
  /// evaluated again afterwards, so accepting it is unaffected.
  value_type tangent_error(step_result const &res) {
    tensor4 fd;
    for (auto const &E : materials::full_basis<value_type>()) {
      evaluate(tensor2{res.F + m_fd_step * E}, res.time, res.dt);
      tensor2 const plus{m_stress};
      evaluate(tensor2{res.F - m_fd_step * E}, res.time, res.dt);
      fd += tmech::otimes(tensor2{(plus - m_stress) / (2 * m_fd_step)}, E);
    }
    evaluate(res.F, res.time, res.dt);
    return tmech::norm(tensor4{fd - res.tangent}) / tmech::norm(res.tangent);
  }

  [[nodiscard]] tensor2 const &accepted_F() const noexcept { return m_F.deformation_gradient().old_value(); }

  [[nodiscard]] material_point<Traits> &point() noexcept { return m_point; }

private:
  static parameter_handler checked(parameter_handler parameters) {
    static auto const check{deformation_gradient_driver::parameters()};
    check.check_parameter(parameters);
    return parameters;
  }

  std::string const &text(std::string const &key) const { return m_parameters.template get<std::string>(key); }

  /// The point's inputs (the kinematics source and the optional time source) and residual settings.
  parameter_handler point_parameters() const {
    std::vector<std::string> inputs{text("deformation_gradient_source")};
    if (!text("time_source").empty())
      inputs.push_back(text("time_source"));
    parameter_handler out;
    out.template insert<std::vector<std::string>>("inputs", std::move(inputs));
    out.template insert<std::string>("residual_source", text("residual_source"));
    out.template insert<value_type>("residual_tolerance", m_parameters.template get<value_type>("residual_tolerance"));
    out.template insert<int>("max_passes", m_parameters.template get<int>("max_passes"));
    return out;
  }

  void evaluate(tensor2 const &F, value_type time, value_type dt) {
    m_point.reset();
    m_F.bind(tensor2{accepted_F()}, F);
    if (m_time)
      m_time->bind(time, time + dt);
    m_point.evaluate();
  }

  parameter_handler m_parameters;
  material_point<Traits> m_point;
  external_deformation_gradient_source<Traits> &m_F;
  external_scalar_source<Traits> *m_time{nullptr};
  tensor2 const &m_stress;
  tensor4 const &m_tangent;
  value_type m_tolerance;
  int m_max_iter;
  value_type m_fd_step;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DRIVER_DEFORMATION_GRADIENT_DRIVER_H
