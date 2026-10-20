#ifndef NUMSIM_MATERIALS_REPORT_CASE_RUNNER_H
#define NUMSIM_MATERIALS_REPORT_CASE_RUNNER_H

#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include <nlohmann/json.hpp>
#include "numsim-materials/report/property_columns.h"
#include "numsim-materials/report/report_json.h"
#include "numsim-materials/report/step_record.h"
#include "numsim-materials/umat/deformation_gradient_driver.h"
#include "numsim-materials/umat/json_model.h"
#include "numsim-materials/umat/material_point_driver.h"

namespace numsim::materials::report {

/// Material graph and evaluator of a report document ("model" and "point").
template <typename Traits>
class material_point {
public:
  using evaluator_type = umat::material_point_evaluator<Traits>;

  explicit material_point(nlohmann::json const &doc) {
    umat::make_json_builder<Traits>(doc.at("model").dump())(m_ctx, {});
    m_ev = std::make_unique<evaluator_type>(m_ctx, parse_point<Traits>(doc));
  }

  [[nodiscard]] material_context<Traits> &context() noexcept { return m_ctx; }
  [[nodiscard]] evaluator_type &evaluator() noexcept { return *m_ev; }

private:
  material_context<Traits> m_ctx;
  std::unique_ptr<evaluator_type> m_ev;
};

namespace detail {

template <typename Traits, typename Sink>
bool run_small_strain(nlohmann::json const &doc, load_case<typename Traits::value_type> const &c, Sink &sink,
                      typename umat::material_point_driver<Traits>::options opts) {
  using T = typename Traits::value_type;
  using tensor2 = tmech::tensor<T, 3, 2>;
  material_point<Traits> point(doc);
  umat::material_point_driver<Traits> driver{point.evaluator(), opts};
  property_columns<Traits> const columns{point.context(), parse_columns(doc)};
  sink.begin(columns.names());

  auto const initial{driver.step(tensor2{}, tensor2{}, umat::load_control<T>::strain(), 0, 0)};
  sink.record({c.path.unloaded(), step_summary<T>::of(initial), columns.values()});
  for (auto const &s : c.path.steps()) {
    auto const res{driver.step(c.strain_at(s), tensor2{s.load * c.stress}, c.control, s.time, s.dt)};
    step_record<Traits> rec{s, step_summary<T>::of(res), columns.values()}; // columns before the tangent check re-evaluates the graph
    if (res.converged && c.tangent_every > 0 && s.index % c.tangent_every == 0)
      rec.tangent_error = driver.tangent_error(res);
    sink.record(rec);
    if (!res.converged)
      return false;
  }
  return true;
}

/// Cauchy stress components (Abaqus order) and J, appended to the property columns.
template <typename T>
std::vector<T> cauchy_and_J(tmech::tensor<T, 3, 2> const &F, tmech::tensor<T, 3, 2> const &P) {
  auto const J{tmech::det(F)};
  tmech::tensor<T, 3, 2> const sigma{P * tmech::trans(F) / J};
  std::vector<T> out;
  for (auto const &[i, j] : voigt_indices)
    out.push_back(sigma(i, j));
  out.push_back(J);
  return out;
}

template <typename Traits, typename Sink>
bool run_finite_strain(nlohmann::json const &doc, load_case<typename Traits::value_type> const &c, Sink &sink,
                       typename umat::deformation_gradient_driver<Traits>::options opts) {
  using T = typename Traits::value_type;
  using tensor2 = tmech::tensor<T, 3, 2>;
  material_point<Traits> point(doc);
  umat::deformation_gradient_driver<Traits> driver{point.evaluator(), opts};
  property_columns<Traits> const columns{point.context(), parse_columns(doc)};
  auto names{columns.names()};
  for (auto const &comp : voigt_components)
    names.push_back("stress_" + std::string{comp});
  names.push_back("J");
  sink.begin(names);
  auto const I{tmech::eye<T, 3, 2>()};
  auto const record = [&](load_step<T> const &s, auto const &res, T tangent_error) {
    auto values{columns.values()};
    for (auto const v : cauchy_and_J<T>(res.F, res.stress))
      values.push_back(v);
    sink.record({s, step_summary<T>::of(res), std::move(values), tangent_error});
  };

  auto const initial{driver.step(I, tensor2{}, umat::deformation_control<T>::deformation(), 0, 0)};
  record(c.path.unloaded(), initial, std::numeric_limits<T>::quiet_NaN());
  for (auto const &s : c.path.steps()) {
    auto const res{driver.step(tensor2{I + c.strain_at(s)}, tensor2{s.load * c.stress}, c.finite_control, s.time, s.dt)};
    auto values{columns.values()}; // before the tangent check re-evaluates the graph
    for (auto const v : cauchy_and_J<T>(res.F, res.stress))
      values.push_back(v);
    T err{std::numeric_limits<T>::quiet_NaN()};
    if (res.converged && c.tangent_every > 0 && s.index % c.tangent_every == 0)
      err = driver.tangent_error(res);
    sink.record({s, step_summary<T>::of(res), std::move(values), err});
    if (!res.converged)
      return false;
  }
  return true;
}

} // namespace detail

/// Runs one load case on a fresh material point; false if a step does not converge (it is still recorded).
/// Small strain drives the strain, finite strain F = I + (interpolated points), and records P, Cauchy stress and J.
template <typename Traits, typename Sink>
  requires step_sink<Sink, Traits>
bool run_case(nlohmann::json const &doc, load_case<typename Traits::value_type> const &c, Sink &sink,
              typename umat::material_point_driver<Traits>::options opts = {}) {
  if (is_finite_strain(doc))
    return detail::run_finite_strain<Traits>(doc, c, sink, {.tolerance = opts.tolerance, .max_iter = opts.max_iter, .fd_step = c.fd_step.value_or(opts.fd_step)});
  opts.fd_step = c.fd_step.value_or(opts.fd_step);
  return detail::run_small_strain<Traits>(doc, c, sink, opts);
}

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_CASE_RUNNER_H
