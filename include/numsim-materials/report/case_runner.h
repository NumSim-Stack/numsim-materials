#ifndef NUMSIM_MATERIALS_REPORT_CASE_RUNNER_H
#define NUMSIM_MATERIALS_REPORT_CASE_RUNNER_H

#include <stdexcept>
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

/// Material graph of a report document ("model").
template <typename Traits>
class case_model {
public:
  explicit case_model(nlohmann::json const &doc) { umat::make_json_builder<Traits>(doc.at("model").dump())(m_ctx, {}); }

  [[nodiscard]] material_context<Traits> &context() noexcept { return m_ctx; }

private:
  material_context<Traits> m_ctx;
};

namespace detail {

/// One step of a case: the columns of the trial, the tangent check when due, then accept, or reject and stop.
template <typename Traits, typename Driver, typename Sink, typename Values>
bool record_step(Driver &driver, typename Driver::step_result const &res, load_step<typename Traits::value_type> const &s,
                 int tangent_every, Sink &sink, Values const &values) {
  step_record<Traits> rec{s, step_summary<typename Traits::value_type>::of(res), values(res)};
  if (res.converged && tangent_every > 0 && s.index % tangent_every == 0)
    rec.tangent_error = driver.tangent_error(res);
  sink.record(rec);
  if (!res.converged) {
    driver.reject();
    return false;
  }
  driver.accept();
  return true;
}

template <typename Traits, typename Sink>
bool run_small_strain(nlohmann::json const &doc, load_case<typename Traits::value_type> const &c, Sink &sink) {
  using T = typename Traits::value_type;
  using tensor2 = tmech::tensor<T, 3, 2>;
  case_model<Traits> model(doc);
  umat::material_point_driver<Traits> driver{model.context(), parse_point<Traits>(doc)};
  property_columns<Traits> const columns{model.context(), parse_columns(doc)};
  sink.begin(columns.names());
  auto const values = [&](auto const &) { return columns.values(); };

  auto const initial{driver.step(tensor2{}, tensor2{}, umat::load_control<T>::strain(), 0, 0)};
  if (!record_step<Traits>(driver, initial, c.path.unloaded(), 0, sink, values))
    return false;
  for (auto const &s : c.path.steps())
    if (!record_step<Traits>(driver, driver.step(c.strain_at(s), tensor2{s.load * c.stress}, c.control, s.time, s.dt), s,
                             c.tangent_every, sink, values))
      return false;
  return true;
}

/// Cauchy stress components (symmetric_components order) and J, appended to the property columns.
template <typename T>
std::vector<T> cauchy_and_J(tmech::tensor<T, 3, 2> const &F, tmech::tensor<T, 3, 2> const &P) {
  auto const J{tmech::det(F)};
  tmech::tensor<T, 3, 2> const sigma{P * tmech::trans(F) / J};
  std::vector<T> out;
  for (auto const &[i, j] : symmetric_indices)
    out.push_back(sigma(i, j));
  out.push_back(J);
  return out;
}

template <typename Traits, typename Sink>
bool run_finite_strain(nlohmann::json const &doc, load_case<typename Traits::value_type> const &c, Sink &sink) {
  using T = typename Traits::value_type;
  using tensor2 = tmech::tensor<T, 3, 2>;
  case_model<Traits> model(doc);
  umat::deformation_gradient_driver<Traits> driver{model.context(), parse_point<Traits>(doc)};
  property_columns<Traits> const columns{model.context(), parse_columns(doc)};
  auto names{columns.names()};
  for (auto const &comp : symmetric_components)
    names.push_back("stress_" + std::string{comp});
  names.push_back("J");
  sink.begin(names);
  auto const values = [&](auto const &res) {
    auto out{columns.values()};
    for (auto const v : cauchy_and_J<T>(res.F, res.stress))
      out.push_back(v);
    return out;
  };

  auto const I{tmech::eye<T, 3, 2>()};
  auto const initial{driver.step(I, tensor2{}, umat::deformation_control<T>::deformation(), 0, 0)};
  if (!record_step<Traits>(driver, initial, c.path.unloaded(), 0, sink, values))
    return false;
  for (auto const &s : c.path.steps())
    if (!record_step<Traits>(driver, driver.step(tensor2{I + c.strain_at(s)}, tensor2{s.load * c.stress}, c.finite_control, s.time, s.dt),
                             s, c.tangent_every, sink, values))
      return false;
  return true;
}

} // namespace detail

/// Runs one load case on a fresh material point; false if a step does not converge (it is still recorded, then
/// rejected). Every converged step is accepted after its columns and tangent check. Small strain drives the
/// strain, finite strain F = I + (interpolated points), and records P, Cauchy stress and J. The driver's
/// parameters are the document's "point".
template <typename Traits, typename Sink>
  requires step_sink<Sink, Traits>
bool run_case(nlohmann::json const &doc, load_case<typename Traits::value_type> const &c, Sink &sink) {
  if (c.finite_strain != is_finite_strain(doc))
    throw std::invalid_argument("case '" + c.name + "': parsed for the other kinematics (parse_case's finite_strain)");
  if (c.finite_strain)
    return detail::run_finite_strain<Traits>(doc, c, sink);
  return detail::run_small_strain<Traits>(doc, c, sink);
}

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_CASE_RUNNER_H
