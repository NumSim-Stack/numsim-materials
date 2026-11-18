#ifndef NUMSIM_MATERIALS_REPORT_CASE_RUNNER_H
#define NUMSIM_MATERIALS_REPORT_CASE_RUNNER_H

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

#include <nlohmann/json.hpp>
#include "numsim-materials/driver/deformation_gradient_driver.h"
#include "numsim-materials/driver/material_point_driver.h"
#include "numsim-materials/report/property_columns.h"
#include "numsim-materials/report/report_json.h"
#include "numsim-materials/report/step_record.h"
#include "numsim-materials/umat/json_model.h"

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

/// The document's "point", with the case's fd_step when it sets one.
template <typename Traits>
typename Traits::ParameterHandler point_for(nlohmann::json const &doc, load_case<typename Traits::value_type> const &c) {
  auto out{parse_point<Traits>(doc)};
  if (c.fd_step)
    out.template insert<typename Traits::value_type>("fd_step", *c.fd_step);
  return out;
}

/// One step of a case: the columns of the trial, the tangent check when due, then accept, or reject and stop.
template <typename Traits, typename Driver, typename Sink, typename Values>
bool record_step(Driver &driver, typename Driver::step_result const &res, load_step<typename Traits::value_type> const &s,
                 int tangent_every, Sink &sink, Values const &values, int cutbacks = 0,
                 std::vector<step_summary<typename Traits::value_type>> substeps = {}) {
  using T = typename Traits::value_type;
  if (substeps.empty())
    substeps.push_back(step_summary<T>::of(res));
  step_record<Traits> rec{s, step_summary<T>::of(res), values(res), std::numeric_limits<T>::quiet_NaN(), cutbacks,
                          std::move(substeps)};
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

/// Step `s` of the case in 2^cut substeps, every one but the last accepted. A substep that does not converge or
/// throws a runtime_error (a cutback_error, a return map) returns to the start of the step and halves again, up to
/// the case's max_cutbacks (0 ... 20); at the last level it is recorded (or rethrown) as without cutbacks. A
/// fatal_error or a logic_error is never retried.
template <typename Traits, typename Driver, typename Sink, typename Values, typename Solve>
bool cut_step(Driver &driver, load_case<typename Traits::value_type> const &c, load_step<typename Traits::value_type> const &s,
              Solve const &solve, Sink &sink, Values const &values) {
  using T = typename Traits::value_type;
  if (c.max_cutbacks < 0 || c.max_cutbacks > 20)
    throw std::invalid_argument("max_cutbacks must be in 0 ... 20");
  if (c.max_cutbacks == 0)
    return record_step<Traits>(driver, solve(s), s, c.tangent_every, sink, values);
  auto const start{driver.point().checkpoint()};
  for (int cut{0};; ++cut) {
    bool const last_level{cut == c.max_cutbacks};
    auto const subs{c.path.substeps(s, 1 << cut)};
    std::vector<step_summary<T>> summaries;
    try {
      for (auto const &sub : subs) {
        auto const res{solve(sub)};
        summaries.push_back(step_summary<T>::of(res));
        if (!res.converged && !last_level)
          break;
        if (!res.converged || &sub == &subs.back())
          return record_step<Traits>(driver, res, s, c.tangent_every, sink, values, cut, std::move(summaries));
        driver.accept();
      }
    } catch (umat::fatal_error const &) {
      throw;
    } catch (std::runtime_error const &) {
      if (last_level)
        throw;
    }
    driver.point().restore(start);
  }
}

template <typename Traits, typename Sink>
bool run_small_strain(nlohmann::json const &doc, load_case<typename Traits::value_type> const &c, Sink &sink) {
  using T = typename Traits::value_type;
  using tensor2 = tmech::tensor<T, 3, 2>;
  case_model<Traits> model(doc);
  material_point_driver<Traits> driver{model.context(), detail::point_for<Traits>(doc, c)};
  property_columns<Traits> const columns{model.context(), parse_columns(doc)};
  sink.begin(columns.names());
  auto const values = [&](auto const &) { return columns.values(); };

  auto const initial{driver.step(tensor2{}, tensor2{}, load_control<T>::strain(), 0, 0)};
  if (!record_step<Traits>(driver, initial, c.path.unloaded(), 0, sink, values))
    return false;
  auto const solve{[&](load_step<T> const &s) {
    return driver.step(c.strain_at(s), tensor2{s.load * c.stress}, c.control, s.time, s.dt);
  }};
  for (auto const &s : c.path.steps())
    if (!cut_step<Traits>(driver, c, s, solve, sink, values))
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
  deformation_gradient_driver<Traits> driver{model.context(), detail::point_for<Traits>(doc, c)};
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
  auto const initial{driver.step(I, tensor2{}, deformation_control<T>::deformation(), 0, 0)};
  if (!record_step<Traits>(driver, initial, c.path.unloaded(), 0, sink, values))
    return false;
  auto const solve{[&](load_step<T> const &s) {
    return driver.step(tensor2{I + c.strain_at(s)}, tensor2{s.load * c.stress}, c.finite_control, s.time, s.dt);
  }};
  for (auto const &s : c.path.steps())
    if (!cut_step<Traits>(driver, c, s, solve, sink, values))
      return false;
  return true;
}

} // namespace detail

/// Runs one load case on a fresh material point; false if a step does not converge after the case's max_cutbacks
/// halvings (it is still recorded, then rejected). Every converged step is accepted after its columns and tangent
/// check. Small strain drives the strain, finite strain F = I + (interpolated points), and records P, Cauchy stress
/// and J. The driver's parameters are the document's "point" and the case's fd_step.
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
