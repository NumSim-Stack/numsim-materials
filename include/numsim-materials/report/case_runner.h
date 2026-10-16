#ifndef NUMSIM_MATERIALS_REPORT_CASE_RUNNER_H
#define NUMSIM_MATERIALS_REPORT_CASE_RUNNER_H

#include <memory>
#include <utility>

#include <nlohmann/json.hpp>
#include "numsim-materials/report/property_columns.h"
#include "numsim-materials/report/report_json.h"
#include "numsim-materials/report/step_record.h"
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

/// Runs one load case on a fresh material point; false if a step does not converge (it is still recorded).
template <typename Traits, typename Sink>
  requires step_sink<Sink, Traits>
bool run_case(nlohmann::json const &doc, load_case<typename Traits::value_type> const &c, Sink &sink,
              typename umat::material_point_driver<Traits>::options opts = {}) {
  using T = typename Traits::value_type;
  using tensor2 = typename umat::material_point_driver<Traits>::tensor2;
  material_point<Traits> point(doc);
  umat::material_point_driver<Traits> driver{point.evaluator(), opts};
  property_columns<Traits> const columns{point.context(), parse_columns(doc)};
  sink.begin(columns.names());

  auto const initial{driver.step(tensor2{}, tensor2{}, umat::load_control<T>::strain(), 0, 0)};
  sink.record({c.path.unloaded(), initial, columns.values()});

  for (auto const &s : c.path.steps()) {
    auto res{driver.step(c.strain_at(s), tensor2{s.load * c.stress}, c.control, s.time, s.dt)};
    step_record<Traits> rec{s, std::move(res), columns.values()}; // columns before the tangent check re-evaluates the graph
    if (!rec.result.converged) {
      sink.record(rec);
      return false;
    }
    if (c.tangent_every > 0 && s.index % c.tangent_every == 0)
      rec.tangent_error = driver.tangent_error(rec.result);
    sink.record(rec);
  }
  return true;
}

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_CASE_RUNNER_H
