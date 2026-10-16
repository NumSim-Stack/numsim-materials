#ifndef NUMSIM_MATERIALS_REPORT_CASE_RUNNER_H
#define NUMSIM_MATERIALS_REPORT_CASE_RUNNER_H

#include <concepts>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "numsim-materials/report/load_driver.h"
#include "numsim-materials/report/property_columns.h"
#include "numsim-materials/report/report_json.h"
#include "numsim-materials/umat/json_model.h"

namespace numsim::materials::report {

/// One row of a load case: the step, the driver's result and the wired columns.
template <typename Traits>
struct step_record {
  using value_type = typename Traits::value_type;
  load_step<value_type> step;
  typename load_driver<Traits>::step_result result;
  std::vector<value_type> columns;
  value_type tangent_error{std::numeric_limits<value_type>::quiet_NaN()};
};

/// Receives the column names once, then one record per step.
template <typename S, typename Traits>
concept step_sink = requires(S &s, std::vector<std::string> const &names, step_record<Traits> const &rec) {
  s.begin(names);
  s.record(rec);
};

/// Material graph and evaluator of a report document ("model" and "point").
template <typename Traits>
class material_point {
public:
  explicit material_point(nlohmann::json const &doc) {
    umat::make_json_builder<Traits>(doc.at("model").dump())(m_ctx, {});
    auto const &p{doc.at("point")};
    typename evaluator::config cfg;
    cfg.strain_source = p.at("strain_source").template get<std::string>();
    cfg.stress_source = p.at("stress_source").template get<std::string>();
    if (p.contains("tangent_source"))
      cfg.tangent_source = p["tangent_source"].template get<std::string>();
    cfg.time_source = p.value("time_source", std::string{});
    m_ev = std::make_unique<evaluator>(m_ctx, std::move(cfg));
  }

  /// Strain and stress of the point first, then the "record" columns.
  [[nodiscard]] static std::vector<std::pair<std::string, std::string>> wiring(nlohmann::json const &doc) {
    auto const &p{doc.at("point")};
    std::vector<std::pair<std::string, std::string>> w{{"strain", p.at("strain_source").template get<std::string>() + "::strain"},
                                                       {"stress", p.at("stress_source").template get<std::string>() + "::stress"}};
    for (auto &entry : parse_record(doc))
      w.push_back(std::move(entry));
    return w;
  }

  [[nodiscard]] material_context<Traits> &context() noexcept { return m_ctx; }
  [[nodiscard]] umat::material_point_evaluator<Traits> &evaluator_ref() noexcept { return *m_ev; }

private:
  using evaluator = umat::material_point_evaluator<Traits>;

  material_context<Traits> m_ctx;
  std::unique_ptr<evaluator> m_ev;
};

/// Runs one load case on a fresh material point; false if a step does not converge (it is still recorded).
template <typename Traits, typename Sink>
  requires step_sink<Sink, Traits>
bool run_case(nlohmann::json const &doc, load_case<typename Traits::value_type> const &c, Sink &sink,
              typename load_driver<Traits>::options opts = {}) {
  using T = typename Traits::value_type;
  using tensor2 = typename load_driver<Traits>::tensor2;
  material_point<Traits> point(doc);
  load_driver<Traits> driver{point.evaluator_ref(), opts};
  property_columns<Traits> const columns{point.context(), point.wiring(doc)};
  sink.begin(columns.names());

  auto const unloaded{c.path.unloaded()};
  auto const initial{driver.step(tensor2{}, tensor2{}, load_control<T>::strain(), 0, 0)};
  sink.record({unloaded, initial, columns.values()});

  for (auto const &s : c.path.steps()) {
    auto res{driver.step(tensor2{s.load * c.direction}, tensor2{s.load * c.stress}, c.control, s.time, s.dt)};
    step_record<Traits> rec{s, std::move(res), columns.values()}; // columns before the tangent check re-evaluates the graph
    if (!rec.result.converged) {
      sink.record(rec);
      return false;
    }
    rec.tangent_error = driver.tangent_error(rec.result);
    sink.record(rec);
  }
  return true;
}

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_CASE_RUNNER_H
