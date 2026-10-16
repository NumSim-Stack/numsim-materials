#ifndef NUMSIM_MATERIALS_REPORT_CSV_SINK_H
#define NUMSIM_MATERIALS_REPORT_CSV_SINK_H

#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "numsim-materials/report/step_record.h"

namespace numsim::materials::report {

/// step_sink writing one CSV row per step and the Newton residuals to a second stream.
template <typename Traits>
class csv_sink {
public:
  csv_sink(std::ostream &steps, std::ostream &residuals) : m_steps(steps), m_residuals(residuals) {
    m_steps.precision(std::numeric_limits<typename Traits::value_type>::max_digits10);
    m_residuals.precision(6);
  }

  void begin(std::vector<std::string> const &names) {
    m_steps << "step,time,load,iterations,converged,tangent_error";
    for (auto const &n : names)
      m_steps << "," << n;
    m_steps << "\n";
    m_residuals << "step,iteration,residual\n";
  }

  void record(step_record<Traits> const &rec) {
    auto const &res{rec.result};
    m_steps << rec.step.index << "," << rec.step.time + rec.step.dt << "," << rec.step.load << ","
            << (res.residuals.empty() ? 0 : res.residuals.size() - 1) << "," << res.converged << "," << rec.tangent_error;
    for (auto const v : rec.columns)
      m_steps << "," << v;
    m_steps << "\n";
    for (std::size_t i{0}; i < res.residuals.size(); ++i)
      m_residuals << rec.step.index << "," << i << "," << res.residuals[i] << "\n";
  }

private:
  std::ostream &m_steps;
  std::ostream &m_residuals;
};

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_CSV_SINK_H
