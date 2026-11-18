#ifndef NUMSIM_MATERIALS_REPORT_STEP_RECORD_H
#define NUMSIM_MATERIALS_REPORT_STEP_RECORD_H

#include <limits>
#include <string>
#include <vector>

#include "numsim-materials/report/load_path.h"

namespace numsim::materials::report {

/// What a sink needs of a driver's step: its Newton history.
template <typename T>
struct step_summary {
  std::vector<T> residuals{};
  bool converged{false};

  template <typename Result>
  static step_summary of(Result const &r) {
    return {r.residuals, r.converged};
  }
};

/// One row of a load case: the step, the Newton summary and the columns. A step cut back ran in 2^cutbacks
/// substeps; `result` is the last one's summary.
template <typename Traits>
struct step_record {
  using value_type = typename Traits::value_type;
  load_step<value_type> step;
  step_summary<value_type> result;
  std::vector<value_type> columns;
  value_type tangent_error{std::numeric_limits<value_type>::quiet_NaN()};
  int cutbacks{0};
  std::vector<step_summary<value_type>> substeps{}; ///< the substeps run at that level, in order
};

/// Receives the column names once, then one record per step.
template <typename S, typename Traits>
concept step_sink = requires(S &s, std::vector<std::string> const &names, step_record<Traits> const &rec) {
  s.begin(names);
  s.record(rec);
};

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_STEP_RECORD_H
