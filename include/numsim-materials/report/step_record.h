#ifndef NUMSIM_MATERIALS_REPORT_STEP_RECORD_H
#define NUMSIM_MATERIALS_REPORT_STEP_RECORD_H

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "numsim-materials/report/load_path.h"

namespace numsim::materials::report {

/// The columns of a step before the property columns, as csv_sink writes them.
inline constexpr std::array<std::string_view, 6> step_columns{"step", "time", "load", "iterations", "converged",
                                                              "tangent_error"};

/// The columns reports/numsim_report/data.py adds: pressure and von Mises stress.
inline constexpr std::array<std::string_view, 2> derived_columns{"p", "q"};

/// `names` distinct and none of step_columns or derived_columns, or an invalid_argument naming the column.
inline void require_unique_columns(std::vector<std::string> const &names) {
  std::vector<std::string_view> seen;
  for (auto const &n : names) {
    std::string_view const name{n};
    if (std::ranges::contains(step_columns, name) || std::ranges::contains(derived_columns, name))
      throw std::invalid_argument("report: the column name '" + n + "' is reserved (step, time, load, iterations, "
                                  "converged, tangent_error, p, q)");
    if (std::ranges::contains(seen, name))
      throw std::invalid_argument("report: two columns are named '" + n + "'");
    seen.push_back(name);
  }
}

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
