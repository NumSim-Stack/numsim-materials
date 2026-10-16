#ifndef NUMSIM_MATERIALS_REPORT_STEP_RECORD_H
#define NUMSIM_MATERIALS_REPORT_STEP_RECORD_H

#include <limits>
#include <string>
#include <vector>

#include "numsim-materials/report/load_path.h"
#include "numsim-materials/umat/material_point_driver.h"

namespace numsim::materials::report {

/// One row of a load case: the step, the driver's result and the wired columns.
template <typename Traits>
struct step_record {
  using value_type = typename Traits::value_type;
  load_step<value_type> step;
  typename umat::material_point_driver<Traits>::step_result result;
  std::vector<value_type> columns;
  value_type tangent_error{std::numeric_limits<value_type>::quiet_NaN()};
};

/// Receives the column names once, then one record per step.
template <typename S, typename Traits>
concept step_sink = requires(S &s, std::vector<std::string> const &names, step_record<Traits> const &rec) {
  s.begin(names);
  s.record(rec);
};

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_STEP_RECORD_H
