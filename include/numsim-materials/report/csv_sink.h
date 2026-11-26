#ifndef NUMSIM_MATERIALS_REPORT_CSV_SINK_H
#define NUMSIM_MATERIALS_REPORT_CSV_SINK_H

#include <array>
#include <charconv>
#include <cstddef>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include "numsim-materials/report/step_record.h"

namespace numsim::materials::report {

namespace detail {
/// `v` as C would print it, whatever the stream's locale and precision: the shortest form that reads back
/// exactly, or `digits` significant digits; nan, inf and -inf as such.
template <typename V>
std::string csv_field(V v, int digits = 0) {
  std::array<char, 64> buf{};
  std::to_chars_result res{};
  if constexpr (std::is_floating_point_v<V>)
    res = digits > 0 ? std::to_chars(buf.data(), buf.data() + buf.size(), v, std::chars_format::general, digits)
                     : std::to_chars(buf.data(), buf.data() + buf.size(), v);
  else
    res = std::to_chars(buf.data(), buf.data() + buf.size(), v);
  return std::string(buf.data(), res.ptr);
}
} // namespace detail

/// step_sink writing one CSV row per step and the Newton residuals (six digits) to a second stream. The
/// columns are step_columns, then the property columns: time is the end of the step (0 for step 0),
/// iterations the Newton corrections, converged 1 or 0, tangent_error nan where it was not checked.
/// Numbers do not depend on the streams' locale or precision, which are left as they are.
template <typename Traits>
class csv_sink {
public:
  csv_sink(std::ostream &steps, std::ostream &residuals) : m_steps(steps), m_residuals(residuals) {}

  void begin(std::vector<std::string> const &names) {
    for (auto const &n : step_columns)
      m_steps << n << (&n == &step_columns.back() ? "" : ",");
    for (auto const &n : names)
      m_steps << "," << n;
    m_steps << "\n";
    m_residuals << "step,iteration,residual\n";
  }

  void record(step_record<Traits> const &rec) {
    using detail::csv_field;
    auto const &res{rec.result};
    m_steps << csv_field(rec.step.index) << "," << csv_field(rec.step.time + rec.step.dt) << "," << csv_field(rec.step.load)
            << "," << csv_field(res.residuals.empty() ? std::size_t{0} : res.residuals.size() - 1) << ","
            << (res.converged ? "1" : "0") << "," << csv_field(rec.tangent_error);
    for (auto const v : rec.columns)
      m_steps << "," << csv_field(v);
    m_steps << "\n";
    for (std::size_t i{0}; i < res.residuals.size(); ++i)
      m_residuals << csv_field(rec.step.index) << "," << csv_field(i) << "," << csv_field(res.residuals[i], 6) << "\n";
  }

private:
  std::ostream &m_steps;
  std::ostream &m_residuals;
};

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_CSV_SINK_H
