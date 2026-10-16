#ifndef NUMSIM_MATERIALS_REPORT_LOAD_PATH_H
#define NUMSIM_MATERIALS_REPORT_LOAD_PATH_H

#include <cstddef>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace numsim::materials::report {

template <typename T>
struct load_step {
  int index{0}; ///< 0 is the unloaded state
  T time{0};    ///< at the start of the step
  T dt{0};
  T load{0};
};

/// Piecewise linear load factor: `steps_per_segment` steps between consecutive points, times `scale`.
template <typename T>
class load_path {
public:
  load_path(std::vector<T> points, int steps_per_segment, T scale = 1, T dt = 1)
      : m_points(std::move(points)), m_steps(steps_per_segment), m_scale(scale), m_dt(dt) {
    if (m_points.size() < 2 || m_steps < 1)
      throw std::invalid_argument("load path: needs two points and at least one step per segment");
  }

  [[nodiscard]] std::size_t size() const noexcept { return (m_points.size() - 1) * static_cast<std::size_t>(m_steps); }

  [[nodiscard]] load_step<T> unloaded() const { return {0, 0, 0, m_scale * m_points.front()}; }

  [[nodiscard]] load_step<T> at(int index) const {
    if (index < 1 || index > static_cast<int>(size()))
      throw std::out_of_range("load path: step " + std::to_string(index) + " of " + std::to_string(size()));
    auto const seg{static_cast<std::size_t>((index - 1) / m_steps)};
    auto const k{T((index - 1) % m_steps + 1) / T(m_steps)};
    return {index, (index - 1) * m_dt, m_dt, m_scale * (m_points[seg] + (m_points[seg + 1] - m_points[seg]) * k)};
  }

  /// Steps 1 ... size() as a view.
  [[nodiscard]] auto steps() const {
    return std::views::iota(1, static_cast<int>(size()) + 1) | std::views::transform([this](int i) { return at(i); });
  }

private:
  std::vector<T> m_points;
  int m_steps;
  T m_scale, m_dt;
};

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_LOAD_PATH_H
