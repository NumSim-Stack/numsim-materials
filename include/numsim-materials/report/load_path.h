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
  int index{0};           ///< 0 is the unloaded state
  T time{0};              ///< at the start of the step
  T dt{0};
  T load{0};              ///< load factor at the end of the step
  std::size_t segment{0}; ///< between points[segment] and points[segment + 1]
  T fraction{0};          ///< position in the segment at the end of the step, (0, 1]
};

/// Step schedule along a piecewise linear path: `steps_per_segment` steps between consecutive
/// points; `factors` (one per point, times `scale`) give the load factor of a step.
template <typename T>
class load_path {
public:
  load_path(std::vector<T> factors, int steps_per_segment, T scale = 1, T dt = 1)
      : m_points(std::move(factors)), m_steps(steps_per_segment), m_scale(scale), m_dt(dt) {
    if (m_points.size() < 2 || m_steps < 1)
      throw std::invalid_argument("load path: needs two points and at least one step per segment");
  }

  [[nodiscard]] std::size_t size() const noexcept { return (m_points.size() - 1) * static_cast<std::size_t>(m_steps); }

  [[nodiscard]] load_step<T> unloaded() const noexcept { return {0, 0, 0, m_scale * m_points.front()}; }

  [[nodiscard]] load_step<T> at(int index) const {
    if (index < 1 || index > static_cast<int>(size()))
      throw std::out_of_range("load path: step " + std::to_string(index) + " of " + std::to_string(size()));
    auto const seg{static_cast<std::size_t>((index - 1) / m_steps)};
    auto const k{T((index - 1) % m_steps + 1) / T(m_steps)};
    return {index, (index - 1) * m_dt, m_dt, m_scale * (m_points[seg] + (m_points[seg + 1] - m_points[seg]) * k), seg, k};
  }

  /// Step `s` in `count` equal substeps from the end of the step before; the last one ends where `s` does.
  [[nodiscard]] std::vector<load_step<T>> substeps(load_step<T> const &s, int count) const {
    auto const before{s.index > 1 ? at(s.index - 1) : unloaded()};
    auto const f0{before.segment == s.segment ? before.fraction : T{0}};
    std::vector<load_step<T>> out;
    for (int k{1}; k <= count; ++k) {
      auto const f{k == count ? s.fraction : f0 + (s.fraction - f0) * T(k) / T(count)};
      auto const load{k == count ? s.load
                                 : m_scale * (m_points[s.segment] + (m_points[s.segment + 1] - m_points[s.segment]) * f)};
      out.push_back({s.index, s.time + s.dt * T(k - 1) / T(count), s.dt / T(count), load, s.segment, f});
    }
    return out;
  }

  /// Steps 1 ... size() as a view.
  [[nodiscard]] auto steps() const noexcept {
    return std::views::iota(1, static_cast<int>(size()) + 1) | std::views::transform([this](int i) { return at(i); });
  }

private:
  std::vector<T> m_points;
  int m_steps;
  T m_scale, m_dt;
};

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_LOAD_PATH_H
