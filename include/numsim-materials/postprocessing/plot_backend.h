#ifndef NUMSIM_MATERIALS_PLOT_BACKEND_H
#define NUMSIM_MATERIALS_PLOT_BACKEND_H

#include <string>

namespace numsim::materials {

/// Abstract plotting backend. Implementations provide the actual rendering
/// (QCustomPlot, gnuplot, file output, etc.).
class plot_backend {
public:
  virtual ~plot_backend() = default;

  /// Add a named data series. Returns a series index.
  virtual int add_series(const std::string& name) = 0;

  /// Append a data point to a series.
  virtual void append(int series, double x, double y) = 0;

  /// Show the plot window (if applicable).
  virtual void show() = 0;

  /// Refresh the display. Called after each batch of append() calls.
  virtual void refresh() = 0;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_PLOT_BACKEND_H
