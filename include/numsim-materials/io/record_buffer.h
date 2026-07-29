#ifndef NUMSIM_MATERIALS_RECORD_BUFFER_H
#define NUMSIM_MATERIALS_RECORD_BUFFER_H

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

namespace numsim::materials {

/// Format-agnostic, column-major record of scalar values captured over a run.
///
/// A post-processor (see postprocessing/property_recorder.h) declares one
/// column per recorded scalar component up front, then appends one row per
/// step. Column-major storage means each column IS a contiguous array — the
/// natural shape for a CSV column or a VTK DataArray, so writers consume it
/// with no repacking.
///
/// This type deliberately knows nothing about tmech / properties / the graph:
/// tensor sources are flattened to component columns (e.g. `stress_00`,
/// `stress_01`, …) by the recorder before they reach the buffer.
class record_buffer {
public:
  /// Declare a column. All columns must be declared before the first row.
  void declare_column(std::string name) {
    m_names.push_back(std::move(name));
    m_columns.emplace_back();
  }

  /// Append one value to column `c`. Call once per column per row, in order.
  void push(std::size_t c, double value) { m_columns[c].push_back(value); }

  [[nodiscard]] std::size_t cols() const noexcept { return m_names.size(); }
  [[nodiscard]] std::size_t rows() const noexcept {
    return m_columns.empty() ? 0 : m_columns.front().size();
  }
  [[nodiscard]] std::string const& name(std::size_t c) const {
    return m_names[c];
  }
  /// The whole column `c` (size == rows()).
  [[nodiscard]] std::vector<double> const& column(std::size_t c) const {
    return m_columns[c];
  }
  [[nodiscard]] double at(std::size_t row, std::size_t col) const {
    return m_columns[col][row];
  }

private:
  std::vector<std::string> m_names;
  std::vector<std::vector<double>> m_columns; // m_columns[col][row]
};

/// Pluggable serialization backend for a record_buffer — the file-output analog
/// of `postprocessing/plot_backend`. Concrete writers (CSV, VTK, …) serialize
/// the same buffer to different formats; the recorder is written once against
/// this interface.
class output_writer {
public:
  virtual ~output_writer() = default;
  virtual void write(record_buffer const& buffer, std::ostream& os) const = 0;
};

/// Comma-separated values: a header row of column names, then one row per step.
/// Row index is not emitted as a column — add an explicit source for it if
/// wanted. Full double precision so a round-trip is lossless.
class csv_writer final : public output_writer {
public:
  void write(record_buffer const& buffer, std::ostream& os) const override {
    auto const cols = buffer.cols();
    for (std::size_t c = 0; c < cols; ++c) {
      if (c) os << ',';
      os << buffer.name(c);
    }
    os << '\n';
    auto const rows = buffer.rows();
    for (std::size_t r = 0; r < rows; ++r) {
      for (std::size_t c = 0; c < cols; ++c) {
        if (c) os << ',';
        write_double(os, buffer.at(r, c));
      }
      os << '\n';
    }
  }

private:
  static void write_double(std::ostream& os, double v) {
    // 17 significant digits round-trips an IEEE-754 double exactly.
    auto const prec = os.precision();
    os.precision(17);
    os << v;
    os.precision(prec);
  }
};

/// Legacy-ASCII VTK PolyData time series: N points laid out along the x-axis at
/// (step, 0, 0), with every recorded column attached as a scalar POINT_DATA
/// array. Opens in ParaView as a poly-line whose point-data are the recorded
/// histories — the natural view for a single-material-point driver. (Spatial /
/// mesh-based VTK, where points come from an external mesh, is a later path.)
class vtk_timeseries_writer final : public output_writer {
public:
  void write(record_buffer const& buffer, std::ostream& os) const override {
    auto const rows = buffer.rows();
    auto const cols = buffer.cols();

    os << "# vtk DataFile Version 3.0\n";
    os << "numsim-materials property_recorder time series\n";
    os << "ASCII\n";
    os << "DATASET POLYDATA\n";
    os << "POINTS " << rows << " double\n";
    for (std::size_t r = 0; r < rows; ++r) {
      write_double(os, static_cast<double>(r));
      os << " 0 0\n";
    }
    // One line cell threading the points, so ParaView draws a curve.
    if (rows > 0) {
      os << "LINES 1 " << (rows + 1) << "\n" << rows;
      for (std::size_t r = 0; r < rows; ++r) os << ' ' << r;
      os << '\n';
    }
    os << "POINT_DATA " << rows << "\n";
    for (std::size_t c = 0; c < cols; ++c) {
      os << "SCALARS " << buffer.name(c) << " double 1\n";
      os << "LOOKUP_TABLE default\n";
      for (std::size_t r = 0; r < rows; ++r) {
        write_double(os, buffer.at(r, c));
        os << '\n';
      }
    }
  }

private:
  static void write_double(std::ostream& os, double v) {
    auto const prec = os.precision();
    os.precision(17);
    os << v;
    os.precision(prec);
  }
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_RECORD_BUFFER_H
