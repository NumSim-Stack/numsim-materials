#include <gtest/gtest.h>

#include <locale>
#include <sstream>
#include <string>
#include <vector>

#include <tmech/tmech.h>

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/materials/scalar_stepper.h"
#include "numsim-materials/materials/tensor_component_stepper.h"
#include "numsim-materials/postprocessing/property_recorder.h"
#include "numsim-materials/default_materials.h"

namespace {

using policy = numsim::materials::material_policy_default;
using T = policy::value_type;
using ctx_type = numsim::materials::material_context<policy>;
using param_type = policy::ParameterHandler;
using tensor2 = tmech::tensor<T, 3, 2>;
using numsim::materials::csv_writer;
using numsim::materials::property_recorder;
using numsim::materials::vtk_timeseries_writer;

// Build: scalar_stepper (sca::state) + tensor_component_stepper (ten::strain)
// feeding a property_recorder, run `steps` updates, and capture the per-step
// expected values by reading the graph after each update.
struct Fixture {
  ctx_type ctx;
  property_recorder<policy>* rec = nullptr;
  std::vector<T> exp_scalar;
  std::vector<T> exp_s00; // strain(0,0)

  explicit Fixture(int steps) {
    param_type p;
    p.insert<std::string>("name", "sca");
    p.insert<T>("increment", T{0.5});
    ctx.create<numsim::materials::scalar_stepper<policy>>(p);

    p.clear();
    p.insert<std::string>("name", "ten");
    p.insert<T>("increment", T{0.1});
    p.insert<std::vector<std::size_t>>("indices", {0, 0});
    ctx.create<numsim::materials::tensor_component_stepper<2, policy>>(p);

    p.clear();
    p.insert<std::string>("name", "rec");
    p.insert<std::vector<std::string>>("scalar_sources", {"sca::state"});
    p.insert<std::vector<std::string>>("tensor_sources", {"ten::strain"});
    // ctx.create returns a reference to the created material (see property_plot).
    rec = &ctx.create<property_recorder<policy>>(p);

    ctx.finalize();
    for (int i = 0; i < steps; ++i) {
      ctx.update();
      exp_scalar.push_back(ctx.get<T>("sca", "state"));
      exp_s00.push_back(ctx.get<tensor2>("ten", "strain")(0, 0));
    }
  }

  property_recorder<policy>& recorder() { return *rec; }
};

TEST(PropertyRecorder, BuffersOneRowPerUpdateWithCorrectValues) {
  Fixture f(3);
  auto const& buf = f.recorder().buffer();

  EXPECT_EQ(buf.rows(), 3u);
  // 1 scalar column + 9 tensor component columns.
  EXPECT_EQ(buf.cols(), 10u);
  EXPECT_EQ(buf.name(0), "sca_state");
  EXPECT_EQ(buf.name(1), "ten_strain_00");
  EXPECT_EQ(buf.name(9), "ten_strain_22");

  for (std::size_t r = 0; r < 3; ++r) {
    EXPECT_NEAR(buf.at(r, 0), f.exp_scalar[r], 1e-12) << "row " << r;
    // strain(0,0) is component (i=0,j=0) → the first tensor column, index 1.
    EXPECT_NEAR(buf.at(r, 1), f.exp_s00[r], 1e-12) << "row " << r;
  }
}

TEST(PropertyRecorder, RejectsRecipeWithNoSources) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "rec");
  // both source lists default to empty
  EXPECT_THROW(ctx.create<property_recorder<policy>>(p), std::runtime_error);
}

TEST(PropertyRecorder, CsvWriterRoundTrips) {
  Fixture f(3);
  std::ostringstream os;
  csv_writer{}.write(f.recorder().buffer(), os);
  auto const& buf = f.recorder().buffer();

  std::istringstream is(os.str());
  std::string line;

  // Header: exactly the column names, comma-separated.
  ASSERT_TRUE(std::getline(is, line));
  {
    std::string expected;
    for (std::size_t c = 0; c < buf.cols(); ++c)
      expected += (c ? "," : "") + buf.name(c);
    EXPECT_EQ(line, expected);
  }
  // Rows: parse back the doubles and compare exactly (17-digit round-trip).
  for (std::size_t r = 0; r < buf.rows(); ++r) {
    ASSERT_TRUE(std::getline(is, line)) << "missing row " << r;
    std::istringstream ls(line);
    std::string cell;
    for (std::size_t c = 0; c < buf.cols(); ++c) {
      ASSERT_TRUE(std::getline(ls, cell, ','));
      EXPECT_DOUBLE_EQ(std::stod(cell), buf.at(r, c)) << "cell " << r << "," << c;
    }
  }
  EXPECT_FALSE(std::getline(is, line)) << "trailing content after last row";
}

// A source string with no alphanumerics, or two sources that sanitize to the
// same column name, must be rejected — otherwise CSV emits nameless/ambiguous
// columns and VTK emits shadowed (silently overwritten) point-data arrays.
TEST(PropertyRecorder, RejectsCollidingColumnNames) {
  ctx_type ctx;
  param_type p;
  p.insert<std::string>("name", "rec");
  // Two sources that map to the same sanitized column name ("a_b").
  p.insert<std::vector<std::string>>("scalar_sources", {"a::b", "a::b"});
  EXPECT_THROW(ctx.create<property_recorder<policy>>(p), std::runtime_error);
}

// H1/M1: number formatting must be independent of the stream's locale AND its
// sticky float flags. to_chars ignores both; the old `os << v` honored both and
// silently corrupted CSV/VTK (comma decimal separator / fixed-notation clipping).
TEST(PropertyRecorder, DoubleFormattingIgnoresLocaleAndFlags) {
  struct comma_numpunct : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
  };
  std::ostringstream os;
  os.imbue(std::locale(os.getloc(), new comma_numpunct));
  os << std::fixed; // sticky flag that would clip small magnitudes with <<
  numsim::materials::detail::write_double(os, 3.5);
  os << '|';
  numsim::materials::detail::write_double(os, 1e-20);
  // '.' decimal separator (not ','), and shortest round-trip (not fixed-clipped).
  EXPECT_EQ(os.str(), "3.5|1e-20");
}

TEST(PropertyRecorder, VtkWriterEmitsWellFormedPolyData) {
  Fixture f(3);
  std::ostringstream os;
  vtk_timeseries_writer{}.write(f.recorder().buffer(), os);
  auto const src = os.str();

  EXPECT_NE(src.find("# vtk DataFile Version"), std::string::npos) << src;
  EXPECT_NE(src.find("ASCII"), std::string::npos) << src;
  EXPECT_NE(src.find("DATASET POLYDATA"), std::string::npos) << src;
  // One VTK point per recorded row.
  EXPECT_NE(src.find("POINTS 3 double"), std::string::npos) << src;
  EXPECT_NE(src.find("POINT_DATA 3"), std::string::npos) << src;
  // Every column becomes a named scalar array.
  EXPECT_NE(src.find("SCALARS sca_state double 1"), std::string::npos) << src;
  EXPECT_NE(src.find("SCALARS ten_strain_00 double 1"), std::string::npos)
      << src;
  EXPECT_NE(src.find("SCALARS ten_strain_22 double 1"), std::string::npos)
      << src;
}

} // namespace
