#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <tmech/tmech.h>
#include <sstream>
#include "numsim-materials/report/case_runner.h"
#include "numsim-materials/report/csv_sink.h"
#include "numsim-materials/report/step_record.h"
#include "numsim-materials/report/graph_description.h"
#include "numsim-materials/umat/load_control.h"
#include "numsim-materials/report/load_path.h"
#include "numsim-materials/report/report_json.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
namespace r = numsim::materials::report;

using policy = nm::material_policy_default;
using T = policy::value_type;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using control = nm::umat::load_control<T>;
using json = nlohmann::json;

TEST(LoadControl, StrainControlPrescribesNoStress) {
  EXPECT_EQ(tmech::norm(control::strain().stress_projector()), 0);
}

TEST(LoadControl, UniaxialStressFreesAllButEleven) {
  auto const I{tmech::eye<T, 3, 2>()};
  tensor2 const free{tmech::dcontract(control::uniaxial_stress().stress_projector(), I)};
  EXPECT_DOUBLE_EQ(free(0, 0), 0);
  EXPECT_DOUBLE_EQ(free(1, 1), 1);
  EXPECT_DOUBLE_EQ(free(2, 2), 1);
}

TEST(LoadControl, MixedSelectsShearComponents) {
  tensor2 g;
  g(0, 1) = g(1, 0) = 1;
  tensor2 const out{tmech::dcontract(control::mixed({"12"}).stress_projector(), g)};
  EXPECT_DOUBLE_EQ(out(0, 1), 1);
  EXPECT_DOUBLE_EQ(out(1, 0), 1);
  EXPECT_THROW(control::mixed({"14"}), std::invalid_argument);
}

TEST(LoadPath, InterpolatesSegmentsWithScaleAndTime) {
  r::load_path<T> const path{{0, 1, -1}, 2, 0.5, 0.1};
  ASSERT_EQ(path.size(), 4u);
  std::vector<T> loads, ends;
  for (auto const &s : path.steps()) {
    loads.push_back(s.load);
    ends.push_back(s.time + s.dt);
  }
  EXPECT_EQ(loads, (std::vector<T>{0.25, 0.5, 0, -0.5}));
  EXPECT_NEAR(ends.back(), 0.4, 1e-15);
  EXPECT_EQ(path.steps().front().index, 1);
}

TEST(LoadPath, RejectsStepsOutsideThePath) {
  r::load_path<T> const path{{0, 1}, 2};
  EXPECT_THROW((void)path.at(0), std::out_of_range);
  EXPECT_THROW((void)path.at(3), std::out_of_range);
  EXPECT_NO_THROW((void)path.at(2));
}

TEST(ReportJson, ParsesThePointAndTheColumns) {
  auto const doc = json::parse(R"({"point": {"strain_source": "s", "stress_source": "m", "tangent_source": "t"},
    "record": {"kappa": "m::kappa"}})");
  auto const cfg{r::parse_point<policy>(doc)};
  EXPECT_EQ(cfg.strain_source, "s");
  EXPECT_EQ(cfg.stress_source, "m");
  ASSERT_TRUE(cfg.tangent_source.has_value());
  EXPECT_EQ(*cfg.tangent_source, "t");
  auto const cols{r::parse_columns(doc)};
  ASSERT_EQ(cols.size(), 3u);
  EXPECT_EQ(cols[0], (std::pair<std::string, std::string>{"strain", "s::strain"}));
  EXPECT_EQ(cols[1], (std::pair<std::string, std::string>{"stress", "m::stress"}));
  EXPECT_EQ(cols[2], (std::pair<std::string, std::string>{"kappa", "m::kappa"}));
}

TEST(ReportJson, ParsesALoadCase) {
  auto const c{r::parse_case<T>(json::parse(R"({"name": "u", "control": "mixed", "stress_components": ["22", "33"],
    "direction": [1, 0, 0, 0.5, 0, 0], "path": [0, 1], "steps": 4, "scale": 0.01})"))};
  EXPECT_EQ(c.name, "u");
  EXPECT_EQ(c.path.size(), 4u);
  EXPECT_DOUBLE_EQ(c.direction(0, 1), 0.5);
  EXPECT_DOUBLE_EQ(c.direction(1, 0), 0.5);
  tensor2 const free{tmech::dcontract(c.control.stress_projector(), tmech::eye<T, 3, 2>())};
  EXPECT_DOUBLE_EQ(free(1, 1), 1);
  EXPECT_DOUBLE_EQ(free(0, 0), 0);
  EXPECT_THROW(r::parse_case<T>(json::parse(R"({"name": "u", "direction": [1, 0, 0, 0, 0, 0]})")), std::exception);
  EXPECT_THROW(r::parse_case<T>(json::parse(R"({"name": "u", "control": "x", "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1]})")),
               std::invalid_argument);
}

struct recording_sink {
  std::vector<std::string> names;
  std::vector<r::step_record<policy>> records;
  void begin(std::vector<std::string> const &n) { names = n; }
  void record(r::step_record<policy> const &rec) { records.push_back(rec); }
};

json elastic_report() {
  return json::parse(R"({
    "model": {"materials": [
      {"type": "external_strain_source", "name": "strain"},
      {"type": "linear_elasticity", "name": "elastic", "strain_producer_name": "strain", "K": 100.0, "G": 50.0}]},
    "point": {"strain_source": "strain", "stress_source": "elastic"},
    "record": {"sig": "elastic::stress[1,1]"}})");
}

TEST(CaseRunner, RecordsTheUnloadedStateAndEveryStep) {
  auto const doc = elastic_report();
  auto const c{r::parse_case<T>(json::parse(R"({"name": "u", "control": "uniaxial_stress",
    "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1], "steps": 3, "scale": 0.003})"))};
  recording_sink sink;
  EXPECT_TRUE(r::run_case<policy>(doc, c, sink));
  ASSERT_EQ(sink.records.size(), 4u);
  EXPECT_EQ(sink.records.front().step.index, 0);
  EXPECT_EQ(sink.names.front(), "strain_11");
  EXPECT_EQ(sink.names.back(), "sig");
  auto const &last{sink.records.back()};
  EXPECT_TRUE(last.result.converged);
  EXPECT_DOUBLE_EQ(last.columns.back(), last.result.stress(0, 0));
  EXPECT_LT(last.tangent_error, 1e-6);
}

TEST(CaseRunner, StopsAtAStepThatDoesNotConverge) {
  auto const doc = elastic_report();
  auto const c{r::parse_case<T>(json::parse(R"({"name": "u", "control": "uniaxial_stress",
    "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1], "steps": 3})"))};
  recording_sink sink;
  EXPECT_FALSE(r::run_case<policy>(doc, c, sink, {.tolerance = 1e-10, .max_iter = 0}));
  ASSERT_EQ(sink.records.size(), 2u);
  EXPECT_FALSE(sink.records.back().result.converged);
}

} // namespace

namespace {

TEST(CsvSink, WritesHeaderRowsAndResiduals) {
  std::ostringstream csv, its;
  r::csv_sink<policy> sink{csv, its};
  sink.begin({"strain_11", "kappa"});
  r::step_record<policy> rec{{1, 0, 1, 0.5}, {}, {1e-3, 2.0}, 1e-9};
  rec.result.converged = true;
  rec.result.residuals = {1.0, 1e-12};
  sink.record(rec);
  EXPECT_EQ(csv.str(), "step,time,load,iterations,converged,tangent_error,strain_11,kappa\n1,1,0.5,1,1,1.0000000000000001e-09,0.001,2\n");
  EXPECT_EQ(its.str(), "step,iteration,residual\n1,0,1\n1,1,1e-12\n");
}

TEST(GraphDescription, ListsMaterialsParametersAndWiring) {
  auto const doc = elastic_report();
  auto const g = r::describe_graph<policy>(doc);
  ASSERT_EQ(g.size(), 2u);
  EXPECT_EQ(g[1]["name"], "elastic");
  EXPECT_EQ(g[1]["inputs"][0]["source"], "strain::strain");
  bool has_K{false};
  for (auto const &p : g[1]["parameters"])
    has_K = has_K || (p["name"] == "K" && p["value"] == 100.0);
  EXPECT_TRUE(has_K);
}

} // namespace
