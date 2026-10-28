#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <tmech/tmech.h>
#include <sstream>
#include "numsim-materials/report/case_runner.h"
#include "numsim-materials/report/csv_sink.h"
#include "numsim-materials/report/step_record.h"
#include "numsim-materials/report/graph_description.h"
#include "numsim-materials/driver/load_control.h"
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
using control = nm::load_control<T>;
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

TEST(LoadPath, StepsKnowTheirSegmentAndFraction) {
  r::load_path<T> const path{{0, 1, 0}, 2};
  auto const s3{path.at(3)};
  EXPECT_EQ(s3.segment, 1u);
  EXPECT_DOUBLE_EQ(s3.fraction, 0.5);
  EXPECT_DOUBLE_EQ(s3.load, 0.5);
}

TEST(ReportJson, ParsesANonProportionalCaseFromPoints) {
  auto const c{r::parse_case<T>(json::parse(R"({"name": "ts", "points": [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0]],
    "scale": 0.01, "steps": 10})"))};
  ASSERT_EQ(c.points.size(), 3u);
  EXPECT_DOUBLE_EQ(c.points[2](0, 0), 0.01);
  EXPECT_DOUBLE_EQ(c.points[2](0, 1), 0.01);
  EXPECT_EQ(c.path.size(), 20u);
  EXPECT_DOUBLE_EQ(c.path.at(15).fraction, 0.5);
  EXPECT_DOUBLE_EQ(c.strain_at(c.path.at(15))(0, 1), 0.005);
  EXPECT_DOUBLE_EQ(c.strain_at(c.path.at(15))(0, 0), 0.01);
  EXPECT_THROW(r::parse_case<T>(json::parse(R"({"name": "x", "points": [[0, 0, 0, 0, 0, 0]], "path": [0, 1]})")), std::invalid_argument);
}

TEST(ReportJson, ADirectionWithAPathExpandsToPoints) {
  auto const c{r::parse_case<T>(json::parse(R"({"name": "u", "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1, -1], "scale": 0.01, "steps": 4})"))};
  ASSERT_EQ(c.points.size(), 3u);
  EXPECT_DOUBLE_EQ(c.points[2](0, 0), -0.01);
  EXPECT_DOUBLE_EQ(c.strain_at(c.path.at(6))(0, 0), 0.0);
}

TEST(LoadPath, SubstepsSplitAStepFromTheEndOfTheOneBefore) {
  r::load_path<T> const path{{0, 1, -1}, 2, 0.5, 0.1};
  auto const s3{path.at(3)}; // first step of the second segment: from load 0.5 to 0
  auto const subs{path.substeps(s3, 4)};
  ASSERT_EQ(subs.size(), 4u);
  std::vector<T> loads, fractions;
  for (auto const &sub : subs) {
    EXPECT_EQ(sub.index, 3);
    EXPECT_EQ(sub.segment, 1u);
    EXPECT_NEAR(sub.dt, 0.025, 1e-15);
    loads.push_back(sub.load);
    fractions.push_back(sub.fraction);
  }
  EXPECT_EQ(fractions, (std::vector<T>{0.125, 0.25, 0.375, 0.5}));
  EXPECT_EQ(loads, (std::vector<T>{0.375, 0.25, 0.125, 0}));
  EXPECT_NEAR(subs.front().time, s3.time, 1e-15);
  EXPECT_NEAR(subs.back().time + subs.back().dt, s3.time + s3.dt, 1e-15);
  EXPECT_EQ(path.substeps(path.at(1), 2).front().load, 0.125) << "the first step starts unloaded";
  auto const whole{path.substeps(s3, 1)};
  ASSERT_EQ(whole.size(), 1u);
  EXPECT_EQ(whole[0].load, s3.load);
  EXPECT_EQ(whole[0].fraction, s3.fraction);
  EXPECT_EQ(whole[0].time, s3.time);
  EXPECT_EQ(whole[0].dt, s3.dt);
}

TEST(ReportJson, ParsesMaxCutbacks) {
  auto const with = [](std::string const &extra) {
    return json::parse(R"({"name": "u", "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1])" + extra + "}");
  };
  EXPECT_EQ(r::parse_case<T>(with("")).max_cutbacks, 0) << "no cutbacks by default";
  EXPECT_EQ(r::parse_case<T>(with(R"(, "max_cutbacks": 3)")).max_cutbacks, 3);
  EXPECT_THROW(r::parse_case<T>(with(R"(, "max_cutbacks": -1)")), std::invalid_argument);
  EXPECT_THROW(r::parse_case<T>(with(R"(, "max_cutbacks": 1.5)")), std::invalid_argument);
  EXPECT_THROW(r::parse_case<T>(with(R"(, "max_cutbacks": 21)")), std::invalid_argument);
}

TEST(LoadPath, RejectsStepsOutsideThePath) {
  r::load_path<T> const path{{0, 1}, 2};
  EXPECT_THROW((void)path.at(0), std::out_of_range);
  EXPECT_THROW((void)path.at(3), std::out_of_range);
  EXPECT_NO_THROW((void)path.at(2));
}

TEST(ReportJson, ParsesThePointAndTheColumns) {
  auto const doc = json::parse(R"({"point": {"strain_source": "s", "stress_source": "m", "tangent_source": "t",
    "stress_property": "damaged_stress", "tangent_property": "damaged_tangent"}, "record": {"kappa": "m::kappa"}})");
  auto const p{r::parse_point<policy>(doc)};
  EXPECT_EQ(p.get<std::string>("strain_source"), "s");
  EXPECT_EQ(p.get<std::string>("stress_source"), "m");
  EXPECT_EQ(p.get<std::string>("tangent_source"), "t");
  EXPECT_EQ(p.get<std::string>("stress_property"), "damaged_stress");
  EXPECT_EQ(p.get<std::string>("tangent_property"), "damaged_tangent");
  EXPECT_THROW(r::parse_point<policy>(json::parse(R"({"point": {"strain": "s"}})")), std::invalid_argument)
      << "not a driver parameter";
  auto const fixed_point{r::parse_point<policy>(json::parse(R"({"point": {"strain_source": "s", "stress_source": "m",
    "residual_source": "scs::residual", "residual_tolerance": 1e-11, "max_passes": 200}})"))};
  EXPECT_EQ(fixed_point.get<std::string>("residual_source"), "scs::residual");
  EXPECT_EQ(fixed_point.get<T>("residual_tolerance"), 1e-11);
  EXPECT_EQ(fixed_point.get<int>("max_passes"), 200);
  auto const cols{r::parse_columns(doc)};
  ASSERT_EQ(cols.size(), 3u);
  EXPECT_EQ(cols[0].name, "strain");
  EXPECT_EQ(cols[0].source, "s::strain");
  EXPECT_FALSE(cols[0].full);
  EXPECT_EQ(cols[1].source, "m::damaged_stress");
  EXPECT_EQ(cols[2].name, "kappa");
  EXPECT_EQ(cols[2].source, "m::kappa");
}

TEST(ReportJson, ParsesRefinementStudies) {
  auto const doc = json::parse(R"({"cases": [{"name": "u", "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1], "steps": 50}],
    "studies": [{"name": "s", "case": "u", "steps": [10, 20]}]})");
  auto const studies{r::parse_studies(doc)};
  ASSERT_EQ(studies.size(), 1u);
  EXPECT_EQ(studies[0].name, "s");
  EXPECT_EQ(studies[0].steps, (std::vector<int>{10, 20}));
  auto const c{r::parse_case<T>(r::with_steps(studies[0].case_spec, 20))};
  EXPECT_EQ(c.path.size(), 20u);
  EXPECT_EQ(c.name, "u");
  EXPECT_TRUE(r::parse_studies(json::parse(R"({"cases": []})")).empty());
  EXPECT_THROW(r::parse_studies(json::parse(R"({"cases": [], "studies": [{"name": "s", "case": "nope", "steps": [1]}]})")),
               std::invalid_argument);
}

TEST(ReportJson, ParsesALoadCase) {
  auto const c{r::parse_case<T>(json::parse(R"({"name": "u", "control": "mixed", "stress_components": ["22", "33"],
    "direction": [1, 0, 0, 0.5, 0, 0], "path": [0, 1], "steps": 4, "scale": 0.01})"))};
  EXPECT_EQ(c.name, "u");
  EXPECT_EQ(c.path.size(), 4u);
  EXPECT_DOUBLE_EQ(c.points[1](0, 1), 0.005); // scale * path * direction
  EXPECT_DOUBLE_EQ(c.points[1](1, 0), 0.005);
  tensor2 const free{tmech::dcontract(c.control.stress_projector(), tmech::eye<T, 3, 2>())};
  EXPECT_DOUBLE_EQ(free(1, 1), 1);
  EXPECT_DOUBLE_EQ(free(0, 0), 0);
  EXPECT_EQ(c.tangent_every, 1);
  EXPECT_FALSE(c.fd_step.has_value()); // the driver's default applies
  EXPECT_DOUBLE_EQ(r::parse_case<T>(json::parse(R"({"name": "u", "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1], "fd_step": 1e-6})")).fd_step.value(), 1e-6);
  EXPECT_EQ(r::parse_case<T>(json::parse(R"({"name": "u", "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1], "tangent_every": 0})")).tangent_every, 0);
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

json neo_hooke_report() {
  return json::parse(R"({
    "model": {"materials": [
      {"type": "external_deformation_gradient_source", "name": "F"},
      {"type": "neo_hooke", "name": "model", "deformation_gradient_source": "F", "K": 100.0, "G": 50.0}]},
    "point": {"deformation_gradient_source": "F", "stress_source": "model"}})");
}

TEST(ReportJson, FiniteStrainPointAndNineComponentTensors) {
  auto const doc = neo_hooke_report();
  EXPECT_TRUE(r::is_finite_strain(doc));
  EXPECT_FALSE(r::is_finite_strain(elastic_report()));
  auto const p{r::parse_point<policy>(doc)};
  EXPECT_EQ(p.get<std::string>("deformation_gradient_source"), "F");
  EXPECT_FALSE(p.contains("strain_source"));
  auto const t{r::parse_tensor<T>(json::parse("[1, 2, 3, 4, 5, 6, 7, 8, 9]"))};
  EXPECT_EQ(t(0, 1), 4);
  EXPECT_EQ(t(1, 0), 7);
  EXPECT_EQ(t(2, 1), 9);
  EXPECT_EQ(r::parse_tensor<T>(json::parse("[1, 2, 3, 4, 5, 6]"))(1, 0), 4);
  EXPECT_THROW(r::parse_tensor<T>(json::parse("[1, 2, 3]")), std::invalid_argument);
}

TEST(ReportJson, FiniteStrainMixedControlMayNameTheLowerComponents) {
  auto const spec = json::parse(R"({"name": "f", "control": "mixed", "stress_components": ["22", "33", "21"],
    "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1]})");
  EXPECT_THROW(r::parse_case<T>(spec), std::invalid_argument) << "21 is no small-strain component";
  auto const c{r::parse_case<T>(spec, true)};
  tensor2 g;
  g(1, 0) = 1;
  EXPECT_DOUBLE_EQ(tensor2{tmech::dcontract(c.finite_control.stress_projector(), g)}(1, 0), 1);
}

TEST(ReportJson, CountsMustBeIntegersInRange) {
  auto const with = [](std::string const &extra) {
    return json::parse(R"({"name": "u", "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1], )" + extra + "}");
  };
  EXPECT_THROW(r::parse_case<T>(with(R"("steps": 2.5)")), std::invalid_argument);
  EXPECT_THROW(r::parse_case<T>(with(R"("steps": 0)")), std::invalid_argument);
  EXPECT_THROW(r::parse_case<T>(with(R"("steps": 1e12)")), std::invalid_argument);
  EXPECT_THROW(r::parse_case<T>(with(R"("tangent_every": -1)")), std::invalid_argument);
  EXPECT_EQ(r::parse_case<T>(with(R"("steps": 7)")).path.size(), 7u);
  auto doc = json::parse(R"({"point": {"strain_source": "s", "stress_source": "m", "max_iter": 1e10}})");
  EXPECT_THROW(static_cast<void>(r::parse_point<policy>(doc)), std::invalid_argument);
  doc["point"]["max_iter"] = 2.5;
  EXPECT_THROW(static_cast<void>(r::parse_point<policy>(doc)), std::invalid_argument);
  doc["point"].erase("max_iter");
  doc["point"]["max_passes"] = 0;
  EXPECT_THROW(static_cast<void>(r::parse_point<policy>(doc)), std::invalid_argument);
  auto const studies = json::parse(R"({"cases": [{"name": "u"}], "studies": [{"name": "s", "case": "u", "steps": [4, 2.5]}]})");
  EXPECT_THROW(static_cast<void>(r::parse_studies(studies)), std::invalid_argument);
}

TEST(ReportJson, OutputNamesStayInTheOutputDirectoryAndAreUnique) {
  for (auto const *bad : {"", ".", "..", "../x", "/tmp/x", "a/b", "a\\b", "a\tb"})
    EXPECT_THROW(static_cast<void>(r::output_stems(json::object({{"cases", json::array({json::object({{"name", bad}})})}}))),
                 std::invalid_argument) << bad;
  EXPECT_THROW(static_cast<void>(r::output_stems(json::parse(R"({"cases": [{"name": "u"}, {"name": "u"}]})"))),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(r::output_stems(json::parse(
                   R"({"cases": [{"name": "u_4"}, {"name": "u"}], "studies": [{"name": "u", "case": "u", "steps": [4]}]})"))),
               std::invalid_argument) << "the study's u_4 collides with the case";
  EXPECT_EQ(r::output_stems(json::parse(
                R"({"cases": [{"name": "u-1.a"}], "studies": [{"name": "a study", "case": "u-1.a", "steps": [2, 4]}]})")),
            (std::vector<std::string>{"u-1.a", "a study_2", "a study_4"}));
}

TEST(CaseRunner, FiniteStrainRecordsFPCauchyAndJ) {
  auto const doc = neo_hooke_report();
  auto const spec = json::parse(R"({"name": "u", "control": "uniaxial_stress",
    "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1], "steps": 2, "scale": 0.1})");
  auto const c{r::parse_case<T>(spec, true)};
  recording_sink sink;
  EXPECT_THROW(r::run_case<policy>(doc, r::parse_case<T>(spec), sink), std::invalid_argument)
      << "parsed with the small-strain control";
  ASSERT_TRUE(r::run_case<policy>(doc, c, sink));
  auto const col = [&](std::string const &n) {
    auto const it{std::find(sink.names.begin(), sink.names.end(), n)};
    EXPECT_NE(it, sink.names.end()) << n;
    return sink.records.back().columns[static_cast<std::size_t>(it - sink.names.begin())];
  };
  EXPECT_DOUBLE_EQ(col("F_11"), 1.1);
  EXPECT_NEAR(col("P_22"), 0, 1e-8);
  EXPECT_GT(col("stress_11"), 0); // Cauchy
  EXPECT_NEAR(col("stress_11"), col("P_11") * col("F_11") / col("J"), 1e-10);
  EXPECT_LT(col("J"), 1.1);
  EXPECT_EQ(sink.names.size(), 9u + 9u + 6u + 1u);
}

TEST(CaseRunner, TheCaseFdStepReachesTheDriver) {
  auto const spec = json::parse(R"({"name": "u", "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1], "steps": 1,
    "scale": 0.001, "fd_step": 0})");
  recording_sink sink;
  EXPECT_THROW(static_cast<void>(r::run_case<policy>(elastic_report(), r::parse_case<T>(spec), sink)),
               nm::umat::fatal_error) << "fd_step must be > 0";
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
  auto const stress_11{static_cast<std::size_t>(std::find(sink.names.begin(), sink.names.end(), "stress_11") - sink.names.begin())};
  EXPECT_DOUBLE_EQ(last.columns.back(), last.columns[stress_11]); // "sig" is elastic::stress[1,1]
  EXPECT_LT(last.tangent_error, 1e-6);
}

TEST(CaseRunner, ChecksTheTangentOnlyEveryNthStep) {
  auto const doc = elastic_report();
  auto const c{r::parse_case<T>(json::parse(R"({"name": "u", "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1],
    "steps": 4, "scale": 1e-3, "tangent_every": 2})"))};
  recording_sink sink;
  ASSERT_TRUE(r::run_case<policy>(doc, c, sink));
  ASSERT_EQ(sink.records.size(), 5u);
  for (std::size_t i{1}; i <= 4; ++i)
    EXPECT_EQ(std::isfinite(sink.records[i].tangent_error), i % 2 == 0) << i;
}

/// Exponential damage under uniaxial stress, one step to 2 %: Newton needs 7 corrections at the full increment, 6 at
/// half of it and at most 5 at a quarter; the point allows 5.
json softening_report() {
  return json::parse(R"({
    "model": {"materials": [
      {"type": "external_strain_source", "name": "strain"},
      {"type": "von_mises_state_function", "name": "state", "strain_source": "strain"},
      {"type": "strain_threshold_yield", "name": "yield", "state_source": "state", "kappa_critical": 0.002},
      {"type": "exponential_damage_law", "name": "law", "yield_source": "yield", "kappa_0": 0.002, "beta": 300.0},
      {"type": "linear_elasticity", "name": "elastic", "strain_producer_name": "strain", "K": 166666.67, "G": 76923.08},
      {"type": "isotropic_damage", "name": "damaged", "elastic_source": "elastic", "damage_source": "law",
       "state_source": "state", "yield_source": "yield"}]},
    "point": {"strain_source": "strain", "stress_source": "damaged", "stress_property": "damaged_stress",
              "tangent_property": "damaged_tangent", "max_iter": 5},
    "record": {"d": "law::damage"}})");
}

json softening_case(int steps, std::string const &extra = "") {
  return json::parse(R"({"name": "u", "control": "uniaxial_stress", "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1],
    "scale": 0.02, "tangent_every": 0, "steps": )" + std::to_string(steps) + extra + "}");
}

/// The case's CSV rows and residuals, as the report writes them.
template <typename Case>
std::pair<std::string, bool> csv_of(json const &doc, Case const &c) {
  std::ostringstream csv, its;
  r::csv_sink<policy> sink{csv, its};
  bool const converged{r::run_case<policy>(doc, c, sink)};
  return {csv.str() + its.str(), converged};
}

TEST(CaseRunner, ASofteningStepThatFailsWholeConvergesInFourSubsteps) {
  auto const doc = softening_report();
  for (int const cutbacks : {0, 1}) {
    recording_sink sink;
    EXPECT_FALSE(r::run_case<policy>(doc, r::parse_case<T>(softening_case(1, R"(, "max_cutbacks": )" + std::to_string(cutbacks))), sink))
        << cutbacks;
    ASSERT_EQ(sink.records.size(), 2u);
    EXPECT_FALSE(sink.records.back().result.converged);
    EXPECT_EQ(sink.records.back().cutbacks, cutbacks);
  }
  recording_sink sink;
  ASSERT_TRUE(r::run_case<policy>(doc, r::parse_case<T>(softening_case(1, R"(, "max_cutbacks": 2)")), sink));
  ASSERT_EQ(sink.records.size(), 2u);
  auto const &rec{sink.records.back()};
  EXPECT_EQ(rec.step.index, 1);
  EXPECT_EQ(rec.step.load, 0.02);
  EXPECT_EQ(rec.cutbacks, 2);
  ASSERT_EQ(rec.substeps.size(), 4u);
  EXPECT_TRUE(std::ranges::all_of(rec.substeps, &r::step_summary<T>::converged));
  EXPECT_EQ(rec.substeps.back().residuals, rec.result.residuals);
  EXPECT_EQ(sink.records.front().cutbacks, 0);
  EXPECT_EQ(sink.records.front().substeps.size(), 1u);

  // the four substeps are the four steps of the case with steps = 4, from the restored start
  recording_sink four;
  ASSERT_TRUE(r::run_case<policy>(doc, r::parse_case<T>(softening_case(4)), four));
  EXPECT_EQ(rec.columns, four.records.back().columns);
}

TEST(CaseRunner, WithoutCutbacksTheCsvIsUnchanged) {
  auto const doc = softening_report();
  auto const plain{csv_of(doc, r::parse_case<T>(softening_case(1)))};
  EXPECT_FALSE(plain.second);
  EXPECT_EQ(csv_of(doc, r::parse_case<T>(softening_case(1, R"(, "max_cutbacks": 0)"))), plain);
  // cutbacks a step does not need change nothing
  auto const four{csv_of(doc, r::parse_case<T>(softening_case(4)))};
  EXPECT_TRUE(four.second);
  EXPECT_EQ(csv_of(doc, r::parse_case<T>(softening_case(4, R"(, "max_cutbacks": 3)"))), four);
}

/// Linear elastic (tangent 2 I^sym + I x I) for a strain increment of at most "limit"; a larger one throws a cutback_error.
template <typename Traits>
class increment_limited final : public nm::material_base<increment_limited<Traits>, Traits> {
public:
  using base = nm::material_base<increment_limited<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit increment_limited(Args &&...args)
      : base(std::forward<Args>(args)...), m_stress(base::template add_output<tensor2>("stress")),
        m_tangent(base::template add_output<tensor4>("tangent")),
        m_limit(base::template get_parameter<value_type>("limit")),
        m_strain(base::template add_input_history<tensor2>(nm::connection_source{"strain", "strain"}, nm::EdgeKind::Global)) {
    base::add_callback(&increment_limited::compute, m_stress, m_tangent).reads(m_strain);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<value_type>(para, "limit");
    return para;
  }

  void compute() {
    if (tmech::norm(tensor2{m_strain.new_value() - m_strain.old_value()}) > m_limit)
      throw nm::umat::cutback_error("increment_limited: the strain increment is above the limit");
    auto const I{tmech::eye<value_type, 3, 2>()};
    m_tangent = 2 * nm::sym_identity<value_type, 3>() + tmech::otimes(I, I);
    m_stress = tmech::dcontract(m_tangent, m_strain.new_value());
  }

private:
  tensor2 &m_stress;
  tensor4 &m_tangent;
  value_type const &m_limit;
  nm::input_history<tensor2, nm::property_traits> const &m_strain;
};

json increment_limited_report() {
  static bool const registered{[] {
    nm::material_factory<policy>::instance().register_type<increment_limited<policy>>("increment_limited");
    return true;
  }()};
  static_cast<void>(registered);
  return json::parse(R"({
    "model": {"materials": [
      {"type": "external_strain_source", "name": "strain"},
      {"type": "increment_limited", "name": "m", "limit": 0.0025}]},
    "point": {"strain_source": "strain", "stress_source": "m"}})");
}

TEST(CaseRunner, ACutbackErrorCutsTheStepToo) {
  auto const doc = increment_limited_report();
  auto const spec = [](int cutbacks) {
    return r::parse_case<T>(json::parse(R"({"name": "u", "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1], "steps": 1,
      "scale": 0.004, "max_cutbacks": )" + std::to_string(cutbacks) + "}"));
  };
  recording_sink failed;
  EXPECT_THROW(static_cast<void>(r::run_case<policy>(doc, spec(0), failed)), nm::umat::cutback_error) << "as without cutbacks";
  recording_sink sink;
  ASSERT_TRUE(r::run_case<policy>(doc, spec(1), sink));
  ASSERT_EQ(sink.records.size(), 2u);
  EXPECT_EQ(sink.records.back().cutbacks, 1);
  EXPECT_EQ(sink.records.back().substeps.size(), 2u);
  EXPECT_DOUBLE_EQ(sink.records.back().columns.front(), 0.004); // strain_11 at the end of the step
}

TEST(CaseRunner, StopsAtAStepThatDoesNotConverge) {
  auto const doc = elastic_report();
  auto doc_without_iterations = doc;
  doc_without_iterations["point"]["max_iter"] = 0;
  auto const c{r::parse_case<T>(json::parse(R"({"name": "u", "control": "uniaxial_stress",
    "direction": [1, 0, 0, 0, 0, 0], "path": [0, 1], "steps": 3})"))};
  recording_sink sink;
  EXPECT_FALSE(r::run_case<policy>(doc_without_iterations, c, sink));
  ASSERT_EQ(sink.records.size(), 2u);
  EXPECT_FALSE(sink.records.back().result.converged);
}

} // namespace

namespace {

TEST(CsvSink, WritesHeaderRowsAndResiduals) {
  std::ostringstream csv, its;
  r::csv_sink<policy> sink{csv, its};
  sink.begin({"strain_11", "kappa"});
  r::step_record<policy> rec;
  rec.step = {1, 0, 1, 0.5};
  rec.result.residuals = {1.0, 1e-12};
  rec.result.converged = true;
  rec.columns = {1e-3, 2.0};
  rec.tangent_error = 1e-9;
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
