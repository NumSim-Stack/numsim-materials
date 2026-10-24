// material_report <report.json> <output directory>
// Writes resolved.json, graph.json, per case <case>.csv and <case>_iterations.csv, and per study
// <study>_<steps>.csv (the study's case run with that many steps per segment).
// Exit code 0: all steps converged, 1: a step did not converge, 2: input error.

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "numsim-materials/report/case_runner.h"
#include "numsim-materials/report/csv_sink.h"
#include "numsim-materials/report/graph_description.h"
#include "numsim-materials/report/report_json.h"
#include "numsim-materials/report/report_spec.h"

namespace {

namespace r = numsim::materials::report;
using policy = numsim::materials::material_policy_default;
using T = policy::value_type;

nlohmann::json load(std::filesystem::path const &file) {
  std::ifstream in{file};
  if (!in)
    throw std::invalid_argument("cannot open " + file.string());
  return r::resolve_report(nlohmann::json::parse(in), file.parent_path());
}

std::ofstream open_out(std::filesystem::path const &file) {
  std::ofstream out{file};
  if (!out)
    throw std::runtime_error("cannot write " + file.string());
  return out;
}

void check_written(std::ofstream const &out, std::filesystem::path const &file) {
  if (!out)
    throw std::runtime_error("writing " + file.string() + " failed");
}

/// Runs one case, writing <stem>.csv and <stem>_iterations.csv; false if a step did not converge.
bool run_one(nlohmann::json const &doc, r::load_case<T> const &c, std::filesystem::path const &stem) {
  std::filesystem::path const steps_file{stem.string() + ".csv"}, residuals_file{stem.string() + "_iterations.csv"};
  auto steps{open_out(steps_file)}, residuals{open_out(residuals_file)};
  r::csv_sink<policy> sink{steps, residuals};
  bool const converged{r::run_case<policy>(doc, c, sink)};
  steps.flush();
  residuals.flush();
  check_written(steps, steps_file);
  check_written(residuals, residuals_file);
  if (!converged)
    std::cerr << "material_report: " << stem.filename().string() << ": a step did not converge\n";
  return converged;
}

bool run(nlohmann::json const &doc, std::filesystem::path const &out) {
  // every input error before the first case runs
  auto const stems{r::output_stems(doc)};
  bool const finite{r::is_finite_strain(doc)};
  std::vector<r::load_case<T>> cases;
  for (auto const &spec : doc.at("cases"))
    cases.push_back(r::parse_case<T>(spec, finite));
  for (auto const &study : r::parse_studies(doc))
    for (auto const n : study.steps)
      cases.push_back(r::parse_case<T>(r::with_steps(study.case_spec, n), finite));

  std::filesystem::create_directories(out);
  for (auto const &[name, content] : {std::pair{"resolved.json", doc}, std::pair{"graph.json", r::describe_graph<policy>(doc)}}) {
    auto file{open_out(out / name)};
    file << content.dump(2) << "\n";
    file.flush();
    check_written(file, out / name);
  }
  bool converged{true};
  for (std::size_t i{0}; i < cases.size(); ++i)
    converged = run_one(doc, cases[i], out / stems[i]) && converged;
  return converged;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: material_report <report.json> <output directory>\n";
    return 2;
  }
  try {
    return run(load(argv[1]), argv[2]) ? 0 : 1;
  } catch (std::exception const &e) {
    std::cerr << "material_report: " << e.what() << "\n";
    return 2;
  }
}
