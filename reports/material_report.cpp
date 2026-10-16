// material_report <report.json> <output directory>
// Writes resolved.json, graph.json, per case <case>.csv and <case>_iterations.csv, and per study
// <study>_<steps>.csv (the study's case run with that many steps per segment).
// Exit code 0: all steps converged, 1: a step did not converge, 2: input error.

#include <filesystem>
#include <fstream>
#include <iostream>

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

/// Runs one case, writing <stem>.csv and <stem>_iterations.csv; false if a step did not converge.
bool run_one(nlohmann::json const &doc, r::load_case<T> const &c, std::filesystem::path const &stem) {
  std::ofstream steps{stem.string() + ".csv"}, residuals{stem.string() + "_iterations.csv"};
  r::csv_sink<policy> sink{steps, residuals};
  if (r::run_case<policy>(doc, c, sink))
    return true;
  std::cerr << "material_report: " << stem.filename().string() << ": a step did not converge\n";
  return false;
}

bool run(nlohmann::json const &doc, std::filesystem::path const &out) {
  std::filesystem::create_directories(out);
  std::ofstream{out / "resolved.json"} << doc.dump(2) << "\n";
  std::ofstream{out / "graph.json"} << r::describe_graph<policy>(doc).dump(2) << "\n";
  bool converged{true};
  for (auto const &spec : doc.at("cases"))
    converged = run_one(doc, r::parse_case<T>(spec), out / spec.at("name").get<std::string>()) && converged;
  for (auto const &study : r::parse_studies(doc))
    for (auto const n : study.steps)
      converged = run_one(doc, r::parse_case<T>(r::with_steps(study.case_spec, n)), out / (study.name + "_" + std::to_string(n))) && converged;
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
