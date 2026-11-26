// material_report <report.json> <output directory>
// Writes resolved.json, graph.json, per case <case>.csv and <case>_iterations.csv, and per study
// <study>_<steps>.csv (the study's case run with that many steps per segment).
// Exit code 0: all steps converged, 1: a case did not converge or failed numerically (the others still run),
// 2: an input error, found before the first case runs.

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "numsim-materials/core/errors.h"
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
  return r::resolve_report(r::detail::parse_file(file), file.parent_path());
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

/// csv_sink that can record the step after the last one as not converged, with NaN columns.
class case_sink {
public:
  case_sink(std::ostream &steps, std::ostream &residuals) : m_csv(steps, residuals) {}

  void begin(std::vector<std::string> const &names) {
    m_columns = names.size();
    m_csv.begin(names);
  }

  void record(r::step_record<policy> const &rec) {
    m_next = rec.step.index + 1;
    m_csv.record(rec);
  }

  void record_failure(r::load_case<T> const &c) {
    if (m_columns == 0)
      return; // no header yet
    r::step_record<policy> rec{m_next == 0 ? c.path.unloaded() : c.path.at(m_next), {},
                               std::vector<T>(m_columns, std::numeric_limits<T>::quiet_NaN())};
    m_csv.record(rec);
  }

private:
  r::csv_sink<policy> m_csv;
  std::size_t m_columns{0};
  int m_next{0};
};

/// Runs one case, writing <stem>.csv and <stem>_iterations.csv; false if a step did not converge or a non-fatal
/// runtime_error (which the runner cuts back) stopped it, that step then recorded as not converged.
bool run_one(nlohmann::json const &doc, r::load_case<T> const &c, std::filesystem::path const &stem) {
  std::filesystem::path const steps_file{stem.string() + ".csv"}, residuals_file{stem.string() + "_iterations.csv"};
  auto steps{open_out(steps_file)}, residuals{open_out(residuals_file)};
  case_sink sink{steps, residuals};
  bool converged{false};
  try {
    converged = r::run_case<policy>(doc, c, sink);
    if (!converged)
      std::cerr << "material_report: '" << stem.filename().string() << "': a step did not converge\n";
  } catch (numsim::materials::fatal_error const &) {
    throw;
  } catch (std::runtime_error const &e) {
    std::cerr << "material_report: '" << stem.filename().string() << "': did not converge: " << e.what() << "\n";
    sink.record_failure(c);
  }
  steps.flush();
  residuals.flush();
  check_written(steps, steps_file);
  check_written(residuals, residuals_file);
  return converged;
}

bool run(nlohmann::json const &doc, std::filesystem::path const &out) {
  // every input error before the first case runs
  r::check_report_keys(doc);
  auto const stems{r::output_stems(doc)};
  bool const finite{r::is_finite_strain(doc)};
  r::check_model_and_columns<policy>(doc);
  std::vector<r::load_case<T>> cases;
  for (auto const &spec : r::detail::case_list(doc))
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
    try {
      converged = run_one(doc, cases[i], out / stems[i]) && converged;
    } catch (std::exception const &e) {
      throw std::runtime_error("'" + stems[i] + "': " + e.what());
    }
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
