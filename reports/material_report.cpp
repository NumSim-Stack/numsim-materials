// material_report <case.json> <output directory>
// Runs the load cases of a material report and writes one CSV per case plus its Newton residuals.

#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "numsim-materials/report/load_driver.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
namespace r = numsim::materials::report;
using policy = nm::material_policy_default;
using T = policy::value_type;
using driver = r::load_driver<policy>;
using evaluator = nm::umat::material_point_evaluator<policy>;

char const *const slot_names[6]{"11", "22", "33", "12", "13", "23"};

r::control_mask control_of(nlohmann::json const &c) {
  auto const kind = c.value("control", std::string{"strain"});
  if (kind == "strain")
    return r::control::strain;
  if (kind == "uniaxial_stress")
    return r::control::uniaxial_stress;
  if (kind == "mixed") {
    r::control_mask m{};
    for (auto const i : c.at("stress_slots").get<std::vector<std::size_t>>())
      m.at(i) = true;
    return m;
  }
  throw std::invalid_argument("unknown control '" + kind + "' (strain, uniaxial_stress, mixed)");
}

driver::vector6 vector_of(nlohmann::json const &j) {
  auto const v = j.get<std::vector<T>>();
  if (v.size() != 6)
    throw std::invalid_argument("expected six Voigt slots (11, 22, 33, 12, 13, 23)");
  driver::vector6 out;
  tmech::adaptor<T const, 6, 1, tmech::full<6>> const view(v.data());
  out = view;
  return out;
}

/// "0..5  j2::plastic_strain  [engineering shear]" -> j2::plastic_strain[0] ... [5]
std::vector<std::string> statev_columns(evaluator const &ev) {
  std::vector<std::string> cols(ev.nstatv());
  std::regex const line{R"((\d+)\.\.(\d+)\s+(\S+).*)"};
  for (auto const &d : ev.describe_statev()) {
    std::smatch m;
    if (!std::regex_match(d, m, line))
      continue;
    auto const first{std::stoul(m[1])}, last{std::stoul(m[2])};
    for (auto i{first}; i <= last; ++i)
      cols[i] = first == last ? m[3].str() : m[3].str() + "[" + std::to_string(i - first) + "]";
  }
  return cols;
}

void run_case(nlohmann::json const &doc, nlohmann::json const &c, std::filesystem::path const &out) {
  nm::material_context<policy> ctx;
  nm::umat::make_json_builder<policy>(doc.at("model").dump())(ctx, {});
  evaluator::config cfg;
  cfg.strain_source = doc.at("strain_source").get<std::string>();
  cfg.stress_source = doc.at("stress_source").get<std::string>();
  if (doc.contains("tangent_source"))
    cfg.tangent_source = doc["tangent_source"].get<std::string>();
  cfg.time_source = doc.value("time_source", std::string{});
  evaluator ev(ctx, cfg);
  driver d(ev);

  auto const name = c.at("name").get<std::string>();
  auto const mask{control_of(c)};
  auto const direction{vector_of(c.at("direction"))};
  auto const stress{c.contains("stress") ? vector_of(c["stress"]) : driver::vector6{}};
  auto const path = c.at("path").get<std::vector<T>>();
  auto const steps{c.value("steps", 50)};
  auto const dt{c.value("dt", T{1})};
  auto const cols{statev_columns(ev)};

  std::ofstream csv{out / (name + ".csv")}, its{out / (name + "_iterations.csv")};
  csv << "step,time,load";
  for (auto const *s : slot_names)
    csv << ",eps" << s;
  for (auto const *s : slot_names)
    csv << ",sig" << s;
  csv << ",iterations,converged,tangent_error";
  for (auto const &col : cols)
    csv << "," << col;
  csv << "\n";
  its << "step,iteration,residual,relative\n";
  csv.precision(12);
  its.precision(6);

  int step{0};
  for (std::size_t seg{1}; seg < path.size(); ++seg)
    for (int k{1}; k <= steps; ++k) {
      ++step;
      auto const load{path[seg - 1] + (path[seg] - path[seg - 1]) * T(k) / T(steps)};
      auto const time{(step - 1) * dt};
      auto const res{d.step(driver::vector6{load * direction}, stress, mask, time, dt)};
      auto const err{d.tangent_error(res)};
      csv << step << "," << time + dt << "," << load;
      for (std::size_t i{0}; i < 6; ++i)
        csv << "," << res.strain(i);
      for (std::size_t i{0}; i < 6; ++i)
        csv << "," << res.stress(i);
      csv << "," << res.residuals.size() - 1 << "," << res.converged << "," << err;
      for (auto const v : d.statev())
        csv << "," << v;
      csv << "\n";
      auto const scale{std::max(tmech::norm(res.stress), T{1e-300})};
      for (std::size_t i{0}; i < res.residuals.size(); ++i)
        its << step << "," << i << "," << res.residuals[i] << "," << res.residuals[i] / scale << "\n";
      if (!res.converged) {
        std::cerr << name << ": step " << step << " did not converge\n";
        return;
      }
    }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: material_report <case.json> <output directory>\n";
    return 2;
  }
  try {
    std::ifstream in{argv[1]};
    auto const doc = nlohmann::json::parse(in);
    std::filesystem::path const out{argv[2]};
    std::filesystem::create_directories(out);
    for (auto const &c : doc.at("cases"))
      run_case(doc, c, out);
  } catch (std::exception const &e) {
    std::cerr << "material_report: " << e.what() << "\n";
    return 1;
  }
}
