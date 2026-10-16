// material_report <report.json> <output directory>
// Runs the load cases of a report document; writes resolved.json, graph.json and per case <case>.csv and
// <case>_iterations.csv. Load = scale * path (piecewise linear, `steps` per segment) times the strain
// `direction` and the stress target `stress`. Exit code 1 if a step does not converge, 2 on input errors.

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "numsim-materials/report/load_driver.h"
#include "numsim-materials/report/property_columns.h"
#include "numsim-materials/report/report_spec.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
namespace r = numsim::materials::report;
using policy = nm::material_policy_default;
using T = policy::value_type;
using driver = r::load_driver<policy>;
using evaluator = nm::umat::material_point_evaluator<policy>;
using context = nm::material_context<policy>;
using json = nlohmann::json;

r::control_mask control_of(json const &c) {
  auto const kind = c.value("control", std::string{"strain"});
  if (kind == "strain")
    return r::control::strain;
  if (kind == "uniaxial_stress")
    return r::control::uniaxial_stress;
  if (kind == "mixed") {
    r::control_mask m{};
    for (auto const &s : c.at("stress_components").get<std::vector<std::string>>()) {
      static std::vector<std::string> const slots{"11", "22", "33", "12", "13", "23"};
      auto const it{std::find(slots.begin(), slots.end(), s)};
      if (it == slots.end())
        throw std::invalid_argument("stress_components: unknown component '" + s + "'");
      m[static_cast<std::size_t>(it - slots.begin())] = true;
    }
    return m;
  }
  throw std::invalid_argument("unknown control '" + kind + "' (strain, uniaxial_stress, mixed)");
}

/// Tensor components [11, 22, 33, 12, 13, 23] of a symmetric tensor.
driver::tensor2 symmetric(json const &j) {
  auto const v = j.get<std::vector<T>>();
  if (v.size() != 6)
    throw std::invalid_argument("expected six components [11, 22, 33, 12, 13, 23]");
  driver::tensor2 t;
  t(0, 0) = v[0];
  t(1, 1) = v[1];
  t(2, 2) = v[2];
  t(0, 1) = t(1, 0) = v[3];
  t(0, 2) = t(2, 0) = v[4];
  t(1, 2) = t(2, 1) = v[5];
  return t;
}

std::vector<std::pair<std::string, std::string>> wiring(json const &doc) {
  std::vector<std::pair<std::string, std::string>> w{{"strain", doc.at("point").at("strain_source").get<std::string>() + "::strain"},
                                                     {"stress", doc.at("point").at("stress_source").get<std::string>() + "::stress"}};
  auto const record = doc.value("record", json::object());
  for (auto const &[name, source] : record.items())
    w.emplace_back(name, source.get<std::string>());
  return w;
}

struct point {
  context ctx;
  std::unique_ptr<evaluator> ev;

  explicit point(json const &doc) {
    nm::umat::make_json_builder<policy>(doc.at("model").dump())(ctx, {});
    auto const &p{doc.at("point")};
    evaluator::config cfg;
    cfg.strain_source = p.at("strain_source").get<std::string>();
    cfg.stress_source = p.at("stress_source").get<std::string>();
    if (p.contains("tangent_source"))
      cfg.tangent_source = p["tangent_source"].get<std::string>();
    cfg.time_source = p.value("time_source", std::string{});
    ev = std::make_unique<evaluator>(ctx, cfg);
  }
};

template <typename V>
bool is_history(nm::property_base *p) {
  return dynamic_cast<nm::history_property<V, nm::property_traits> const *>(p) != nullptr;
}

/// Materials, their parameters as used, and the properties they produce with their inputs.
json graph_of(json const &doc) {
  point p{doc};
  auto &factory{nm::object_store<policy>::factory_type::instance()};
  auto out = json::array();
  for (auto const &m : doc.at("model").at("materials")) {
    auto entry = json::object();
    auto const name = m.at("name").get<std::string>();
    auto const type = m.at("type").get<std::string>();
    entry["name"] = name;
    entry["type"] = type;
    entry["parameters"] = json::array();
    if (factory.contains(type))
      for (auto const &[key, param] : factory.schema(type)) {
        auto row = json::object();
        row["name"] = key;
        row["value"] = m.contains(key) ? m[key] : json{};
        row["description"] = param->description();
        entry["parameters"].push_back(row);
      }
    entry["properties"] = json::array();
    entry["inputs"] = json::array();
    for (auto *mat : p.ctx.materials()) {
      if (mat->name() != name)
        continue;
      for (auto const *prop_base : mat->get_property_registry().produced_properties()) {
        auto prop = json::object();
        prop["name"] = prop_base->traits().id.name;
        auto *base{p.ctx.find_property(name, prop_base->traits().id.name)};
        prop["history"] = is_history<T>(base) || is_history<driver::tensor2>(base) || is_history<driver::tensor4>(base);
        entry["properties"].push_back(prop);
      }
      for (auto const &input : mat->typed_inputs()) {
        auto in = json::object();
        in["source"] = input->source_owner() + "::" + input->source_name();
        in["kind"] = input->edge_kind() == nm::EdgeKind::Global ? "global" : "local";
        entry["inputs"].push_back(in);
      }
    }
    out.push_back(entry);
  }
  return out;
}

bool run_case(json const &doc, json const &c, std::filesystem::path const &out) {
  point p{doc};
  driver d(*p.ev);
  r::property_columns<policy> const cols(p.ctx, wiring(doc));

  auto const name = c.at("name").get<std::string>();
  auto const mask{control_of(c)};
  auto const direction{symmetric(c.at("direction"))};
  auto const stress{c.contains("stress") ? symmetric(c["stress"]) : driver::tensor2{}};
  auto const path = c.at("path").get<std::vector<T>>();
  auto const steps{c.value("steps", 50)};
  auto const dt{c.value("dt", T{1})};
  auto const scale{c.value("scale", T{1})};

  std::ofstream csv{out / (name + ".csv")}, its{out / (name + "_iterations.csv")};
  csv.precision(std::numeric_limits<T>::max_digits10);
  its.precision(6);
  csv << "step,time,load,iterations,converged,tangent_error";
  for (auto const &col : cols.names())
    csv << "," << col;
  csv << "\n";
  its << "step,iteration,residual\n";

  auto const write = [&](int step, T time, T load, driver::step_result const &res, T err) {
    csv << step << "," << time << "," << load << "," << res.residuals.size() - 1 << "," << res.converged << "," << err;
    for (auto const v : cols.values())
      csv << "," << v;
    csv << "\n";
    for (std::size_t i{0}; i < res.residuals.size(); ++i)
      its << step << "," << i << "," << res.residuals[i] << "\n";
  };

  // step 0: the unloaded state
  auto const initial{d.step(driver::tensor2{}, driver::tensor2{}, r::control::strain, 0, 0)};
  write(0, 0, scale * path.front(), initial, std::numeric_limits<T>::quiet_NaN());

  int step{0};
  for (std::size_t seg{1}; seg < path.size(); ++seg)
    for (int k{1}; k <= steps; ++k) {
      ++step;
      auto const load{scale * (path[seg - 1] + (path[seg] - path[seg - 1]) * T(k) / T(steps))};
      auto const res{d.step(driver::tensor2{load * direction}, driver::tensor2{load * stress}, mask, (step - 1) * dt, dt)};
      if (!res.converged) {
        write(step, step * dt, load, res, std::numeric_limits<T>::quiet_NaN());
        std::cerr << "material_report: " << name << ": step " << step << " did not converge\n";
        return false;
      }
      auto const values{cols.values()}; // read before the tangent check re-evaluates the graph
      auto const err{d.tangent_error(res)};
      csv << step << "," << step * dt << "," << load << "," << res.residuals.size() - 1 << ",1," << err;
      for (auto const v : values)
        csv << "," << v;
      csv << "\n";
      for (std::size_t i{0}; i < res.residuals.size(); ++i)
        its << step << "," << i << "," << res.residuals[i] << "\n";
    }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: material_report <report.json> <output directory>\n";
    return 2;
  }
  try {
    std::filesystem::path const file{argv[1]};
    std::ifstream in{file};
    if (!in)
      throw std::invalid_argument("cannot open " + file.string());
    auto const doc = r::resolve_report(json::parse(in), file.parent_path());
    std::filesystem::path const out{argv[2]};
    std::filesystem::create_directories(out);
    std::ofstream{out / "resolved.json"} << doc.dump(2) << "\n";
    std::ofstream{out / "graph.json"} << graph_of(doc).dump(2) << "\n";
    bool ok{true};
    for (auto const &c : doc.at("cases"))
      ok = run_case(doc, c, out) && ok;
    return ok ? 0 : 1;
  } catch (std::exception const &e) {
    std::cerr << "material_report: " << e.what() << "\n";
    return 2;
  }
}
