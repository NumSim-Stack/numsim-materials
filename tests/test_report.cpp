#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <tmech/tmech.h>
#include "numsim-materials/umat/material_point_driver.h"
#include "numsim-materials/report/property_columns.h"
#include "numsim-materials/report/report_spec.h"
#include "numsim-materials/umat/json_model.h"

namespace {

namespace nm = numsim::materials;
namespace r = numsim::materials::report;

using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using evaluator = nm::umat::material_point_evaluator<policy>;
using driver = nm::umat::material_point_driver<policy>;
using columns = r::property_columns<policy>;
using json = nlohmann::json;

constexpr T H{1000.0};

json j2_model() {
  return json::parse(R"({"materials": [
    {"type": "external_strain_source", "name": "strain"},
    {"type": "local_newton", "name": "solver"},
    {"type": "linear_isotropic_hardening", "name": "hardening", "source": "j2", "K": 1000.0},
    {"type": "j2_plasticity", "name": "j2", "strain_source": "strain", "solver_source": "solver",
     "hardening_source": "hardening", "K": 166.67, "G": 76.92, "sigma_0": 0.5}]})");
}

struct j2_point {
  ctx_type ctx;
  std::unique_ptr<evaluator> ev;
  std::unique_ptr<driver> d;
  driver::step_result res;

  j2_point() {
    nm::umat::make_json_builder<policy>(j2_model().dump())(ctx, {});
    evaluator::config cfg;
    cfg.strain_source = "strain";
    cfg.stress_source = "j2";
    ev = std::make_unique<evaluator>(ctx, cfg);
    d = std::make_unique<driver>(*ev);
    driver::tensor2 eps;
    eps(0, 0) = 5e-3;
    res = d->step(eps, driver::tensor2{}, nm::umat::load_control<T>::uniaxial_stress(), 0, 1);
  }
};

TEST(PropertyColumns, ScalarHistoryPropertyIsOneColumn) {
  j2_point p;
  columns const c(p.ctx, {{"kappa", "j2::equivalent_plastic_strain"}});
  ASSERT_EQ(c.names(), std::vector<std::string>{"kappa"});
  auto const v{c.values()};
  EXPECT_GT(v[0], 0);
  EXPECT_DOUBLE_EQ(v[0], p.ctx.get<T>("j2", "equivalent_plastic_strain"));
}

TEST(PropertyColumns, TensorExpandsToSixSymmetricComponents) {
  j2_point p;
  columns const c(p.ctx, {{"sig", "j2::stress"}});
  EXPECT_EQ(c.names(), (std::vector<std::string>{"sig_11", "sig_22", "sig_33", "sig_12", "sig_13", "sig_23"}));
  auto const v{c.values()};
  EXPECT_NEAR(v[0], p.res.stress(0, 0), 1e-12);
  EXPECT_NEAR(v[1], 0, 1e-8);
}

TEST(PropertyColumns, ComponentSelectionIsOneBased) {
  j2_point p;
  columns const c(p.ctx, {{"eps_p11", "j2::plastic_strain[1,1]"}, {"C1111", "j2::tangent[1,1,1,1]"}});
  ASSERT_EQ(c.names().size(), 2u);
  auto const v{c.values()};
  auto const &eps_p{p.ctx.get<driver::tensor2>("j2", "plastic_strain")};
  EXPECT_DOUBLE_EQ(v[0], eps_p(0, 0));
  EXPECT_DOUBLE_EQ(v[1], p.res.tangent(0, 0, 0, 0));
}

TEST(PropertyColumns, WiresPropertiesOfOtherMaterials) {
  j2_point p;
  columns const c(p.ctx, {{"R", "hardening::hardening_stress"}, {"kappa", "j2::equivalent_plastic_strain"}});
  auto const v{c.values()};
  EXPECT_NEAR(v[0], H * v[1], 1e-12);
}

TEST(PropertyColumns, RejectsUnknownAndUnsupported) {
  j2_point p;
  EXPECT_THROW(columns(p.ctx, {{"x", "j2::nope"}}), std::invalid_argument);
  EXPECT_THROW(columns(p.ctx, {{"x", "nope::stress"}}), std::invalid_argument);
  EXPECT_THROW(columns(p.ctx, {{"x", "j2::tangent"}}), std::invalid_argument);   // rank 4 needs a component
  EXPECT_THROW(columns(p.ctx, {{"x", "j2::stress[1,4]"}}), std::invalid_argument);
  EXPECT_THROW(columns(p.ctx, {{"x", "j2::stress[1]"}}), std::invalid_argument);
}

TEST(ReportSpec, SubstitutesParameters) {
  auto const spec = json::parse(R"({"parameters": {"K": 2.5, "name": "j2"},
    "model": {"materials": [{"type": "x", "K": "$K", "list": [1, "$K"], "src": "$name"}]}})");
  auto const out = r::resolve_report(spec, ".");
  EXPECT_EQ(out["model"]["materials"][0]["K"], 2.5);
  EXPECT_EQ(out["model"]["materials"][0]["list"][1], 2.5);
  EXPECT_EQ(out["model"]["materials"][0]["src"], "j2");
}

TEST(ReportSpec, RecordsWhereEachParameterIsUsed) {
  auto const spec = json::parse(R"({"parameters": {"K": 2.5, "unused": 1},
    "model": {"materials": [{"type": "x", "name": "m", "K": "$K"}]}, "cases": [{"scale": "$K"}]})");
  auto const out = r::resolve_report(spec, ".");
  EXPECT_EQ(out["parameter_usage"]["K"], json::parse(R"(["cases[0].scale", "model.materials[0].K"])"));
  EXPECT_EQ(out["parameter_usage"]["unused"], json::array());
}

TEST(ReportSpec, UnknownParameterThrows) {
  auto const spec = json::parse(R"({"parameters": {}, "model": {"K": "$K"}})");
  EXPECT_THROW(r::resolve_report(spec, "."), std::invalid_argument);
}

TEST(ReportSpec, IncludeAppendsArraysAndKeepsLocalValues) {
  auto const dir{std::filesystem::temp_directory_path() / "numsim_report_spec_test"};
  std::filesystem::create_directories(dir);
  std::ofstream{dir / "standard.json"} << R"({"cases": [{"name": "a"}], "plots": [{"case": "a"}],
    "parameters": {"steps": 10, "K": 1}})";
  auto const spec = json::parse(R"({"include": ["standard.json"], "parameters": {"K": 2},
    "cases": [{"name": "b", "steps": "$steps"}]})");
  auto const out = r::resolve_report(spec, dir.string());
  ASSERT_EQ(out["cases"].size(), 2u);
  EXPECT_EQ(out["cases"][0]["name"], "a");
  EXPECT_EQ(out["cases"][1]["steps"], 10);
  EXPECT_EQ(out["plots"].size(), 1u);
  EXPECT_EQ(out["parameters"]["K"], 2);
  EXPECT_FALSE(out.contains("include"));
}

} // namespace
