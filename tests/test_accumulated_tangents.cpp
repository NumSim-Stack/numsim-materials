#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <tmech/tmech.h>
#include "numsim-materials/report/case_runner.h"
#include "numsim-materials/report/report_spec.h"

/// The derivative accumulated from declared partials (#219) against the hand-written tangent at every step of the
/// report load paths: loading, unloading through zero, reloading, under strain and stress control.
namespace {

namespace nm = numsim::materials;
namespace r = numsim::materials::report;
using policy = nm::material_policy_default;
using T = policy::value_type;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

/// d target / d seed against the tangent output, each "material::property" or an input of the point; "{q}" runs
/// over q = 0, 1, ... while the material exists.
struct accumulated {
  std::string target, seed, tangent;
};

struct family {
  std::string report;
  std::vector<accumulated> pairs;
  int steps;                      ///< per path segment
  std::vector<std::string> cases{}; ///< the report cases run, empty: all
};

void PrintTo(family const &f, std::ostream *os) { *os << f.report; }

nlohmann::json load_report(std::string const &name) {
  auto const file{std::filesystem::path{NUMSIM_MATERIALS_DOCS_DIR} / name / "report.json"};
  std::ifstream in{file};
  return r::resolve_report(nlohmann::json::parse(in), file.parent_path());
}

std::string with_q(std::string text, std::size_t q) {
  for (auto at{text.find("{q}")}; at != std::string::npos; at = text.find("{q}"))
    text.replace(at, 3, std::to_string(q));
  return text;
}

struct check {
  std::string name;
  tensor4 const &derivative;
  tensor4 const &tangent;
};

class AccumulatedTangent : public ::testing::TestWithParam<family> {};

TEST_P(AccumulatedTangent, EqualsTheHandWrittenTangentAlongTheReportPaths) {
  nlohmann::json const doc = load_report(GetParam().report);
  for (auto const &spec : doc.at("cases")) {
    if (!GetParam().cases.empty() && !std::ranges::contains(GetParam().cases, spec.at("name").get<std::string>()))
      continue;
    auto const c{r::parse_case<T>(r::with_steps(spec, GetParam().steps))};
    r::case_model<policy> model(doc);
    nm::material_point_driver<policy> driver{model.context(), r::parse_point<policy>(doc)};
    auto &point{driver.point()};
    std::vector<check> checks;
    for (auto const &[target, seed, tangent] : GetParam().pairs)
      for (std::size_t q{0};; ++q) {
        auto const name{with_q(target, q)};
        if (!model.context().find(nm::connection_source::parse(name).material))
          break;
        checks.push_back({name, point.derivative<tensor4>(name, with_q(seed, q)), point.output<tensor4>(with_q(tangent, q))});
        if (name == target)
          break;
      }
    ASSERT_GE(checks.size(), GetParam().pairs.size());
    auto const compare{[&](auto const &res, int step) {
      ASSERT_TRUE(res.converged) << c.name << " step " << step;
      for (auto const &[name, derivative, tangent] : checks)
        EXPECT_LE(tmech::norm(tensor4{derivative - tangent}), 1e-12 * tmech::norm(tangent)) << c.name << " step " << step << ": " << name;
      driver.accept();
    }};
    compare(driver.step(tensor2{}, tensor2{}, nm::load_control<T>::strain(), 0, 0), 0);
    for (auto const &s : c.path.steps())
      compare(driver.step(c.strain_at(s), tensor2{s.load * c.stress}, c.control, s.time, s.dt), s.index);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Reports, AccumulatedTangent,
    ::testing::Values(
        family{"linear_elasticity", {{"elastic::stress", "strain", "elastic::tangent"}}, 10},
        family{"incremental_linear_elasticity", {{"inc::stress", "strain", "inc::tangent"}}, 10},
        family{"isotropic_damage", {{"damaged::damaged_stress", "strain", "damaged::damaged_tangent"}}, 10},
        family{"mean_field_damage",
               {{"damage::damaged_stress", "mt_strain::strain_matrix", "damage::damaged_tangent"},
                {"stiff::stress", "mt_strain::strain_spheres", "C_i::tangent"},
                {"intact::stress", "strain", "mt_intact::tangent"}},
               10},
        family{"short_fibre_damage",
               {{"fibre_{q}::damaged_stress", "sf_strain::strain_sf_{q}", "fibre_{q}::damaged_tangent"},
                {"lfibre_{q}::damaged_stress", "sfl_strain::strain_sfl_{q}", "lfibre_{q}::damaged_tangent"},
                {"intact::stress", "strain", "intact_tangent::tangent"}},
               10,
               {"uniaxial_strain"}}, // the nonlinear localisation needs small steps: one case keeps the test short
        family{"mean_field_composite_global",
               {{"mt_stress::stress", "strain", "mt::tangent"},
                {"dilute_stress::stress", "strain", "dilute::tangent"},
                {"scs_stress::stress", "strain", "scs::tangent"},
                {"ud_stress::stress", "strain", "ud::tangent"}},
               2},
        family{"mean_field_composite_global_rule_of_mixture",
               {{"voigt_stress::stress", "strain", "voigt::tangent"},
                {"reuss_stress::stress", "strain", "reuss::tangent"},
                {"hill_stress::stress", "strain", "hill::tangent"}},
               2},
        family{"short_fibre_composite",
               {{"ud_stress::stress", "strain", "ud::tangent"},
                {"aligned_stress::stress", "strain", "aligned::tangent"},
                {"injection_stress::stress", "strain", "injection::tangent"},
                {"random_stress::stress", "strain", "random::tangent"},
                {"random_stiffness_stress::stress", "strain", "random_stiffness::tangent"}},
               2}),
    [](auto const &info) { return info.param.report; });

} // namespace
