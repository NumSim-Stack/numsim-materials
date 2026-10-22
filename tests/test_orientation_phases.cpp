/// Short fibres at their local strains: one fibre phase per orientation node, localised by a kernel with
/// the initial moduli (linear localisation; the phase materials may still damage).
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <tmech/tmech.h>

#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/orientation.h"
#include "numsim-materials/default_materials.h"
#include "numsim-materials/io/json_material_factory.h"
#include "numsim-materials/umat/external_state_source.h"
#include "numsim-materials/umat/json_model.h"
#include "numsim-materials/umat/material_point_driver.h"
#include "numsim-materials/umat/material_point_evaluator.h"

namespace {

namespace nm = numsim::materials;
namespace u = numsim::materials::umat;
using policy = nm::material_policy_default;
using T = policy::value_type;
using ctx_type = nm::material_context<policy>;
using json = nlohmann::json;
using tensor1 = tmech::tensor<T, 3, 1>;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

struct Registration {
  Registration() { nm::register_default_materials<policy>(); }
};
const Registration registration_{};

json const injection = json::array({0.7, 0.25, 0.05, 0.0, 0.0, 0.0});
json const aligned = json::array({1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
json const random_ = json::array({1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0, 0.0, 0.0});

tensor2 orientation_tensor(json const &a) {
  tensor2 A;
  A(0, 0) = a[0];
  A(1, 1) = a[1];
  A(2, 2) = a[2];
  A(0, 1) = A(1, 0) = a[3];
  A(0, 2) = A(2, 0) = a[4];
  A(1, 2) = A(2, 1) = a[5];
  return A;
}

/// Elastic fibres, one linear_stress per direction.
json elastic_fibres() {
  json const fibre = {{"type", "linear_stress"}, {"name", "fibre_{q}"}, {"tangent_source", "C_f"},
                      {"strain_source", "sf_strain"}, {"strain_property", "strain_sf_{q}"}};
  return json::array({fibre});
}

/// Damaging fibres: the chain of test_mean_field_damage per direction.
json damaging_fibres() {
  json const elastic = {{"type", "linear_elasticity"}, {"name", "fe_{q}"}, {"strain_producer_name", "sf_strain"},
                        {"strain_property", "strain_sf_{q}"}, {"K", 166666.67}, {"G", 76923.08}};
  json const state = {{"type", "von_mises_state_function"}, {"name", "fs_{q}"}, {"strain_source", "sf_strain"},
                      {"strain_property", "strain_sf_{q}"}};
  json const yield = {{"type", "strain_threshold_yield"}, {"name", "fy_{q}"}, {"state_source", "fs_{q}"}, {"kappa_critical", 0.002}};
  json const law = {{"type", "exponential_damage_law"}, {"name", "fl_{q}"}, {"yield_source", "fy_{q}"}, {"kappa_0", 0.002}, {"beta", 300.0}};
  json const damage = {{"type", "isotropic_damage"}, {"name", "fibre_{q}"}, {"elastic_source", "fe_{q}"}, {"damage_source", "fl_{q}"},
                       {"state_source", "fs_{q}"}, {"yield_source", "fy_{q}"}};
  return json::array({elastic, state, yield, law, damage});
}

/// The orientation_phases entry over the matrix phase and the fibre fraction "c_".
json short_fibres(json const &orientation, json const &per_direction, bool damaging) {
  json e = {{"type", "orientation_phases"}, {"name", "sf"}, {"strain_source", "strain"}, {"matrix", "matrix"},
            {"matrix_stress", "elastic_m::stress"}, {"fraction_source", "c_"}, {"orientation", orientation}, {"precision", 11},
            {"kernel", {{"type", "mori_tanaka"},
                        {"eshelby", {{"type", "eshelby_spheroid"}, {"aspect_ratio", 15.0}, {"nu_source", "moduli"}, {"nu_property", "poisson_ratio"}}}}},
            {"phase", {{"tangent_source", "C_f"}}}, {"per_direction", per_direction}, {"stress", "fibre_{q}::stress"}};
  if (damaging) { // the phase tangents (initial moduli) localise, the damaged ones average
    e["stress"] = "fibre_{q}::damaged_stress";
    e["tangent"] = "fibre_{q}::damaged_tangent";
    e["matrix_tangent"] = "elastic_m::tangent";
  }
  return e;
}

/// Damaging fibres localised with their current tangents: the phase nodes read the damaged tangent over a
/// local edge, starting from the elastic one.
json nonlinear(json e) {
  e["localisation"] = "nonlinear";
  e["phase"] = {{"tangent_source", "fibre_{q}"}, {"tangent_property", "damaged_tangent"}, {"tangent_edge", "local"}, {"initial_tangent_source", "C_f"}};
  return e;
}

/// The matrix, the fibre tangent and, next to the local composite, the global short_fibre_composite.
json deck(json const &orientation, json const &per_direction, bool damaging = false, bool nonlinear_ = false) {
  json m = json::array();
  m.push_back({{"type", "external_strain_source"}, {"name", "strain"}});
  for (auto const &[n, v] : std::array<std::pair<char const *, T>, 6>{{{"Km", 3333.33}, {"Gm", 1153.85}, {"Kf", 166666.67},
                                                                       {"Gf", 76923.08}, {"c_", 0.2}, {"cm_", 0.8}}})
    m.push_back({{"type", "constant_scalar"}, {"name", n}, {"value", v}});
  m.push_back({{"type", "isotropic_tangent"}, {"name", "C_m"}, {"K_source", "Km"}, {"G_source", "Gm"}});
  m.push_back({{"type", "isotropic_tangent"}, {"name", "C_f"}, {"K_source", "Kf"}, {"G_source", "Gf"}});
  m.push_back({{"type", "isotropic_moduli"}, {"name", "moduli"}, {"tangent_source", "C_m"}});
  m.push_back({{"type", "phase"}, {"name", "matrix"}, {"tangent_source", "C_m"}, {"fraction_source", "cm_"}});
  m.push_back({{"type", "linear_elasticity"}, {"name", "elastic_m"}, {"strain_producer_name", "sf_strain"},
               {"strain_property", "strain_matrix"}, {"K", 3333.33}, {"G", 1153.85}});
  m.push_back(nonlinear_ ? nonlinear(short_fibres(orientation, per_direction, damaging)) : short_fibres(orientation, per_direction, damaging));
  // the global composite over the same matrix and fibre, aligned along x1
  m.push_back({{"type", "eshelby_spheroid"}, {"name", "shape"}, {"aspect_ratio", 15.0}, {"nu_source", "moduli"}, {"nu_property", "poisson_ratio"}});
  m.push_back({{"type", "phase"}, {"name", "fibre"}, {"tangent_source", "C_f"}, {"fraction_source", "c_"}, {"eshelby_source", "shape"}});
  m.push_back({{"type", "short_fibre_composite"}, {"name", "global"}, {"matrix", "matrix"}, {"fibre", "fibre"},
               {"orientation", orientation}, {"precision", 11}});
  return {{"materials", m}};
}

std::unique_ptr<ctx_type> build(json const &doc) {
  auto ctx{std::make_unique<ctx_type>()};
  u::make_json_builder<policy>(doc.dump())(*ctx, {});
  return ctx;
}

tensor2 uniaxial(T e) {
  tensor2 eps;
  eps(0, 0) = e;
  return eps;
}

void step(ctx_type &ctx, tensor2 const &from, tensor2 const &to) {
  dynamic_cast<nm::external_strain_source<policy> &>(*ctx.find("strain")).bind(from, to);
  ctx.update();
  ctx.commit();
}

T relative(tensor4 const &a, tensor4 const &b) { return tmech::norm(tensor4{a - b}) / tmech::norm(b); }

std::vector<json> expand(json const &e) { return nm::json_expansion_registry<json>::instance().expand("orientation_phases", e); }

json named(std::vector<json> const &entries, std::string const &name) {
  auto const it{std::ranges::find_if(entries, [&](json const &e) { return e.at("name") == name; })};
  return it == entries.end() ? json::object() : *it;
}

TEST(OrientationNodes, OneNodePerAntipodalPairCarryingBothWeights) {
  auto const acg{nm::orientation::fit_acg(orientation_tensor(injection))};
  auto const full{nm::orientation::nodes(acg, 11)};
  auto const half{nm::orientation::half_nodes(acg, 11)};
  ASSERT_EQ(half.size(), full.size() / 2);
  T sum{0};
  tensor2 moment_half, moment_full;
  for (auto const &[n, w] : half) {
    sum += w;
    moment_half += w * tmech::otimes(n, n);
    for (auto const &[m, v] : half)
      EXPECT_GT(tmech::norm(tensor1{n + m}), 1e-8); // no antipode kept
  }
  for (auto const &[n, w] : full)
    moment_full += w * tmech::otimes(n, n);
  EXPECT_NEAR(sum, 1, 1e-14);
  EXPECT_LT(tmech::norm(tensor2{moment_half - moment_full}), 1e-14);
}

TEST(OrientationNodes, AnAlignedDistributionIsOneNode) {
  auto const half{nm::orientation::half_nodes(nm::orientation::fit_acg(orientation_tensor(aligned)), 11)};
  ASSERT_EQ(half.size(), 1u);
  EXPECT_EQ(half.front().second, 1);
}

TEST(OrientationPhases, ExpandsIntoOnePhasePerDirection) {
  auto const entries = expand(short_fibres(injection, damaging_fibres(), true));
  // fractions, then per direction Eshelby, phase and the five-node chain, then the local pair
  ASSERT_EQ(entries.size(), 1 + 25 * (2 + 5) + 2);
  EXPECT_EQ(named(entries, "sf_fractions").at("type"), "scalar_split");
  EXPECT_EQ(named(entries, "sf_S_24").at("type"), "eshelby_spheroid");
  EXPECT_EQ(named(entries, "sf_24").at("eshelby_source"), "sf_S_24");
  EXPECT_EQ(named(entries, "sf_24").at("fraction_property"), "value_24");
  EXPECT_EQ(named(entries, "fl_24").at("yield_source"), "fy_24");
  json const strain = named(entries, "sf_strain");
  ASSERT_EQ(strain.at("phases").size(), 26u);
  EXPECT_EQ(strain.at("phases")[0], "matrix"); // the matrix first
  EXPECT_EQ(strain.at("kernel"), json({{"type", "mori_tanaka"}})); // the shape went to the phases
  json const composite = named(entries, "sf");
  EXPECT_EQ(composite.at("type"), "mean_field_composite_local_linear");
  ASSERT_EQ(composite.at("phases").size(), 26u);
  EXPECT_EQ(composite.at("phases")[0], json({{"concentration", "sf_strain::concentration_matrix"}, {"stress", "elastic_m::stress"},
                                             {"tangent", "elastic_m::tangent"}, {"fraction", "matrix::volume_fraction"}}));
  EXPECT_EQ(composite.at("phases")[4], json({{"concentration", "sf_strain::concentration_sf_3"}, {"stress", "fibre_3::damaged_stress"},
                                             {"tangent", "fibre_3::damaged_tangent"}, {"fraction", "sf_3::volume_fraction"}}));
}

TEST(OrientationPhases, TangentsDefaultToThePhaseNodes) {
  json const composite = named(expand(short_fibres(injection, elastic_fibres(), false)), "sf");
  EXPECT_EQ(composite.at("phases")[0].at("tangent"), "matrix::tangent");
  EXPECT_EQ(composite.at("phases")[1].at("tangent"), "sf_0::tangent");
}

class ElasticFibres : public ::testing::TestWithParam<json> {};

// Mori-Tanaka over the node phases is the global concentration average: A_m = inv(cm I + c <T>)
TEST_P(ElasticFibres, OnePassIsTheGlobalShortFibreComposite) {
  auto ctx{build(deck(GetParam(), elastic_fibres()))};
  tensor2 eps{uniaxial(0.001)};
  eps(1, 2) = eps(2, 1) = 0.0004;
  step(*ctx, tensor2{}, eps);
  tensor4 const &C{ctx->get<tensor4>("global", "tangent")};
  EXPECT_LT(relative(ctx->get<tensor4>("sf", "tangent"), C), 1e-10);
  tensor2 const sigma{tmech::dcontract(C, eps)};
  EXPECT_LT(tmech::norm(tensor2{ctx->get<tensor2>("sf", "stress") - sigma}) / tmech::norm(sigma), 1e-10);
  // the fibre fractions split c over the directions
  T c{0};
  for (std::size_t q{0}; ctx->find("sf_" + std::to_string(q)); ++q)
    c += ctx->get<T>("sf_fractions", "value_" + std::to_string(q));
  EXPECT_NEAR(c, 0.2, 1e-14);
}

INSTANTIATE_TEST_SUITE_P(Orientations, ElasticFibres, ::testing::Values(aligned, injection, random_));

json kernel_of(std::string const &type) {
  json const eshelby = {{"type", "eshelby_spheroid"}, {"aspect_ratio", 15.0}, {"nu_source", "moduli"}, {"nu_property", "poisson_ratio"}};
  if (type == "dilute" || type == "mori_tanaka")
    return {{"type", type}, {"eshelby", eshelby}};
  if (type == "self_consistent")
    return {{"type", type}, {"shape", {{"type", "spheroid"}, {"aspect_ratio", 15.0}}}, {"relaxation", 0.8}};
  return {{"type", type}};
}

class Kernels : public ::testing::TestWithParam<std::string> {};

// with elastic phases one pass of the local composite is the global composite over the same phases
TEST_P(Kernels, LocaliseLikeTheGlobalComposite) {
  json doc = deck(injection, elastic_fibres());
  for (auto &e : doc["materials"])
    if (e["type"] == "orientation_phases")
      e["kernel"] = kernel_of(GetParam());
  json e = short_fibres(injection, elastic_fibres(), false);
  e["kernel"] = kernel_of(GetParam());
  json const strain = named(expand(e), "sf_strain");
  doc["materials"].push_back({{"type", "mean_field_composite_global"}, {"name", "same"}, {"phases", strain.at("phases")}, {"kernel", strain.at("kernel")}});
  auto ctx{build(doc)};
  step(*ctx, tensor2{}, uniaxial(0.001));
  EXPECT_EQ(ctx->find("sf_S_0") != nullptr, GetParam() == "dilute" || GetParam() == "mori_tanaka");
  EXPECT_LT(relative(ctx->get<tensor4>("sf", "tangent"), ctx->get<tensor4>("same", "tangent")), 1e-9);
}

INSTANTIATE_TEST_SUITE_P(All, Kernels, ::testing::Values("voigt", "reuss", "hill", "dilute", "mori_tanaka", "self_consistent"));

TEST(OrientationPhases, TheDiluteKernelReadsTheShape) {
  json e = short_fibres(injection, elastic_fibres(), false);
  e["kernel"]["type"] = "dilute";
  auto const entries = expand(e);
  EXPECT_EQ(std::ranges::count_if(entries, [](json const &x) { return x.at("type") == "eshelby_spheroid"; }), 25);
  EXPECT_EQ(named(entries, "sf_strain").at("kernel"), json({{"type", "dilute"}}));
  e["kernel"].erase("eshelby");
  EXPECT_THROW((void)expand(e), std::invalid_argument);
}

// self-consistent: the shape template along every node, the matrix a sphere unless "matrix_shape" says otherwise
TEST(OrientationPhases, TheSelfConsistentKernelOwnsTheShapes) {
  json e = short_fibres(injection, elastic_fibres(), false);
  e["kernel"] = kernel_of("self_consistent");
  auto const entries = expand(e);
  EXPECT_EQ(std::ranges::count_if(entries, [](json const &x) { return x.at("type") == "eshelby_spheroid"; }), 0);
  json const kernel = named(entries, "sf_strain").at("kernel");
  EXPECT_FALSE(kernel.contains("shape"));
  EXPECT_EQ(kernel.at("relaxation"), 0.8);
  ASSERT_EQ(kernel.at("shapes").size(), 26u);
  EXPECT_EQ(kernel.at("shapes").at("matrix"), json({{"type", "sphere"}}));
  auto const half{nm::orientation::half_nodes(nm::orientation::fit_acg(orientation_tensor(injection)), 11)};
  json const s3 = kernel.at("shapes").at("sf_3");
  EXPECT_EQ(s3.at("type"), "spheroid");
  for (int i{0}; i < 3; ++i)
    EXPECT_EQ(s3.at("direction")[i].get<T>(), half[3].first(i));
  e["kernel"]["matrix_shape"] = {{"type", "spheroid"}, {"aspect_ratio", 2.0}};
  EXPECT_EQ(named(expand(e), "sf_strain").at("kernel").at("shapes").at("matrix").at("aspect_ratio"), 2.0);
  e["kernel"].erase("shape");
  EXPECT_THROW((void)expand(e), std::invalid_argument);
}

/// Damage of the fibre phase closest to (first) and furthest from (second) the x1 axis.
std::pair<T, T> along_and_across(ctx_type &ctx) {
  auto const half{nm::orientation::half_nodes(nm::orientation::fit_acg(orientation_tensor(injection)), 11)};
  auto const by_x1 = [&](auto const &a, auto const &b) { return std::abs(a.first(0)) < std::abs(b.first(0)); };
  auto const q_along{std::ranges::max_element(half, by_x1) - half.begin()};
  auto const q_across{std::ranges::min_element(half, by_x1) - half.begin()};
  return {ctx.get<T>("fl_" + std::to_string(q_along), "damage"), ctx.get<T>("fl_" + std::to_string(q_across), "damage")};
}

TEST(DamagingFibres, FibresAlongTheLoadDamageFirst) {
  auto ctx{build(deck(injection, damaging_fibres(), true))};
  for (int n{1}; n <= 10; ++n)
    step(*ctx, uniaxial(0.0006 * (n - 1)), uniaxial(0.0006 * n));
  auto const [along, across]{along_and_across(*ctx)};
  EXPECT_GT(along, 0);
  EXPECT_LT(across, along);
}

// with the concentrations fixed the tangent sum c_r C_r : A_r is the exact derivative of the stress
TEST(DamagingFibres, TangentMatchesFiniteDifferences) {
  auto ctx{build(deck(injection, damaging_fibres(), true))};
  u::material_point_evaluator<policy> ev(*ctx, {.strain_source = "strain", .stress_source = "sf"});
  u::material_point_driver<policy> d(ev);
  for (int n{1}; n <= 10; ++n)
    ASSERT_TRUE(d.step(uniaxial(n * 6e-4), tensor2{}, u::load_control<T>::strain(), n - 1, 1).converged) << n;
  auto const [along, across]{along_and_across(*ctx)};
  ASSERT_GT(along, 0); // a loading step on damaging phases
  auto const res{d.step(uniaxial(10 * 6e-4 + 1e-6), tensor2{}, u::load_control<T>::strain(), 10, 1)};
  ASSERT_TRUE(res.converged);
  EXPECT_LT(d.tangent_error(res), 1e-4);
}

TEST(NonlinearFibres, ExpandIntoTheNonlinearComposite) {
  json e = nonlinear(short_fibres(injection, damaging_fibres(), true));
  json const composite = named(expand(e), "sf");
  EXPECT_EQ(composite.at("type"), "mean_field_composite_local_nonlinear");
  EXPECT_FALSE(composite.at("phases")[0].contains("used_tangent")); // an elastic matrix
  EXPECT_EQ(composite.at("phases")[4].at("used_tangent"), "sf_3::tangent");
  EXPECT_EQ(named(expand(e), "sf_3").at("tangent_source"), "fibre_3");
  e["matrix_used_tangent"] = "matrix::tangent";
  EXPECT_EQ(named(expand(e), "sf").at("phases")[0].at("used_tangent"), "matrix::tangent");
}

TEST(NonlinearFibres, NeedALocalTangentEdge) {
  json e = nonlinear(short_fibres(injection, damaging_fibres(), true));
  e["phase"].erase("tangent_edge");
  try {
    (void)expand(e);
    FAIL() << "a nonlinear localisation over a global tangent edge is a cycle";
  } catch (std::invalid_argument const &x) {
    EXPECT_NE(std::string{x.what()}.find("tangent_edge"), std::string::npos) << x.what();
  }
  e = short_fibres(injection, elastic_fibres(), false);
  e["localisation"] = "plastic";
  EXPECT_THROW((void)expand(e), std::invalid_argument);
  e["localisation"] = "linear";
  e["matrix_used_tangent"] = "matrix::tangent"; // only a nonlinear composite reads it
  EXPECT_THROW((void)expand(e), std::invalid_argument);
}

u::material_point_evaluator<policy>::config nonlinear_point() {
  return {.strain_source = "strain", .stress_source = "sf", .residual_property = "sf::residual", .max_passes = 200};
}

// before any damage the current tangents are the initial ones: the first pass is converged and linear
TEST(NonlinearFibres, BelowTheThresholdAreTheLinearLocalisation) {
  auto lin{build(deck(injection, damaging_fibres(), true))};
  auto nl{build(deck(injection, damaging_fibres(), true, true))};
  step(*lin, tensor2{}, uniaxial(0.0006));
  step(*nl, tensor2{}, uniaxial(0.0006));
  EXPECT_EQ(along_and_across(*nl).first, 0);
  EXPECT_LT(nl->get<T>("sf", "residual"), 1e-12);
  tensor2 const &sigma{lin->get<tensor2>("sf", "stress")};
  EXPECT_LT(tmech::norm(tensor2{nl->get<tensor2>("sf", "stress") - sigma}) / tmech::norm(sigma), 1e-12);
}

// a softening fibre attracts strain once it localises with its damaged tangent: more damage than linear
TEST(NonlinearFibres, ConvergedPassesSendTheStrainToTheDamagedFibres) {
  auto lin{build(deck(injection, damaging_fibres(), true))};
  auto nl{build(deck(injection, damaging_fibres(), true, true))};
  u::material_point_evaluator<policy> ev(*nl, nonlinear_point());
  u::material_point_driver<policy> d(ev);
  for (int n{1}; n <= 10; ++n) {
    step(*lin, uniaxial(0.0006 * (n - 1)), uniaxial(0.0006 * n));
    ASSERT_TRUE(d.step(uniaxial(n * 6e-4), tensor2{}, u::load_control<T>::strain(), n - 1, 1).converged) << n;
    EXPECT_LE(nl->get<T>("sf", "residual"), 1e-10) << n;
  }
  auto const [along, across]{along_and_across(*nl)};
  EXPECT_GT(along, 0);
  EXPECT_LT(across, along);
  EXPECT_GT(along, along_and_across(*lin).first);
  EXPECT_LT(nl->get<tensor2>("sf", "stress")(0, 0), lin->get<tensor2>("sf", "stress")(0, 0));
}

// the published tangent sum c_r C_r : A_r holds the concentrations fixed: first order in the step
TEST(NonlinearFibres, TangentMatchesFiniteDifferencesOnASmallStep) {
  auto ctx{build(deck(injection, damaging_fibres(), true, true))};
  u::material_point_evaluator<policy> ev(*ctx, nonlinear_point());
  u::material_point_driver<policy> d(ev);
  for (int n{1}; n <= 10; ++n)
    ASSERT_TRUE(d.step(uniaxial(n * 6e-4), tensor2{}, u::load_control<T>::strain(), n - 1, 1).converged) << n;
  ASSERT_GT(along_and_across(*ctx).first, 0);
  auto const res{d.step(uniaxial(10 * 6e-4 + 1e-6), tensor2{}, u::load_control<T>::strain(), 10, 1)};
  ASSERT_TRUE(res.converged);
  EXPECT_LT(d.tangent_error(res), 1e-2);
}

TEST(OrientationPhases, RejectsAMalformedEntry) {
  auto const message = [](json e) {
    ctx_type ctx;
    try {
      nm::create_from_json(ctx, e);
    } catch (std::exception const &x) {
      return std::string{x.what()};
    }
    return std::string{};
  };
  json e = short_fibres(injection, elastic_fibres(), false);
  e.erase("per_direction");
  EXPECT_NE(message(e).find("per_direction"), std::string::npos) << message(e);
  e = short_fibres(json::array({0.5, 0.5}), elastic_fibres(), false);
  EXPECT_NE(message(e).find("orientation"), std::string::npos) << message(e);
  e = short_fibres(injection, elastic_fibres(), false);
  e["precision"] = 4;
  EXPECT_NE(message(e).find("precision"), std::string::npos) << message(e);
  e = short_fibres(injection, json::array(), false);
  EXPECT_NE(message(e).find("per_direction"), std::string::npos) << message(e);
  e = short_fibres(injection, elastic_fibres(), false);
  e["kernel"] = "mori_tanaka"; // an object naming its type and carrying its data
  EXPECT_NE(message(e).find("kernel"), std::string::npos) << message(e);
  e["kernel"] = {{"type", "self_consistent"}};
  EXPECT_NE(message(e).find("shape"), std::string::npos) << message(e);
  e = short_fibres(injection, elastic_fibres(), false);
  e.erase("matrix_stress");
  EXPECT_NE(message(e).find("matrix_stress"), std::string::npos) << message(e);
}

} // namespace
