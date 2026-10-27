/// The mean-field kernels: strain concentration tensors of phases from their tangents, fractions and shapes.
#include <gtest/gtest.h>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <tmech/tmech.h>

#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/core/traits.h"
#include "numsim-materials/io/json_parameter_converter.h"
#include "numsim-materials/io/object_readers.h"
#include "numsim-materials/mean_field/kernel_registry.h"

namespace {

namespace nm = numsim::materials;
namespace mf = numsim::materials::mean_field;
using policy = nm::material_policy_default;
using T = policy::value_type;
using PH = policy::ParameterHandler;
using spec = nm::object_spec<PH>;
using tensor4 = tmech::tensor<T, 3, 4>;
using json = nlohmann::json;

constexpr T Km{3.0}, Gm{1.0}, Ki{60.0}, Gi{30.0}, c{0.3};

tensor4 const &Is() { return nm::sym_identity<T, 3>(); }
tensor4 iso(T K, T G) { return nm::isotropic_stiffness<T, 3>(K, G); }
T nu_m() { return (3 * Km - 2 * Gm) / (2 * (3 * Km + Gm)); }

/// Matrix and spheres as the nodes hand them to a kernel: matrix first.
struct two_phases {
  tensor4 Cm{iso(Km, Gm)}, Ci{iso(Ki, Gi)}, Sm{nm::eshelby::sphere(nu_m())}, Si{nm::eshelby::sphere(nu_m())};
  std::vector<mf::phase_view<T>> views() const { return {{1 - c, &Cm, &Sm}, {c, &Ci, &Si}}; }
};

std::unique_ptr<mf::kernel<policy>> make(std::string type, PH p = {}) {
  auto k{mf::kernel_registry<policy>::instance().make(spec{std::move(type), std::move(p)})};
  k->bind({"matrix", "spheres"});
  return k;
}

std::vector<tensor4> concentrations(mf::kernel<policy> &k, two_phases const &ph) {
  std::vector<tensor4> A(2);
  auto const v{ph.views()};
  k.concentrations(v, A);
  return A;
}

tensor4 effective(two_phases const &ph, std::vector<tensor4> const &A) {
  return tensor4{(1 - c) * tmech::dcontract(ph.Cm, A[0]) + c * tmech::dcontract(ph.Ci, A[1])};
}

T rel(tensor4 const &a, tensor4 const &b) { return tmech::norm(tensor4{a - b}) / tmech::norm(b); }

void expect_partition(std::vector<tensor4> const &A) {
  EXPECT_LT(rel(tensor4{(1 - c) * A[0] + c * A[1]}, Is()), 1e-12);
}

/// Hashin-Shtrikman zeta of a reference medium with moduli K, G.
T zeta(T K, T G) { return G * (9 * K + 8 * G) / (6 * (K + 2 * G)); }

TEST(MeanFieldKernels, VoigtIsEqualStrain) {
  two_phases ph;
  auto k{make("voigt")};
  EXPECT_FALSE(k->needs().tangents);
  auto const A{concentrations(*k, ph)};
  EXPECT_LT(rel(A[0], Is()), 1e-14);
  EXPECT_LT(rel(A[1], Is()), 1e-14);
}

TEST(MeanFieldKernels, ReussIsEqualStress) {
  two_phases ph;
  auto k{make("reuss")};
  auto const A{concentrations(*k, ph)};
  expect_partition(A);
  tensor4 const C_R{tmech::inv(tensor4{(1 - c) * tmech::inv(ph.Cm) + c * tmech::inv(ph.Ci)})};
  EXPECT_LT(rel(effective(ph, A), C_R), 1e-12);
  EXPECT_LT(rel(tensor4{tmech::dcontract(ph.Cm, A[0])}, tensor4{tmech::dcontract(ph.Ci, A[1])}), 1e-12);
}

TEST(MeanFieldKernels, HillIsTheMeanOfVoigtAndReuss) {
  two_phases ph;
  auto hill{make("hill")};
  auto reuss{make("reuss")};
  auto const A{concentrations(*hill, ph)}, R{concentrations(*reuss, ph)};
  EXPECT_LT(rel(A[1], tensor4{(Is() + R[1]) / 2}), 1e-12);
  expect_partition(A);
}

TEST(MeanFieldKernels, DiluteSpheresAreTheEshelbySolution) {
  two_phases ph;
  auto k{make("dilute")};
  EXPECT_TRUE(k->needs().eshelby);
  auto const A{concentrations(*k, ph)};
  EXPECT_LT(rel(A[0], Is()), 1e-14);
  T const vol{(3 * Km + 4 * Gm) / (3 * Ki + 4 * Gm)}, dev{(Gm + zeta(Km, Gm)) / (Gi + zeta(Km, Gm))};
  tensor4 const expected{vol * nm::vol_projector<T, 3>() + dev * nm::dev_projector<T, 3>()};
  EXPECT_LT(rel(A[1], expected), 1e-12);
}

TEST(MeanFieldKernels, MoriTanakaSpheresAreTheClosedForm) {
  two_phases ph;
  auto k{make("mori_tanaka")};
  auto const A{concentrations(*k, ph)};
  expect_partition(A);
  T const K{Km + c * (Ki - Km) * (3 * Km + 4 * Gm) / (3 * Km + 4 * Gm + 3 * (1 - c) * (Ki - Km))};
  T const G{Gm + c * (Gi - Gm) / (1 + (1 - c) * (Gi - Gm) / (Gm + zeta(Km, Gm)))};
  EXPECT_LT(rel(effective(ph, A), iso(K, G)), 1e-12);
}

PH self_consistent_parameters() {
  PH p;
  p.insert<std::map<std::string, spec>>("shapes", {{"matrix", spec{"sphere", {}}}, {"spheres", spec{"sphere", {}}}});
  p.insert<T>("tolerance", 1e-12);
  return p;
}

// Hill's self-consistent equations for spheres: sum_r c_r (K_r - K)/(K_r + 4/3 G) = 0 and
// sum_r c_r (G_r - G)/(G_r + zeta(K, G)) = 0 at the effective moduli
TEST(MeanFieldKernels, SelfConsistentSpheresSolveHillsEquations) {
  two_phases ph;
  auto k{make("self_consistent", self_consistent_parameters())};
  EXPECT_FALSE(k->needs().eshelby); // its shapes are its own: S at the effective medium
  auto const A{concentrations(*k, ph)};
  expect_partition(A);
  auto const [K, G]{nm::isotropize(effective(ph, A))};
  EXPECT_NEAR((1 - c) * (Km - K) / (Km + 4 * G / 3) + c * (Ki - K) / (Ki + 4 * G / 3), 0, 1e-10);
  EXPECT_NEAR((1 - c) * (Gm - G) / (Gm + zeta(K, G)) + c * (Gi - G) / (Gi + zeta(K, G)), 0, 1e-10);
}

TEST(MeanFieldKernels, SelfConsistentThrowsWhenItDoesNotConverge) {
  two_phases ph;
  PH p{self_consistent_parameters()};
  p.insert<int>("max_iterations", 2);
  auto k{make("self_consistent", p)};
  EXPECT_THROW(concentrations(*k, ph), std::runtime_error);
}

TEST(MeanFieldKernels, SelfConsistentNeedsAShapePerPhase) {
  PH p;
  p.insert<std::map<std::string, spec>>("shapes", {{"matrix", spec{"sphere", {}}}});
  auto k{mf::kernel_registry<policy>::instance().make(spec{"self_consistent", p})};
  EXPECT_THROW(k->bind({"matrix", "spheres"}), std::invalid_argument);
}

TEST(MeanFieldKernels, ParametersAreValidated) {
  PH p{self_consistent_parameters()};
  p.insert<T>("relaxation", 1.5);
  EXPECT_THROW(make("self_consistent", p), std::invalid_argument);
  try {
    make("mori_tanak");
    FAIL() << "an unknown kernel was accepted";
  } catch (std::invalid_argument const &e) {
    EXPECT_NE(std::string{e.what()}.find("mori_tanaka"), std::string::npos) << e.what();
  }
}

// a kernel is a JSON object: its type and its data, shapes included, read through the material machinery
TEST(MeanFieldKernels, ReadFromAJsonObject) {
  auto registry{nm::make_default_json_registry<json>()};
  nm::add_object_readers<policy>(registry);
  json const j = {{"type", "self_consistent"}, {"relaxation", 0.8},
                  {"shapes", {{"matrix", {{"type", "sphere"}}}, {"spheres", {{"type", "spheroid"}, {"aspect_ratio", 1.0}}}}}};
  auto const s{std::any_cast<spec>(registry.read(typeid(spec), j, "kernel"))};
  EXPECT_EQ(s.type, "self_consistent");
  auto k{mf::kernel_registry<policy>::instance().make(s)};
  k->bind({"matrix", "spheres"});
  two_phases ph; // a spheroid of aspect ratio one is a sphere
  auto const A{concentrations(*k, ph)};
  expect_partition(A);
  json bad = j;
  bad["shapes"]["spheres"] = {{"type", "cube"}};
  EXPECT_THROW((void)registry.read(typeid(spec), bad, "kernel"), std::runtime_error);
}

} // namespace
