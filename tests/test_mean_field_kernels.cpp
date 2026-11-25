/// The mean-field kernels: strain concentration tensors of phases from their tangents, fractions and shapes.
#include <gtest/gtest.h>
#include <any>
#include <array>
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
#include "numsim-materials/umat/errors.h"
#include "test_support.h"

namespace {

namespace nm = numsim::materials;
namespace mf = numsim::materials::mean_field;
using policy = nm::material_policy_default;
using T = policy::value_type;
using PH = policy::ParameterHandler;
using spec = nm::object_spec<PH>;
using tensor4 = tmech::tensor<T, 3, 4>;
using json = nlohmann::json;
using numsim::materials::test::relative;

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

void expect_partition(std::vector<tensor4> const &A) {
  EXPECT_LT(relative(tensor4{(1 - c) * A[0] + c * A[1]}, Is()), 1e-12);
}

/// Hashin-Shtrikman zeta of a reference medium with moduli K, G.
T zeta(T K, T G) { return G * (9 * K + 8 * G) / (6 * (K + 2 * G)); }

TEST(MeanFieldKernels, VoigtIsEqualStrain) {
  two_phases ph;
  auto k{make("voigt")};
  EXPECT_FALSE(k->needs().tangents);
  auto const A{concentrations(*k, ph)};
  EXPECT_LT(relative(A[0], Is()), 1e-14);
  EXPECT_LT(relative(A[1], Is()), 1e-14);
}

TEST(MeanFieldKernels, ReussIsEqualStress) {
  two_phases ph;
  auto k{make("reuss")};
  auto const A{concentrations(*k, ph)};
  expect_partition(A);
  tensor4 const C_R{tmech::inv(tensor4{(1 - c) * tmech::inv(ph.Cm) + c * tmech::inv(ph.Ci)})};
  EXPECT_LT(relative(effective(ph, A), C_R), 1e-12);
  EXPECT_LT(relative(tensor4{tmech::dcontract(ph.Cm, A[0])}, tensor4{tmech::dcontract(ph.Ci, A[1])}), 1e-12);
}

TEST(MeanFieldKernels, HillIsTheMeanOfVoigtAndReuss) {
  two_phases ph;
  auto hill{make("hill")};
  auto reuss{make("reuss")};
  auto const A{concentrations(*hill, ph)}, R{concentrations(*reuss, ph)};
  EXPECT_LT(relative(A[1], tensor4{(Is() + R[1]) / 2}), 1e-12);
  expect_partition(A);
}

TEST(MeanFieldKernels, DiluteSpheresAreTheEshelbySolution) {
  two_phases ph;
  auto k{make("dilute")};
  EXPECT_TRUE(k->needs().eshelby);
  auto const A{concentrations(*k, ph)};
  expect_partition(A); // the matrix takes the rest of the strain
  T const vol{(3 * Km + 4 * Gm) / (3 * Ki + 4 * Gm)}, dev{(Gm + zeta(Km, Gm)) / (Gi + zeta(Km, Gm))};
  tensor4 const expected{vol * nm::vol_projector<T, 3>() + dev * nm::dev_projector<T, 3>()};
  EXPECT_LT(relative(A[1], expected), 1e-12);
}

TEST(MeanFieldKernels, MoriTanakaSpheresAreTheClosedForm) {
  two_phases ph;
  auto k{make("mori_tanaka")};
  auto const A{concentrations(*k, ph)};
  expect_partition(A);
  T const K{Km + c * (Ki - Km) * (3 * Km + 4 * Gm) / (3 * Km + 4 * Gm + 3 * (1 - c) * (Ki - Km))};
  T const G{Gm + c * (Gi - Gm) / (1 + (1 - c) * (Gi - Gm) / (Gm + zeta(Km, Gm)))};
  EXPECT_LT(relative(effective(ph, A), iso(K, G)), 1e-12);
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
  for (auto const *reference : {"anisotropic", "isotropised"}) {
    SCOPED_TRACE(reference);
    two_phases ph;
    PH p{self_consistent_parameters()};
    p.insert<std::string>("reference", reference);
    auto k{make("self_consistent", p)};
    EXPECT_FALSE(k->needs().eshelby); // its shapes are its own: S at the effective medium
    auto const A{concentrations(*k, ph)};
    expect_partition(A);
    auto const [K, G]{nm::isotropize(effective(ph, A))};
    EXPECT_NEAR((1 - c) * (Km - K) / (Km + 4 * G / 3) + c * (Ki - K) / (Ki + 4 * G / 3), 0, 1e-10);
    EXPECT_NEAR((1 - c) * (Gm - G) / (Gm + zeta(K, G)) + c * (Gi - G) / (Gi + zeta(K, G)), 0, 1e-10);
  }
}

/// Continuous fibres along x1 in a matrix (the inclusion study of #160): both phases cylinders, equal nu.
struct aligned_fibres {
  static constexpr T nu{0.3}, cf{0.3}, Em{3000.0}, Ef{200000.0};
  tensor4 Cm{iso(nm::bulk_modulus(Em, nu), nm::shear_modulus(Em, nu))}, Cf{iso(nm::bulk_modulus(Ef, nu), nm::shear_modulus(Ef, nu))};
  std::vector<mf::phase_view<T>> views() const { return {{1 - cf, &Cm, nullptr}, {cf, &Cf, nullptr}}; }
};

PH fibre_parameters(spec const &shape, std::string const &reference = {}) {
  PH p;
  p.insert<std::map<std::string, spec>>("shapes", {{"matrix", shape}, {"fibres", shape}});
  if (!reference.empty())
    p.insert<std::string>("reference", reference);
  p.insert<T>("tolerance", 1e-12);
  p.insert<int>("max_iterations", 1000);
  return p;
}

PH cylinder_x1() {
  PH p;
  p.insert<std::vector<T>>("direction", {1, 0, 0});
  return p;
}

tensor4 effective_of(mf::kernel<policy> &k, aligned_fibres const &ph) {
  k.bind({"matrix", "fibres"});
  std::vector<tensor4> A(2);
  auto const v{ph.views()};
  k.concentrations(v, A);
  EXPECT_LT(relative(tensor4{(1 - ph.cf) * A[0] + ph.cf * A[1]}, Is()), 1e-12);
  return tensor4{(1 - ph.cf) * tmech::dcontract(ph.Cm, A[0]) + ph.cf * tmech::dcontract(ph.Cf, A[1])};
}

tensor4 major_transpose(tensor4 const &C) { return tensor4{tmech::basis_change<tmech::sequence<3, 4, 1, 2>>(C)}; }

// equal nu: a uniaxial stress along the fibres strains both phases alike, so E1 is the rule of mixtures
TEST(MeanFieldKernels, SelfConsistentAlignedFibresFollowTheRuleOfMixtures) {
  for (std::string const reference : {"", "anisotropic", "isotropised"}) {
    SCOPED_TRACE(reference);
    aligned_fibres ph;
    auto k{mf::kernel_registry<policy>::instance().make(spec{"self_consistent", fibre_parameters(spec{"cylinder", cylinder_x1()}, reference)})};
    tensor4 const C{effective_of(*k, ph)};
    tensor4 const compliance{tmech::inv(C)};
    T const E1{1 / compliance(0, 0, 0, 0)};
    EXPECT_NEAR(E1 / ((1 - ph.cf) * ph.Em + ph.cf * ph.Ef), 1, 1e-10);
    EXPECT_LT(relative(C, major_transpose(C)), 1e-10);
  }
}

// cylinders in the transversely isotropic medium: Hill's (1965) self-consistent k, m, p and E2 from them
TEST(MeanFieldKernels, SelfConsistentAlignedFibresAreHillsTransverselyIsotropicSolution) {
  aligned_fibres ph;
  auto k{mf::kernel_registry<policy>::instance().make(spec{"self_consistent", fibre_parameters(spec{"cylinder", cylinder_x1()}, "anisotropic")})};
  tensor4 const C{effective_of(*k, ph)};
  std::array<T, 2> const fraction{1 - ph.cf, ph.cf}, E{ph.Em, ph.Ef};
  std::array<T, 2> G{}, k_r{};
  for (std::size_t r{0}; r < 2; ++r) {
    G[r] = nm::shear_modulus(E[r], ph.nu);
    k_r[r] = nm::bulk_modulus(E[r], ph.nu) + G[r] / 3; // plane-strain bulk modulus
  }
  auto const solve = [&](std::array<T, 2> const &moduli, T shift) { // the root of sum_r c_r (x_r - x) / (x_r + shift)
    T num{0}, den{0};
    for (std::size_t r{0}; r < 2; ++r) {
      num += fraction[r] * moduli[r] / (moduli[r] + shift);
      den += fraction[r] / (moduli[r] + shift);
    }
    return num / den;
  };
  T kt{k_r[0]}, m{G[0]}, p{G[0]};
  for (int it{0}; it < 10000; ++it) {
    kt = solve(k_r, m);
    m = solve(G, kt * m / (kt + 2 * m));
    p = solve(G, p);
  }
  T const E1{(1 - ph.cf) * ph.Em + ph.cf * ph.Ef}, E2{1 / (1 / (4 * kt) + 1 / (4 * m) + ph.nu * ph.nu / E1)};
  tensor4 const compliance{tmech::inv(C)};
  EXPECT_NEAR(C(0, 1, 0, 1) / p, 1, 1e-10);         // G12
  EXPECT_NEAR(C(1, 2, 1, 2) / m, 1, 1e-10);         // G23
  EXPECT_NEAR(1 / compliance(1, 1, 1, 1) / E2, 1, 1e-10);
  EXPECT_NEAR((C(1, 1, 1, 1) + C(1, 1, 2, 2)) / 2 / kt, 1, 1e-10);
}

T energy(tensor4 const &C, tmech::tensor<T, 3, 2> const &e) { return tmech::dcontract(e, tmech::tensor<T, 3, 2>{tmech::dcontract(C, e)}); }

// aligned spheroids: major symmetric, between Voigt and Reuss, and settled in the quadrature order
TEST(MeanFieldKernels, SelfConsistentAlignedSpheroidsAreSymmetricAndBounded) {
  aligned_fibres ph;
  PH shape;
  shape.insert<T>("aspect_ratio", 5);
  auto const at = [&](int order) {
    PH p{fibre_parameters(spec{"spheroid", shape}, "anisotropic")};
    p.insert<int>("quadrature_order", order);
    auto k{mf::kernel_registry<policy>::instance().make(spec{"self_consistent", p})};
    return effective_of(*k, ph);
  };
  tensor4 const C{at(12)}, coarse{at(4)}, fine{at(24)};
  EXPECT_LT(relative(C, major_transpose(C)), 1e-10);
  EXPECT_LT(relative(C, fine), 1e-10);
  EXPECT_LT(relative(C, fine), relative(coarse, fine));
  tensor4 const voigt{(1 - ph.cf) * ph.Cm + ph.cf * ph.Cf}, reuss{tmech::inv(tensor4{(1 - ph.cf) * tmech::inv(ph.Cm) + ph.cf * tmech::inv(ph.Cf)})};
  std::vector<tmech::tensor<T, 3, 2>> strains(4);
  strains[0](0, 0) = 1;
  strains[1](1, 1) = 1;
  strains[2](0, 1) = strains[2](1, 0) = 1;
  strains[3](0, 0) = 0.3;
  strains[3](1, 1) = -1;
  strains[3](2, 2) = 0.5;
  strains[3](1, 2) = strains[3](2, 1) = 0.2;
  for (auto const &e : strains) {
    EXPECT_GT(energy(voigt, e), energy(C, e));
    EXPECT_GT(energy(C, e), energy(reuss, e));
  }
}

TEST(MeanFieldKernels, SelfConsistentReferenceIsValidated) {
  PH p{self_consistent_parameters()};
  p.insert<std::string>("reference", "isotropic");
  try {
    make("self_consistent", p);
    FAIL() << "an unknown reference was accepted";
  } catch (std::invalid_argument const &e) {
    std::string const what{e.what()};
    EXPECT_NE(what.find("anisotropic"), std::string::npos) << what;
    EXPECT_NE(what.find("isotropised"), std::string::npos) << what;
  }
  PH q{self_consistent_parameters()};
  q.insert<int>("quadrature_order", 1);
  EXPECT_THROW(make("self_consistent", q), std::invalid_argument);
}

TEST(MeanFieldKernels, SelfConsistentThrowsWhenItDoesNotConverge) {
  two_phases ph;
  PH p{self_consistent_parameters()};
  p.insert<int>("max_iterations", 2);
  auto k{make("self_consistent", p)};
  EXPECT_THROW(concentrations(*k, ph), nm::umat::cutback_error);
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
  json const j = {{"type", "self_consistent"}, {"relaxation", 0.8}, {"reference", "isotropised"}, {"quadrature_order", 8},
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
  json unknown = j;
  unknown["reference"] = "voigt";
  EXPECT_THROW((void)mf::kernel_registry<policy>::instance().make(std::any_cast<spec>(registry.read(typeid(spec), unknown, "kernel"))),
               std::invalid_argument);
}

} // namespace
