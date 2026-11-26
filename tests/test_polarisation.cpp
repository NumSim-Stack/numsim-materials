/// Hill's polarisation tensor of ellipsoids in an anisotropic medium, as the quadrature rules of the shapes.
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>
#include <tmech/tmech.h>

#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/core/lebedev.h"
#include "numsim-materials/core/polarisation.h"
#include "numsim-materials/core/traits.h"
#include "numsim-materials/mean_field/shapes.h"
#include "test_support.h"

namespace {

namespace nm = numsim::materials;
namespace mf = numsim::materials::mean_field;
namespace pol = numsim::materials::polarisation;
using policy = nm::material_policy_default;
using T = policy::value_type;
using PH = policy::ParameterHandler;
using spec = nm::object_spec<PH>;
using tensor1 = tmech::tensor<T, 3, 1>;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using numsim::materials::test::relative;

constexpr T inf{std::numeric_limits<T>::infinity()};

tensor2 identity() { return tensor2{tmech::eye<T, 3, 2>()}; }

tensor4 hill(std::vector<pol::node<T>> const &rule, tensor4 const &C) { return pol::hill<T>(rule, C); }

pol::scheme gl(std::size_t order) { return {pol::quadrature::legendre, order}; }
pol::scheme leb(std::size_t degree) { return {pol::quadrature::lebedev, degree}; }

std::unique_ptr<mf::shape<policy>> make_shape(std::string type, PH p = {}) {
  return mf::shape_registry<policy>().make(spec{std::move(type), std::move(p)});
}

PH with_direction(PH p, std::vector<T> d) {
  p.insert<std::vector<T>>("direction", std::move(d));
  return p;
}

PH with_aspect_ratio(T ar) {
  PH p;
  p.insert<T>("aspect_ratio", ar);
  return p;
}

/// A transversely isotropic stiffness about x1, roughly that of aligned stiff fibres.
tensor4 fibre_medium() {
  tensor4 C;
  T const n{60000}, l{3000}, k{6000}, m{2000}, p{2500};
  for (std::size_t i{0}; i < 3; ++i)
    for (std::size_t j{0}; j < 3; ++j) {
      bool const ia{i == 0}, ja{j == 0};
      C(i, i, j, j) = ia && ja ? n : (ia || ja ? l : (i == j ? k + m : k - m));
      if (i != j) {
        T const g{ia || ja ? p : m};
        C(i, j, i, j) = C(i, j, j, i) = g;
      }
    }
  return C;
}

TEST(Polarisation, GaussLegendreIntegratesPolynomialsOfDegreeTwoNMinusOne) {
  auto const gl{pol::gauss_legendre<T>(5)};
  T w{0}, x8{0};
  for (auto const &[x, wx] : gl) {
    w += wx;
    x8 += wx * std::pow(x, 8);
  }
  EXPECT_NEAR(w, 2, 1e-14);
  EXPECT_NEAR(x8, T{2} / 9, 1e-14);
}

// in an isotropic medium S = P : C is the closed form of every shape, elongated and flat ones included
TEST(Polarisation, InAnIsotropicMediumItIsTheClosedFormEshelbyTensor) {
  T const K{2.5}, G{1.15}, nu{nm::poisson_ratio(K, G)};
  tensor4 const C{nm::isotropic_stiffness<T, 3>(K, G)};
  PH ellipsoid_321, ellipsoid_flat;
  ellipsoid_321.insert<std::vector<T>>("semi_axes", {3, 2, 1});
  ellipsoid_flat.insert<std::vector<T>>("semi_axes", {1, 20, 4});
  std::vector<std::pair<std::string, std::unique_ptr<mf::shape<policy>>>> shapes;
  shapes.emplace_back("sphere", make_shape("sphere"));
  shapes.emplace_back("cylinder", make_shape("cylinder", with_direction({}, {1, 2, -0.5})));
  for (T ar : {2.0, 5.0, 100.0, 1000.0, 0.2, 0.01})
    shapes.emplace_back("spheroid " + std::to_string(ar), make_shape("spheroid", with_direction(with_aspect_ratio(ar), {0.3, -1, 2})));
  shapes.emplace_back("ellipsoid 3 2 1", make_shape("ellipsoid", ellipsoid_321));
  shapes.emplace_back("ellipsoid 1 20 4", make_shape("ellipsoid", ellipsoid_flat));
  for (auto const &[name, shape] : shapes) {
    SCOPED_TRACE(name);
    tensor4 const S{tmech::dcontract(hill(shape->polarisation_rule(gl(12)), C), C)};
    EXPECT_LT(relative(S, shape->eshelby(nu)), 1e-10);
  }
}

TEST(Polarisation, AnEllipticCylinderIsTheClosedForm) {
  T const nu{0.3};
  tensor4 const C{nm::isotropic_stiffness<T, 3>(nm::bulk_modulus(T{1}, nu), nm::shear_modulus(T{1}, nu))};
  tensor4 const S{tmech::dcontract(hill(pol::legendre<T>({2, inf, 0.5}, identity(), 12), C), C)};
  EXPECT_LT(relative(S, nm::eshelby::ellipsoid(nu, T{2}, inf, T{0.5})), 1e-10);
}

/// Independent reference: midpoint rule in z = cos(theta) about the global x3 axis (Richardson-extrapolated),
/// trapezoid rule in phi, plain loops and an own 3x3 inverse.
tensor4 brute_force_hill(std::array<T, 3> const &a, tensor2 const &R, tensor4 const &C, std::size_t nz, std::size_t nphi) {
  constexpr T pi{std::numbers::pi_v<T>};
  auto const integrate = [&](std::size_t n) {
    std::array<T, 81> X{};
    for (std::size_t iz{0}; iz < n; ++iz) {
      T const z{-1 + T(2 * iz + 1) / T(n)}, s{std::sqrt(1 - z * z)};
      for (std::size_t ip{0}; ip < nphi; ++ip) {
        T const phi{2 * pi * T(ip) / T(nphi)};
        T const xi[3]{s * std::cos(phi), s * std::sin(phi), z};
        T Axi2{0};
        for (std::size_t i{0}; i < 3; ++i) {
          T const local{R(0, i) * xi[0] + R(1, i) * xi[1] + R(2, i) * xi[2]};
          Axi2 += a[i] * a[i] * local * local;
        }
        T const w{a[0] * a[1] * a[2] / (Axi2 * std::sqrt(Axi2)) * (2 / T(n)) * (2 * pi / T(nphi)) / (4 * pi)};
        T K[3][3]{};
        for (std::size_t i{0}; i < 3; ++i)
          for (std::size_t k{0}; k < 3; ++k)
            for (std::size_t j{0}; j < 3; ++j)
              for (std::size_t l{0}; l < 3; ++l)
                K[i][k] += C(i, j, k, l) * xi[j] * xi[l];
        T const det{K[0][0] * (K[1][1] * K[2][2] - K[1][2] * K[2][1]) - K[0][1] * (K[1][0] * K[2][2] - K[1][2] * K[2][0]) +
                    K[0][2] * (K[1][0] * K[2][1] - K[1][1] * K[2][0])};
        T N[3][3];
        for (std::size_t i{0}; i < 3; ++i)
          for (std::size_t j{0}; j < 3; ++j)
            N[i][j] = (K[(j + 1) % 3][(i + 1) % 3] * K[(j + 2) % 3][(i + 2) % 3] - K[(j + 1) % 3][(i + 2) % 3] * K[(j + 2) % 3][(i + 1) % 3]) / det;
        for (std::size_t i{0}; i < 3; ++i)
          for (std::size_t j{0}; j < 3; ++j)
            for (std::size_t k{0}; k < 3; ++k)
              for (std::size_t l{0}; l < 3; ++l)
                X[27 * i + 9 * j + 3 * k + l] += w * N[i][k] * xi[j] * xi[l];
      }
    }
    return X;
  };
  auto const coarse{integrate(nz)}, fine{integrate(2 * nz)};
  auto const X = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
    std::size_t const m{27 * i + 9 * j + 3 * k + l};
    return (4 * fine[m] - coarse[m]) / 3;
  };
  tensor4 P;
  for (std::size_t i{0}; i < 3; ++i)
    for (std::size_t j{0}; j < 3; ++j)
      for (std::size_t k{0}; k < 3; ++k)
        for (std::size_t l{0}; l < 3; ++l)
          P(i, j, k, l) = (X(i, j, k, l) + X(j, i, k, l) + X(i, j, l, k) + X(j, i, l, k)) / 4;
  return P;
}

tensor2 tilt() {
  tmech::tensor<T, 3, 1> axis;
  axis(0) = 1;
  axis(1) = 2;
  axis(2) = 3;
  return nm::rotation(tensor1{axis / tmech::norm(axis)}, T{0.8});
}

// a tilted triaxial ellipsoid in the fibre medium against the independent integration (brute grid good to ~3e-8)
TEST(Polarisation, MatchesAnIndependentIntegrationForATiltedEllipsoid) {
  tensor4 const C{fibre_medium()};
  std::array<T, 3> const a{4, 2, 1};
  tensor2 const R{tilt()};
  tensor4 const reference{brute_force_hill(a, R, C, 400, 64)};
  EXPECT_LT(relative(hill(pol::legendre<T>(a, R, 12), C), reference), 1e-6);
  EXPECT_LT(relative(hill(pol::legendre<T>(a, R, 24), C), reference), 1e-7);
  EXPECT_LT(relative(hill(pol::lebedev<T>(a, R, 77), C), reference), 5e-6);
}

// the sphere's default rule does not single out an axis: as accurate in a tilted medium as in an aligned one
// (P of the rotated medium is the rotated P; brute grid good to ~3e-8)
TEST(Polarisation, ASphereIsAccurateInATiltedMedium) {
  auto const sphere{make_shape("sphere")};
  tensor4 const C{fibre_medium()}, reference{brute_force_hill({1, 1, 1}, identity(), C, 400, 128)};
  EXPECT_LT(relative(hill(sphere->polarisation_rule(gl(12)), C), reference), 1e-7);
  EXPECT_LT(relative(hill(sphere->polarisation_rule(gl(12)), nm::rotate(C, tilt())), nm::rotate(reference, tilt())), 1e-7);
}

// spheroids tilted against the fibre medium: the error falls with the order (the medium's anisotropy limits the rate)
TEST(Polarisation, ConvergesWithTheQuadratureOrderInAnAnisotropicMedium) {
  tensor4 const C{fibre_medium()};
  for (T ar : {0.1, 2.0, 20.0}) {
    SCOPED_TRACE(ar);
    auto const shape{make_shape("spheroid", with_direction(with_aspect_ratio(ar), {1, 1, 0}))};
    tensor4 const reference{hill(shape->polarisation_rule(gl(48)), C)};
    T const e4{relative(hill(shape->polarisation_rule(gl(4)), C), reference)}, e12{relative(hill(shape->polarisation_rule(gl(12)), C), reference)},
        e24{relative(hill(shape->polarisation_rule(gl(24)), C), reference)};
    EXPECT_LT(e12, e4);
    EXPECT_LT(e24, e12);
    EXPECT_LT(e24, 1e-6);
  }
}

TEST(Polarisation, RejectsInvalidShapes) {
  EXPECT_THROW((void)pol::legendre<T>({1, 1, 1}, identity(), 1), std::invalid_argument);
  EXPECT_THROW((void)pol::legendre<T>({1, 0, 1}, identity(), 8), std::invalid_argument);
  EXPECT_THROW((void)pol::legendre<T>({inf, inf, 1}, identity(), 8), std::invalid_argument);
}

// mapped Lebedev: one node per antipodal pair (Gamma is even), weights summing to one
TEST(Polarisation, LebedevKeepsOneNodePerAntipodalPair) {
  for (auto const d : nm::lebedev::precisions()) {
    auto const rule{pol::lebedev<T>({5, 2, 1}, identity(), d)};
    EXPECT_EQ(rule.size(), nm::lebedev::rule(d).size() / 2) << d;
    T sum{0};
    for (auto const &n : rule)
      sum += n.weight;
    EXPECT_NEAR(sum, 1, 1e-12) << d;
  }
}

TEST(Polarisation, LebedevIsTheClosedFormInAnIsotropicMedium) {
  T const K{2.5}, G{1.15}, nu{nm::poisson_ratio(K, G)};
  tensor4 const C{nm::isotropic_stiffness<T, 3>(K, G)};
  PH ellipsoid_321;
  ellipsoid_321.insert<std::vector<T>>("semi_axes", {3, 2, 1});
  auto const sphere{make_shape("sphere")}, cylinder{make_shape("cylinder", with_direction({}, {1, 2, -0.5}))},
      spheroid{make_shape("spheroid", with_direction(with_aspect_ratio(2), {0.3, -1, 2}))}, ellipsoid{make_shape("ellipsoid", ellipsoid_321)};
  EXPECT_LT(relative(tensor4{tmech::dcontract(hill(sphere->polarisation_rule(leb(11)), C), C)}, sphere->eshelby(nu)), 1e-12);
  EXPECT_LT(relative(tensor4{tmech::dcontract(hill(cylinder->polarisation_rule(leb(23)), C), C)}, cylinder->eshelby(nu)), 1e-12);
  EXPECT_LT(relative(tensor4{tmech::dcontract(hill(spheroid->polarisation_rule(leb(77)), C), C)}, spheroid->eshelby(nu)), 1e-10);
  EXPECT_LT(relative(tensor4{tmech::dcontract(hill(ellipsoid->polarisation_rule(leb(77)), C), C)}, ellipsoid->eshelby(nu)), 1e-8);
}

// both schemes converge to the same P; for a sphere in the fibre medium Lebedev gets there with fewer nodes
TEST(Polarisation, LebedevAndLegendreAgreeInAnAnisotropicMedium) {
  tensor4 const C{fibre_medium()};
  auto const sphere{make_shape("sphere")};
  tensor4 const reference{hill(sphere->polarisation_rule(gl(48)), C)};
  EXPECT_LT(relative(hill(sphere->polarisation_rule(leb(59)), C), reference), 1e-8);
  auto const spheroid{make_shape("spheroid", with_direction(with_aspect_ratio(2), {1, 1, 0}))};
  tensor4 const ref2{hill(spheroid->polarisation_rule(gl(48)), C)};
  T const e23{relative(hill(spheroid->polarisation_rule(leb(23)), C), ref2)}, e41{relative(hill(spheroid->polarisation_rule(leb(41)), C), ref2)},
      e77{relative(hill(spheroid->polarisation_rule(leb(77)), C), ref2)};
  EXPECT_LT(e41, e23);
  EXPECT_LT(e77, e41);
  EXPECT_LT(e77, 1e-7);
}

// an infinite axis under Lebedev: the line rule has enough points at every degree
TEST(Polarisation, LebedevOnACylinderIsTheClosedForm) {
  T const nu{0.3};
  tensor4 const C{nm::isotropic_stiffness<T, 3>(nm::bulk_modulus(T{1}, nu), nm::shear_modulus(T{1}, nu))};
  tensor1 e1;
  e1(0) = 1;
  for (auto const d : nm::lebedev::precisions()) {
    SCOPED_TRACE(d);
    EXPECT_LT(relative(tensor4{tmech::dcontract(hill(pol::lebedev<T>({inf, 1, 1}, identity(), d), C), C)}, nm::eshelby::cylinder(nu, e1)), 1e-10);
    EXPECT_LT(relative(tensor4{tmech::dcontract(hill(pol::lebedev<T>({inf, 2, 0.5}, identity(), d), C), C)},
                       nm::eshelby::ellipsoid(nu, inf, T{2}, T{0.5})),
              1e-10);
  }
}

TEST(Polarisation, LebedevRejectsADegreeWithoutATable) {
  EXPECT_THROW((void)pol::lebedev<T>({1, 1, 1}, identity(), 12), std::invalid_argument);
  EXPECT_THROW((void)pol::lebedev<T>({inf, inf, 1}, identity(), 23), std::invalid_argument);
}

} // namespace
