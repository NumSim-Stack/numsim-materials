/// Hill's polarisation tensor of ellipsoids in an anisotropic medium, as the quadrature rules of the shapes.
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
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

// P carries the symmetries of a compliance in any medium
TEST(Polarisation, IsSymmetricInAnAnisotropicMedium) {
  tensor4 const C{fibre_medium()};
  auto const shape{make_shape("spheroid", with_direction(with_aspect_ratio(5), {1, 1, 0}))};
  tensor4 const P{hill(shape->polarisation_rule(gl(12)), C)};
  EXPECT_LT(relative(tensor4{tmech::basis_change<tmech::sequence<3, 4, 1, 2>>(P)}, P), 1e-13);
  EXPECT_LT(relative(tensor4{tmech::basis_change<tmech::sequence<2, 1, 3, 4>>(P)}, P), 1e-13);
  EXPECT_LT(relative(tensor4{tmech::basis_change<tmech::sequence<1, 2, 4, 3>>(P)}, P), 1e-13);
}

// spheroids tilted against the fibre medium: the error falls with the order (the medium's anisotropy limits the rate)
TEST(Polarisation, ConvergesWithTheQuadratureOrderInAnAnisotropicMedium) {
  tensor4 const C{fibre_medium()};
  for (T ar : {0.1, 1.0, 20.0}) {
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

TEST(Polarisation, LebedevRejectsADegreeWithoutATable) {
  EXPECT_THROW((void)pol::lebedev<T>({1, 1, 1}, identity(), 12), std::invalid_argument);
  EXPECT_THROW((void)pol::lebedev<T>({inf, inf, 1}, identity(), 23), std::invalid_argument);
}

} // namespace
