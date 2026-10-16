#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <numbers>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/core/tensor_identities.h"

namespace {

namespace nm = numsim::materials;
namespace es = numsim::materials::eshelby;
using T = double;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using vector = tmech::tensor<T, 3, 1>;

constexpr T nu{0.3};
constexpr T inf{std::numeric_limits<T>::infinity()};

T diff(tensor4 const &a, tensor4 const &b) { return tmech::norm(tensor4{a - b}) / tmech::norm(a); }

/// S_iikk = (1 + nu) / (1 - nu) for every ellipsoid in an isotropic matrix.
T trace(tensor4 const &S) {
  auto const I{tmech::eye<T, 3, 2>()};
  return tmech::dcontract(I, tensor2{tmech::dcontract(S, I)});
}

void expect_minor_symmetric(tensor4 const &S) {
  EXPECT_NEAR(tmech::norm(tensor4{S - tmech::basis_change<tmech::sequence<2, 1, 3, 4>>(S)}), 0, 1e-14);
  EXPECT_NEAR(tmech::norm(tensor4{S - tmech::basis_change<tmech::sequence<1, 2, 4, 3>>(S)}), 0, 1e-14);
}

TEST(Eshelby, Sphere) {
  tensor4 const S{es::sphere<T>(nu)};
  EXPECT_NEAR(S(0, 0, 0, 0), (7 - 5 * nu) / (15 * (1 - nu)), 1e-15);
  EXPECT_NEAR(S(0, 0, 1, 1), (5 * nu - 1) / (15 * (1 - nu)), 1e-15);
  EXPECT_NEAR(S(0, 1, 0, 1), (4 - 5 * nu) / (15 * (1 - nu)), 1e-15);
  EXPECT_NEAR(S(0, 0, 0, 1), 0, 1e-15);
  expect_minor_symmetric(S);
  EXPECT_NEAR(trace(S), (1 + nu) / (1 - nu), 1e-14);
}

TEST(Eshelby, CircularCylinderAlongX3) {
  tensor4 const S{es::cylinder<T>(nu, 2)};
  T const f{1 / (8 * (1 - nu))};
  EXPECT_NEAR(S(0, 0, 0, 0), (5 - 4 * nu) * f, 1e-15);
  EXPECT_NEAR(S(1, 1, 1, 1), (5 - 4 * nu) * f, 1e-15);
  EXPECT_NEAR(S(0, 0, 1, 1), (4 * nu - 1) * f, 1e-15);
  EXPECT_NEAR(S(0, 0, 2, 2), nu / (2 * (1 - nu)), 1e-15);
  EXPECT_NEAR(S(0, 1, 0, 1), (3 - 4 * nu) * f, 1e-15);
  EXPECT_NEAR(S(0, 2, 0, 2), 0.25, 1e-15);
  EXPECT_NEAR(S(2, 2, 2, 2), 0, 1e-15);
  EXPECT_NEAR(S(2, 2, 0, 0), 0, 1e-15);
  expect_minor_symmetric(S);
  EXPECT_NEAR(trace(S), (1 + nu) / (1 - nu), 1e-14);
}

TEST(Eshelby, CylinderAxisAndDirectionAgree) {
  vector e2;
  e2(1) = 1;
  EXPECT_NEAR(diff(es::cylinder<T>(nu, e2), es::cylinder<T>(nu, 1)), 0, 1e-14);
  tensor4 const along_1{es::cylinder<T>(nu, 0)};
  EXPECT_NEAR(along_1(1, 1, 1, 1), (5 - 4 * nu) / (8 * (1 - nu)), 1e-15); // the transverse plane is now 2-3
  EXPECT_NEAR(along_1(0, 0, 0, 0), 0, 1e-15);
  // a cylinder along (1,1,1)/sqrt3 is the x1 one rotated there
  vector d;
  d(0) = d(1) = d(2) = 1 / std::sqrt(3.0);
  EXPECT_NEAR(diff(es::cylinder<T>(nu, d), nm::rotate(along_1, nm::rotation_to(d))), 0, 1e-13);
}

TEST(Eshelby, EllipticCylinderReducesToTheCircularOne) {
  EXPECT_NEAR(diff(es::elliptic_cylinder<T>(nu, 1.0, 1.0, 2), es::cylinder<T>(nu, 2)), 0, 1e-14);
  tensor4 const S{es::elliptic_cylinder<T>(nu, 2.0, 1.0, 2)}; // a1 = 2 along x1, a2 = 1 along x2
  T const a1{2}, a2{1}, v{1 / (2 * (1 - nu))}, s{(a1 + a2) * (a1 + a2)};
  EXPECT_NEAR(S(0, 0, 0, 0), v * ((a2 * a2 + 2 * a1 * a2) / s + (1 - 2 * nu) * a2 / (a1 + a2)), 1e-15); // Mura 11.21
  EXPECT_NEAR(S(1, 2, 1, 2), a1 / (2 * (a1 + a2)), 1e-15);
  EXPECT_NEAR(S(2, 0, 2, 0), a2 / (2 * (a1 + a2)), 1e-15);
  expect_minor_symmetric(S);
  EXPECT_NEAR(trace(S), (1 + nu) / (1 - nu), 1e-14);
}

/// Spheroid a1 = a2 = a, a3 = eta a along n: the closed form of uvwmaterial (eshelby_tensor_spheroidal_func).
tensor4 spheroid_reference(T nu_, T eta, vector const &n) {
  T const eta2{eta * eta};
  T const g{eta < 1 ? eta / std::pow(1 - eta2, 1.5) * (std::acos(eta) - eta * std::sqrt(1 - eta2))
                    : eta / std::pow(eta2 - 1, 1.5) * (eta * std::sqrt(eta2 - 1) - std::acosh(eta))};
  auto const I{tmech::eye<T, 3, 2>()};
  tensor2 const p{tmech::otimes(n, n)};
  tensor4 const II1{(tmech::otimesu(I, I) + tmech::otimesl(I, I)) * 0.5}, II2{tmech::otimes(I, I)}, II3{tmech::otimes(p, I)},
      II4{tmech::otimes(I, p)}, II5{(tmech::otimesu(p, I) + tmech::otimesl(p, I) + tmech::otimesu(I, p) + tmech::otimesl(I, p)) * 0.5},
      II6{tmech::otimes(p, p)};
  T const c{(eta2 - 1) * nu_ - eta2 + 1};
  T const c1{((8 * (eta2 - 1) * nu_ - 4 * eta2 + 7) * g - 2 * eta2) / (8 * c)};
  T const c2{-((8 * (eta2 - 1) * nu_ - 4 * eta2 + 1) * g + 2 * eta2) / (16 * c)};
  T const c3{((24 * (eta2 - 1) * nu_ - 12 * eta2 - 3) * g + 16 * (1 - eta2) * nu_ + 10 * eta2) / (16 * c)};
  T const c4{-((12 * eta2 + 3) * g - 10 * eta2) / (16 * c)};
  T const c5{-((12 * (eta2 - 1) * nu_ + 15) * g + 8 * (1 - eta2) * nu_ - 2 * eta2 - 8) / (8 * c)};
  T const c6{((60 * eta2 + 45) * g - 54 * eta2 - 16) / (16 * c)};
  return tensor4{c1 * II1 + c2 * II2 + c3 * II3 + c4 * II4 + c5 * II5 + c6 * II6};
}

TEST(Eshelby, EllipsoidLimits) {
  EXPECT_NEAR(diff(es::ellipsoid<T>(nu, 1.0, 1.0, 1.0), es::sphere<T>(nu)), 0, 1e-13);
  EXPECT_NEAR(diff(es::ellipsoid<T>(nu, 2.0, 1.0, 1e6), es::elliptic_cylinder<T>(nu, 2.0, 1.0, 2)), 0, 1e-5);
  EXPECT_NEAR(diff(es::ellipsoid<T>(nu, 2.0, 1.0, inf), es::elliptic_cylinder<T>(nu, 2.0, 1.0, 2)), 0, 1e-14);
  EXPECT_NEAR(diff(es::ellipsoid<T>(nu, inf, 1.0, 1.0), es::cylinder<T>(nu, 0)), 0, 1e-14);
}

TEST(Eshelby, SpheroidsAgreeWithTheClosedForm) {
  vector e3;
  e3(2) = 1;
  for (T const eta : {0.2, 0.5, 2.0, 10.0}) {
    tensor4 const S{es::ellipsoid<T>(nu, 1.0, 1.0, eta)};
    EXPECT_NEAR(diff(S, spheroid_reference(nu, eta, e3)), 0, 1e-12) << eta;
    EXPECT_NEAR(trace(S), (1 + nu) / (1 - nu), 1e-13) << eta;
  }
}

TEST(Eshelby, GeneralEllipsoid) {
  tensor4 const S{es::ellipsoid<T>(nu, 3.0, 2.0, 1.0)};
  expect_minor_symmetric(S);
  EXPECT_NEAR(trace(S), (1 + nu) / (1 - nu), 1e-13);
  // swapping two axes swaps the components
  tensor4 const swapped{es::ellipsoid<T>(nu, 2.0, 3.0, 1.0)};
  EXPECT_NEAR(S(0, 0, 0, 0), swapped(1, 1, 1, 1), 1e-14);
  EXPECT_NEAR(S(0, 0, 2, 2), swapped(1, 1, 2, 2), 1e-14);
  EXPECT_NEAR(S(0, 1, 0, 1), swapped(0, 1, 0, 1), 1e-14);
  // the long axis is the softest constraint: S_1111 < S_3333
  EXPECT_LT(S(0, 0, 0, 0), S(2, 2, 2, 2));
  // a general ellipsoid with two nearly equal axes is continuous into the spheroid
  EXPECT_NEAR(diff(es::ellipsoid<T>(nu, 1.0, 1.0 + 1e-7, 2.0), es::ellipsoid<T>(nu, 1.0, 1.0, 2.0)), 0, 1e-6);
}

TEST(Eshelby, RejectsInvalidInput) {
  EXPECT_THROW(es::sphere<T>(0.5), std::domain_error);
  EXPECT_THROW(es::ellipsoid<T>(nu, 0.0, 1.0, 1.0), std::domain_error);
  EXPECT_THROW(es::ellipsoid<T>(nu, inf, inf, 1.0), std::domain_error);
  EXPECT_THROW(es::cylinder<T>(nu, 3), std::out_of_range);
}

} // namespace
