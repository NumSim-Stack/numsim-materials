#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/tensor_identities.h"

namespace {

namespace nm = numsim::materials;
using T = double;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using vector = tmech::tensor<T, 3, 1>;

TEST(ElasticConstants, RoundTrips) {
  constexpr T E{200000}, nu{0.3};
  constexpr auto K{nm::bulk_modulus(E, nu)};
  constexpr auto G{nm::shear_modulus(E, nu)};
  static_assert(K > 0 && G > 0);
  EXPECT_NEAR(K, E / (3 * (1 - 2 * nu)), 1e-9);
  EXPECT_NEAR(G, E / (2 * (1 + nu)), 1e-9);
  EXPECT_NEAR(nm::youngs_modulus(K, G), E, 1e-9);
  EXPECT_NEAR(nm::poisson_ratio(K, G), nu, 1e-15);
  EXPECT_NEAR(nm::lame_lambda(K, G), K - 2 * G / 3, 1e-9);
  EXPECT_NEAR(nm::bulk_modulus_from_lame(nm::lame_lambda(K, G), G), K, 1e-9);
}

TEST(ElasticConstants, IsotropicStiffnessFromKG) {
  tensor4 const C{nm::isotropic_stiffness<T, 3>(166666.67, 76923.08)};
  EXPECT_NEAR(C(0, 0, 0, 0), 166666.67 + 4 * 76923.08 / 3, 1e-6);
  EXPECT_NEAR(C(0, 0, 1, 1), 166666.67 - 2 * 76923.08 / 3, 1e-6);
  EXPECT_NEAR(C(0, 1, 0, 1), 76923.08, 1e-9);
}

tensor2 rotation_z(T angle) {
  tensor2 R;
  R(0, 0) = std::cos(angle);
  R(0, 1) = -std::sin(angle);
  R(1, 0) = std::sin(angle);
  R(1, 1) = std::cos(angle);
  R(2, 2) = 1;
  return R;
}

TEST(Rotation, Rank2IsRART) {
  tensor2 A;
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 2) = 3;
  auto const R{rotation_z(0.7)};
  tensor2 const expected{R * A * tmech::trans(R)};
  EXPECT_NEAR(tmech::norm(tensor2{nm::rotate(A, R) - expected}), 0, 1e-14);
}

TEST(Rotation, IsotropicRank4IsInvariantAndTransverselyIsotropicMoves) {
  auto const R{rotation_z(std::numbers::pi / 2)};
  tensor4 const C{nm::isotropic_stiffness<T, 3>(100.0, 40.0)};
  EXPECT_NEAR(tmech::norm(tensor4{nm::rotate(C, R) - C}), 0, 1e-12);
  // a fibre stiffness along e1 becomes one along e2
  vector e1, e2;
  e1(0) = 1;
  e2(1) = 1;
  tensor4 const along_1{tmech::otimes(tmech::otimes(e1, e1), tmech::otimes(e1, e1))};
  tensor4 const along_2{tmech::otimes(tmech::otimes(e2, e2), tmech::otimes(e2, e2))};
  EXPECT_NEAR(tmech::norm(tensor4{nm::rotate(along_1, R) - along_2}), 0, 1e-14);
}

TEST(Rotation, FromAxisAndAngle) {
  vector axis;
  axis(2) = 1;
  auto const R{nm::rotation(axis, 0.7)};
  EXPECT_NEAR(tmech::norm(tensor2{R - rotation_z(0.7)}), 0, 1e-14);
  EXPECT_NEAR(tmech::det(R), 1, 1e-14);
  // rotation taking e1 onto a direction
  vector d;
  d(0) = d(1) = d(2) = 1 / std::sqrt(3.0);
  tensor2 const Rd{nm::rotation_to(d)};
  vector e1;
  e1(0) = 1;
  EXPECT_NEAR(tmech::norm(vector{Rd * e1 - d}), 0, 1e-14);
  EXPECT_NEAR(tmech::norm(tensor2{tmech::trans(Rd) * Rd - tmech::eye<T, 3, 2>()}), 0, 1e-14);
  EXPECT_NEAR(tmech::norm(tensor2{nm::rotation_to(e1) - tmech::eye<T, 3, 2>()}), 0, 1e-14);
}

} // namespace
