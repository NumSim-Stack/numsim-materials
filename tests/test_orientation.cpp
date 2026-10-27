#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include <boost/math/special_functions/ellint_rd.hpp>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/lebedev.h"
#include "numsim-materials/core/orientation.h"

namespace {

namespace nm = numsim::materials;
namespace ori = numsim::materials::orientation;
using T = double;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using vector = tmech::tensor<T, 3, 1>;

tensor2 diagonal(T a1, T a2, T a3) {
  tensor2 A;
  A(0, 0) = a1;
  A(1, 1) = a2;
  A(2, 2) = a3;
  return A;
}

tensor2 second_moment(tensor2 const &A, std::size_t precision) {
  return ori::average(A, precision, [](vector const &n) { return tensor2{tmech::otimes(n, n)}; });
}

TEST(Orientation, IsotropicTensorGivesTheUniformDistribution) {
  auto const d{ori::fit_acg(diagonal(1.0 / 3, 1.0 / 3, 1.0 / 3))};
  for (auto const b : d.b)
    EXPECT_NEAR(b, 1, 1e-10);
  EXPECT_FALSE(d.aligned);
  EXPECT_NEAR(tmech::norm(tensor2{second_moment(diagonal(1.0 / 3, 1.0 / 3, 1.0 / 3), 5) - tmech::eye<T, 3, 2>() / 3}), 0, 1e-12);
}

TEST(Orientation, FittedParametersReproduceTheEigenvaluesExactly) {
  auto const d{ori::fit_acg(diagonal(0.6, 0.3, 0.1))};
  EXPECT_NEAR(d.b[0] * d.b[1] * d.b[2], 1, 1e-10);
  EXPECT_NEAR(boost::math::ellint_rd(d.b[1], d.b[2], d.b[0]) / 3, 0.6, 1e-10);
  EXPECT_NEAR(boost::math::ellint_rd(d.b[2], d.b[0], d.b[1]) / 3, 0.3, 1e-10);
  EXPECT_NEAR(boost::math::ellint_rd(d.b[0], d.b[1], d.b[2]) / 3, 0.1, 1e-10);
  EXPECT_LT(d.b[0], d.b[1]); // concentration along x1: the smallest b
}

TEST(Orientation, QuadratureRecoversTheSecondMoment) {
  tensor2 const A{diagonal(0.6, 0.3, 0.1)};
  T const coarse{tmech::norm(tensor2{second_moment(A, 7) - A})}, fine{tmech::norm(tensor2{second_moment(A, 23) - A})};
  EXPECT_LT(fine, 5e-3);
  EXPECT_LT(fine, coarse);
  T sum{0};
  for (auto const &[n, w] : ori::nodes(ori::fit_acg(A), 15))
    sum += w;
  EXPECT_NEAR(sum, 1, 1e-12);
}

TEST(Orientation, RotationCovariant) {
  vector axis;
  axis(0) = axis(2) = 1 / std::sqrt(2.0);
  auto const R{nm::rotation(axis, 0.8)};
  tensor2 const A{diagonal(0.7, 0.2, 0.1)};
  tensor2 const Ar{nm::rotate(A, R)};
  EXPECT_NEAR(tmech::norm(tensor2{second_moment(Ar, 23) - nm::rotate(second_moment(A, 23), R)}), 0, 1e-12);
  // a transversely isotropic function of n averages to the rotated result
  auto const f = [](vector const &n) { return tensor4{tmech::otimes(tmech::otimes(n, n), tmech::otimes(n, n))}; };
  EXPECT_NEAR(tmech::norm(tensor4{ori::average(Ar, 15, f) - nm::rotate(ori::average(A, 15, f), R)}), 0, 1e-12);
}

TEST(Orientation, AlignedTensorIsASingleDirection) {
  auto const d{ori::fit_acg(diagonal(1.0, 0.0, 0.0))};
  EXPECT_TRUE(d.aligned);
  auto const ns{ori::nodes(d, 23)};
  ASSERT_EQ(ns.size(), 1u);
  EXPECT_NEAR(std::abs(ns[0].first(0)), 1, 1e-14);
  EXPECT_DOUBLE_EQ(ns[0].second, 1);
  EXPECT_NEAR(tmech::norm(tensor2{second_moment(diagonal(1.0, 0.0, 0.0), 7) - diagonal(1.0, 0.0, 0.0)}), 0, 1e-14);
}

TEST(Orientation, PlanarRandomIsHandledByClamping) {
  tensor2 const A{diagonal(0.5, 0.5, 0.0)};
  tensor2 const m{second_moment(A, 23)};
  EXPECT_NEAR(m(0, 0), 0.5, 2e-2);
  EXPECT_LT(m(2, 2), 2e-2);
}

TEST(Orientation, RejectsInvalidTensors) {
  EXPECT_THROW(ori::fit_acg(diagonal(0.5, 0.3, 0.3)), std::invalid_argument); // trace 1.1
  tensor2 skew{diagonal(0.4, 0.3, 0.3)};
  skew(0, 1) = 0.1;
  EXPECT_THROW(ori::fit_acg(skew), std::invalid_argument); // not symmetric
  EXPECT_THROW(ori::fit_acg(diagonal(1.2, -0.1, -0.1)), std::invalid_argument);
}

} // namespace
