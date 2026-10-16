#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include "numsim-materials/core/elliptic_integrals.h"

namespace {

namespace el = numsim::materials::elliptic;

// Reference values: Carlson, Numerical computation of real or complex elliptic integrals (1995), Table 1.
TEST(Carlson, RF) {
  EXPECT_NEAR(el::rf(1.0, 2.0, 0.0), 1.3110287771461, 1e-13);
  EXPECT_NEAR(el::rf(2.0, 3.0, 4.0), 0.58408284167715, 1e-13);
  EXPECT_NEAR(el::rf(0.5, 1.0, 0.0), 1.8540746773014, 1e-13);
  EXPECT_NEAR(el::rf(3.0, 3.0, 3.0), 1 / std::sqrt(3.0), 1e-15);
  EXPECT_NEAR(el::rf(1.0, 2.0, 0.0), el::rf(2.0, 0.0, 1.0), 1e-15); // symmetric
}

TEST(Carlson, RC) {
  EXPECT_NEAR(el::rc(0.0, 0.25), std::numbers::pi, 1e-13);
  EXPECT_NEAR(el::rc(2.25, 2.0), std::numbers::ln2, 1e-13);
  EXPECT_NEAR(el::rc(0.25, -2.0), std::numbers::ln2 / 3, 1e-13); // Cauchy principal value
}

TEST(Carlson, RD) {
  EXPECT_NEAR(el::rd(0.0, 2.0, 1.0), 1.7972103521034, 1e-13);
  EXPECT_NEAR(el::rd(2.0, 3.0, 4.0), 0.16510527294261, 1e-13);
  EXPECT_NEAR(el::rd(2.0, 3.0, 4.0), el::rd(3.0, 2.0, 4.0), 1e-15); // symmetric in the first two
  EXPECT_NEAR(el::rd(1.0, 1.0, 1.0), 1.0, 1e-15);
}

TEST(Carlson, RJ) {
  EXPECT_NEAR(el::rj(0.0, 1.0, 2.0, 3.0), 0.77688623778582, 1e-13);
  EXPECT_NEAR(el::rj(2.0, 3.0, 4.0, 5.0), 0.14297579667157, 1e-13);
  EXPECT_NEAR(el::rj(2.0, 3.0, 4.0, 4.0), el::rd(2.0, 3.0, 4.0), 1e-13); // R_J(x,y,z,z) = R_D(x,y,z)
}

TEST(Carlson, CompleteIntegrals) {
  EXPECT_NEAR(el::complete_k(0.0), std::numbers::pi / 2, 1e-15);
  EXPECT_NEAR(el::complete_e(0.0), std::numbers::pi / 2, 1e-15);
  EXPECT_NEAR(el::complete_k(0.5), 1.6857503548125961, 1e-14); // k = 0.5
  EXPECT_NEAR(el::complete_e(0.5), 1.4674622093394272, 1e-14);
  EXPECT_NEAR(el::complete_e(1.0), 1.0, 1e-14);
}

TEST(Carlson, RejectsInvalidArguments) {
  EXPECT_THROW(el::rf(-1.0, 1.0, 1.0), std::domain_error);
  EXPECT_THROW(el::rf(0.0, 0.0, 1.0), std::domain_error); // at most one zero
  EXPECT_THROW(el::rd(1.0, 1.0, 0.0), std::domain_error);
  EXPECT_THROW(el::rj(1.0, 1.0, 1.0, 0.0), std::domain_error);
}

} // namespace
