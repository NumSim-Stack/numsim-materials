#include <gtest/gtest.h>
#include <stdexcept>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/isotropize.h"

namespace {

namespace nm = numsim::materials;
using T = double;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using vector = tmech::tensor<T, 3, 1>;

TEST(Isotropize, IsotropicTensorIsItsOwnProjection) {
  tensor4 const C{nm::isotropic_stiffness<T, 3>(100.0, 40.0)};
  auto const [K, G]{nm::isotropize(C)};
  EXPECT_NEAR(K, 100, 1e-12);
  EXPECT_NEAR(G, 40, 1e-12);
  EXPECT_NEAR(tmech::norm(tensor4{nm::isotropic_part(C) - C}), 0, 1e-12);
}

TEST(Isotropize, TwoDimensionalRoundTrip) {
  tmech::tensor<T, 2, 4> const C{nm::isotropic_stiffness<T, 2>(100.0, 40.0)};
  auto const [K, G]{nm::isotropize(C)};
  EXPECT_NEAR(K, 100, 1e-12);
  EXPECT_NEAR(G, 40, 1e-12);
}

TEST(Isotropize, DeviatoricRankOneTermShiftsOnlyG) {
  tensor2 N;
  N(0, 0) = 1;
  N(1, 1) = N(2, 2) = -0.5; // deviatoric, N:N = 3/2
  tensor4 const C{nm::isotropic_stiffness<T, 3>(100.0, 40.0) + 20.0 * tmech::otimes(N, N)};
  auto const [K, G]{nm::isotropize(C)};
  EXPECT_NEAR(K, 100, 1e-12);
  EXPECT_NEAR(G, 40 + 20 * 1.5 / 10, 1e-12); // Pdev :: N(x)N = N:N, divided by Pdev::Pdev = 5, over 2
}

TEST(Isotropize, RotationInvariant) {
  vector e1;
  e1(0) = 1;
  tensor4 const C{nm::isotropic_stiffness<T, 3>(100.0, 40.0) + 300.0 * tmech::otimes(tmech::otimes(e1, e1), tmech::otimes(e1, e1))};
  vector axis;
  axis(2) = 1;
  auto const [K, G]{nm::isotropize(C)};
  auto const [Kr, Gr]{nm::isotropize(nm::rotate(C, nm::rotation(axis, 0.6)))};
  EXPECT_NEAR(K, Kr, 1e-12);
  EXPECT_NEAR(G, Gr, 1e-12);
  EXPECT_GT(K, 100);
  EXPECT_GT(G, 40);
}

TEST(Isotropize, RequireIsotropicNamesTheCaller) {
  tensor4 const C{nm::isotropic_stiffness<T, 3>(100.0, 40.0)};
  auto const [K, G]{nm::require_isotropic(C, "who")};
  EXPECT_NEAR(K, 100, 1e-12);
  EXPECT_NEAR(G, 40, 1e-12);
  vector e1;
  e1(0) = 1;
  tensor4 const aniso{C + 30.0 * tmech::otimes(tmech::otimes(e1, e1), tmech::otimes(e1, e1))};
  try {
    (void)nm::require_isotropic(aniso, "who");
    FAIL() << "an anisotropic tangent must be rejected";
  } catch (std::invalid_argument const &e) {
    EXPECT_NE(std::string{e.what()}.find("who"), std::string::npos);
  }
}

} // namespace
