#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <numbers>
#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/core/lebedev.h"
#include "numsim-materials/core/tensor_identities.h"

namespace {

namespace nm = numsim::materials;
using T = double;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;
using vector = tmech::tensor<T, 3, 1>;
constexpr T inf{std::numeric_limits<T>::infinity()};

// --- Lebedev ---------------------------------------------------------------------------------

TEST(Lebedev, WeightsSumToOneAndNodesAreUnit) {
  for (auto const p : nm::lebedev::precisions()) {
    T sum{0};
    for (auto const &n : nm::lebedev::rule(p)) {
      sum += n.w;
      EXPECT_NEAR(n.x * n.x + n.y * n.y + n.z * n.z, 1, 1e-14);
    }
    EXPECT_NEAR(sum, 1, 1e-12) << p; // the tables carry 15 digits
  }
  EXPECT_EQ(nm::lebedev::rule(3).size(), 6u);
  EXPECT_EQ(nm::lebedev::rule(23).size(), 194u);
  EXPECT_THROW((void)nm::lebedev::rule(4), std::invalid_argument);
}

TEST(Lebedev, SecondAndFourthMoments) {
  auto const I{tmech::eye<T, 3, 2>()};
  for (auto const p : nm::lebedev::precisions()) {
    tensor2 const m2{nm::lebedev::average(p, [](vector const &n) { return tensor2{tmech::otimes(n, n)}; })};
    EXPECT_NEAR(tmech::norm(tensor2{m2 - I / 3}), 0, 1e-12) << p;
    if (p < 5)
      continue;
    tensor4 const m4{nm::lebedev::average(p, [](vector const &n) { return tensor4{tmech::otimes(tmech::otimes(n, n), tmech::otimes(n, n))}; })};
    tensor4 const exact{(tmech::otimes(I, I) + 2 * nm::sym_identity<T, 3>()) / 15};
    EXPECT_NEAR(tmech::norm(tensor4{m4 - exact}), 0, 1e-13) << p;
  }
}

TEST(Lebedev, DegreeOfExactness) {
  // n1^4 has mean 1/5; the 6-point rule (degree 3) misses it, degree 5 gets it
  auto const f = [](vector const &n) { return std::pow(n(0), 4); };
  EXPECT_NEAR(nm::lebedev::average(3, f), 1.0 / 3, 1e-14);
  EXPECT_NEAR(nm::lebedev::average(5, f), 1.0 / 5, 1e-14);
  EXPECT_NEAR(nm::lebedev::integrate(5, [](vector const &) { return 1.0; }), 4 * std::numbers::pi, 1e-13);
}

// --- depolarisation ---------------------------------------------------------------------------

TEST(Depolarisation, SphereCylinderAndTrace) {
  auto const I{tmech::eye<T, 3, 2>()};
  EXPECT_NEAR(tmech::norm(tensor2{nm::eshelby::depolarisation<T>(1.0, 1.0, 1.0) - I / 3}), 0, 1e-14);
  tensor2 const needle{nm::eshelby::depolarisation<T>(1.0, 1.0, inf)};
  EXPECT_NEAR(needle(0, 0), 0.5, 1e-14);
  EXPECT_NEAR(needle(2, 2), 0, 1e-14);
  EXPECT_NEAR(tmech::norm(tensor2{nm::eshelby::depolarisation<T>(1.0, 1.0, 1e6) - needle}), 0, 1e-5);
  tensor2 const elliptic{nm::eshelby::depolarisation<T>(2.0, 1.0, inf)};
  EXPECT_NEAR(elliptic(0, 0), 1.0 / 3, 1e-14); // b / (a + b)
  EXPECT_NEAR(elliptic(1, 1), 2.0 / 3, 1e-14);
  for (auto const &N : {nm::eshelby::depolarisation<T>(3.0, 2.0, 1.0), nm::eshelby::depolarisation<T>(1.0, 1.0, 0.2)})
    EXPECT_NEAR(tmech::trace(N), 1, 1e-13);
  vector d;
  d(0) = d(1) = 1 / std::sqrt(2.0);
  EXPECT_NEAR(tmech::norm(tensor2{nm::eshelby::depolarisation_cylinder(d) - (I - tmech::otimes(d, d)) / 2}), 0, 1e-14);
}

TEST(Depolarisation, SpheroidsMatchTheClosedForms) {
  // prolate: a = b = 1 < c, e = sqrt(1 - 1/c^2): N_c = (1 - e^2) / e^3 (atanh e - e)
  T const c{3}, e{std::sqrt(1 - 1 / (c * c))};
  EXPECT_NEAR(nm::eshelby::depolarisation<T>(1.0, 1.0, c)(2, 2), (1 - e * e) / (e * e * e) * (std::atanh(e) - e), 1e-13);
  // oblate: a = b = 1 > c, e = sqrt(1/c^2 - 1): N_c = (1 + e^2) / e^3 (e - atan e)
  T const c2{0.4}, e2{std::sqrt(1 / (c2 * c2) - 1)};
  EXPECT_NEAR(nm::eshelby::depolarisation<T>(1.0, 1.0, c2)(2, 2), (1 + e2 * e2) / (e2 * e2 * e2) * (e2 - std::atan(e2)), 1e-13);
}

} // namespace
