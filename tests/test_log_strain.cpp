#include <gtest/gtest.h>
#include <cmath>
#include <tmech/tmech.h>
#include "numsim-materials/core/log_strain.h"

namespace {

namespace nm = numsim::materials;
using T = double;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

tensor2 diag(T a, T b, T c) {
  tensor2 F;
  F(0, 0) = a;
  F(1, 1) = b;
  F(2, 2) = c;
  return F;
}

tensor4 fd(tensor2 const &F, T h) {
  tensor4 out;
  for (std::size_t k{0}; k < 3; ++k)
    for (std::size_t l{0}; l < 3; ++l) {
      tensor2 Fp{F}, Fm{F};
      Fp(k, l) += h;
      Fm(k, l) -= h;
      tensor2 const d{(nm::log_strain(Fp).strain - nm::log_strain(Fm).strain) / (2 * h)};
      for (std::size_t i{0}; i < 3; ++i)
        for (std::size_t j{0}; j < 3; ++j)
          out(i, j, k, l) = d(i, j);
    }
  return out;
}

TEST(LogStrain, MatchesClosedFormsAndTmech) {
  auto const [eps, deps]{nm::log_strain(diag(1.5, 0.8, 1.2))};
  EXPECT_NEAR(eps(0, 0), std::log(1.5), 1e-14);
  EXPECT_NEAR(eps(1, 1), std::log(0.8), 1e-14);
  EXPECT_NEAR(eps(0, 1), 0, 1e-14);
  tensor2 F{tmech::eye<T, 3, 2>()};
  F(0, 1) = 0.3;
  F(2, 0) = -0.2;
  F(1, 1) = 1.1;
  tensor2 const reference{tmech::log(tensor2{tmech::polar_decomposition(F).V()})}; // generic state: tmech is fine
  EXPECT_NEAR(tmech::norm(tensor2{nm::log_strain(F).strain - reference}), 0, 1e-12);
  EXPECT_NEAR(tmech::norm(tensor2{eps - tmech::trans(eps)}), 0, 1e-15);
}

TEST(LogStrain, DerivativeMatchesFiniteDifferencesAtRepeatedAndNearlyRepeatedStretches) {
  for (auto const &F : {diag(1.15, 1.15, 1.0), diag(1.24, 1.24, 1.0), diag(1.2, 1.2, 1.2), diag(1.204, 0.9458, 0.9458),
                        diag(1.15, 1.15 + 1e-7, 1.0), diag(1.15, 1.15 + 1e-9, 1.0), diag(1.0, 1.0, 1.0)}) {
    auto const [eps, deps]{nm::log_strain(F)};
    for (T const h : {1e-5, 1e-6, 1e-7, 1e-8}) {
      tensor4 const numeric{fd(F, h)};
      EXPECT_NEAR(tmech::norm(tensor4{deps - numeric}) / tmech::norm(deps), 0, 1e-6) << "h = " << h << " F11 " << F(0, 0) << " F22 " << F(1, 1);
    }
  }
}

TEST(LogStrain, DerivativeAtAGenericNonSymmetricF) {
  tensor2 F{tmech::eye<T, 3, 2>()};
  F(0, 0) = 1.3;
  F(0, 1) = 0.2;
  F(1, 1) = 0.9;
  F(2, 0) = -0.1;
  F(2, 2) = 1.1;
  auto const [eps, deps]{nm::log_strain(F)};
  EXPECT_NEAR(tmech::norm(tensor4{deps - fd(F, 1e-6)}) / tmech::norm(deps), 0, 1e-8);
  // symmetric in the first index pair
  EXPECT_NEAR(tmech::norm(tensor4{deps - tmech::basis_change<tmech::sequence<2, 1, 3, 4>>(deps)}), 0, 1e-14);
}

TEST(LogStrain, RejectsASingularF) {
  EXPECT_THROW((void)nm::log_strain(diag(1.0, 0.0, 1.0)), std::domain_error);
}

} // namespace
