#include <gtest/gtest.h>
#include <cmath>
#include <tmech/tmech.h>
#include "numsim-materials/umat/tensor_conversion.h"

namespace {

namespace u = numsim::materials::umat;
using T = double;
using tensor2 = tmech::tensor<T, 3, 2>;
using tensor4 = tmech::tensor<T, 3, 4>;

constexpr T tol = 1e-14;

/// A symmetric rank-2 tensor with all six components distinct, so a slot
/// permutation cannot pass by coincidence.
tensor2 make_sym() {
  tensor2 t;
  t.fill(0.0);
  t(0, 0) = 1.0;  t(1, 1) = 2.0;  t(2, 2) = 3.0;
  t(0, 1) = t(1, 0) = 4.0;
  t(0, 2) = t(2, 0) = 5.0;
  t(1, 2) = t(2, 1) = 6.0;
  return t;
}

/// C = A (x) B with A != B. Minor-symmetric (both factors are symmetric) but
/// deliberately major-ASYMMETRIC: C_ijkl = A_ij B_kl != C_klij. This is the
/// shape a non-associative tangent has, and the only kind that can detect a
/// row/column-major transpose slip.
tensor4 make_major_asymmetric() {
  tensor2 A, B;
  A.fill(0.0);
  B.fill(0.0);
  A(0, 0) = 1.0;  A(1, 1) = 2.0;  A(2, 2) = 3.0;  A(0, 1) = A(1, 0) = 4.0;
  B(0, 0) = 5.0;  B(1, 1) = 6.0;  B(2, 2) = 7.0;  B(1, 2) = B(2, 1) = 8.0;
  tensor4 C;
  C = tmech::otimes(A, B);
  return C;
}

/// Isotropic elastic tangent, same construction as linear_elasticity.
tensor4 make_isotropic(T K, T G) {
  const auto I = tmech::eye<T, 3, 2>();
  const auto IIsym = (tmech::otimesu(I, I) + tmech::otimesl(I, I)) * 0.5;
  const auto IIvol = tmech::otimes(I, I) / 3.0;
  const auto IIdev = IIsym - IIvol;
  tensor4 C;
  C = 3.0 * K * IIvol + 2.0 * G * IIdev;
  return C;
}

// ---------------------------------------------------------------------------
// Slot ordering
// ---------------------------------------------------------------------------

TEST(UmatConversionOrdering, CanonicalSlotsAreAbaqusStandard) {
  // Abaqus/Standard 3D order: {11, 22, 33, 12, 13, 23}
  const std::pair<int, int> expected[6] = {
      {0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};

  for (std::size_t s = 0; s < 6; ++s) {
    T buf[6] = {0, 0, 0, 0, 0, 0};
    buf[s] = 1.0;
    const auto sig = u::stress_from_buffer<T>(buf);
    const auto [i, j] = expected[s];
    EXPECT_NEAR(sig(i, j), 1.0, tol) << "slot " << s;
    EXPECT_NEAR(sig(j, i), 1.0, tol) << "slot " << s << " (minor symmetry)";

    // Every other component must be zero.
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        if (!((a == i && b == j) || (a == j && b == i)))
          EXPECT_NEAR(sig(a, b), 0.0, tol)
              << "slot " << s << " leaked into (" << a << "," << b << ")";
  }
}

// ---------------------------------------------------------------------------
// Shear conventions — the asymmetry between strain, stress and tangent
// ---------------------------------------------------------------------------

TEST(UmatConversionShear, StrainBufferCarriesEngineeringShear) {
  // STRAN slot 3 is gamma_12 = 2*eps_12, so a unit entry is eps_12 = 0.5.
  T buf[6] = {0, 0, 0, 1.0, 0, 0};
  const auto eps = u::strain_from_buffer<T>(buf);
  EXPECT_NEAR(eps(0, 1), 0.5, tol);
  EXPECT_NEAR(eps(1, 0), 0.5, tol);
}

TEST(UmatConversionShear, StressBufferHasNoShearFactor) {
  T buf[6] = {0, 0, 0, 1.0, 0, 0};
  const auto sig = u::stress_from_buffer<T>(buf);
  EXPECT_NEAR(sig(0, 1), 1.0, tol);
}

TEST(UmatConversionShear, StrainRoundTripsThroughBuffer) {
  const auto eps = make_sym();
  T buf[6];
  u::strain_to_buffer<T>(eps, buf);

  // Diagonals unscaled, off-diagonals doubled to engineering shear.
  EXPECT_NEAR(buf[0], 1.0, tol);
  EXPECT_NEAR(buf[1], 2.0, tol);
  EXPECT_NEAR(buf[2], 3.0, tol);
  EXPECT_NEAR(buf[3], 8.0, tol);   // 2 * eps_12
  EXPECT_NEAR(buf[4], 10.0, tol);  // 2 * eps_13
  EXPECT_NEAR(buf[5], 12.0, tol);  // 2 * eps_23

  const auto back = u::strain_from_buffer<T>(buf);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) EXPECT_NEAR(back(i, j), eps(i, j), tol);
}

TEST(UmatConversionShear, StressRoundTripsThroughBuffer) {
  const auto sig = make_sym();
  T buf[6];
  u::stress_to_buffer<T>(sig, buf);
  for (std::size_t s = 0; s < 6; ++s)
    EXPECT_NEAR(buf[s], static_cast<T>(s + 1), tol) << "slot " << s;

  const auto back = u::stress_from_buffer<T>(buf);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) EXPECT_NEAR(back(i, j), sig(i, j), tol);
}

/// The tangent must NOT be shear-scaled: DDSDDE maps engineering strain to
/// stress, and the factor 2 is already absorbed by the minor-symmetry sum over
/// kl = 12 and 21. Scaling it again would double-count.
///
/// Checked physically rather than by inspecting slots: sigma = C : eps computed
/// in tensor space must equal DDSDDE * (engineering strain vector).
TEST(UmatConversionShear, TangentTimesEngineeringStrainEqualsTensorContraction) {
  const auto C = make_isotropic(166.67, 76.92);
  const auto eps = make_sym();

  tensor2 sig;
  sig = tmech::dcontract(C, eps);

  T eps_buf[6], sig_buf[6], c_buf[36];
  u::strain_to_buffer<T>(eps, eps_buf);  // engineering
  u::stress_to_buffer<T>(sig, sig_buf);
  u::tangent_to_buffer<T>(C, c_buf);

  for (std::size_t i = 0; i < 6; ++i) {
    T acc = 0.0;
    for (std::size_t j = 0; j < 6; ++j) acc += c_buf[i * 6 + j] * eps_buf[j];
    EXPECT_NEAR(acc, sig_buf[i], 1e-11) << "row " << i;
  }
}

// ---------------------------------------------------------------------------
// Rank-4 packing and storage order
// ---------------------------------------------------------------------------

TEST(UmatConversionTangent, PacksRowMajor) {
  // For C = A (x) B the packed matrix is exactly the outer product of the two
  // packed vectors, which pins down the row-major convention unambiguously.
  tensor2 A, B;
  A.fill(0.0);
  B.fill(0.0);
  A(0, 0) = 1.0;  A(1, 1) = 2.0;  A(2, 2) = 3.0;  A(0, 1) = A(1, 0) = 4.0;
  B(0, 0) = 5.0;  B(1, 1) = 6.0;  B(2, 2) = 7.0;  B(1, 2) = B(2, 1) = 8.0;
  tensor4 C;
  C = tmech::otimes(A, B);

  T a6[6], b6[6], c36[36];
  u::stress_to_buffer<T>(A, a6);
  u::stress_to_buffer<T>(B, b6);
  u::tangent_to_buffer<T>(C, c36);

  for (std::size_t i = 0; i < 6; ++i)
    for (std::size_t j = 0; j < 6; ++j)
      EXPECT_NEAR(c36[i * 6 + j], a6[i] * b6[j], 1e-12) << i << "," << j;
}

TEST(UmatConversionTangent, MajorAsymmetricTangentIsGenuinelyAsymmetric) {
  // Guards the guard: if this fixture ever became symmetric, every transpose
  // test below would silently stop testing anything.
  const auto C = make_major_asymmetric();
  T c36[36];
  u::tangent_to_buffer<T>(C, c36);
  EXPECT_GT(std::abs(c36[0 * 6 + 3] - c36[3 * 6 + 0]), 1.0);
}

TEST(UmatConversionTangent, MajorAsymmetricTangentRoundTrips) {
  const auto C = make_major_asymmetric();
  T c36[36];
  u::tangent_to_buffer<T>(C, c36);
  const auto back = u::tangent_from_buffer<T>(c36);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          EXPECT_NEAR(back(i, j, k, l), C(i, j, k, l), 1e-12)
              << i << j << k << l;
}

TEST(UmatConversionTangent, NarrowMatrixWritesColumnMajor) {
  // With an asymmetric tangent, reading the host array as Fortran column-major
  // must reproduce C[a][b]. If narrow_matrix wrote row-major instead, this is
  // the test that fails.
  const auto C = make_major_asymmetric();
  T c36[36], host[36];
  u::tangent_to_buffer<T>(C, c36);
  u::narrow_matrix<T>(c36, u::element_case::solid3d, host);

  for (std::size_t a = 0; a < 6; ++a)
    for (std::size_t b = 0; b < 6; ++b)
      EXPECT_NEAR(host[a + b * 6], c36[a * 6 + b], 1e-12) << a << "," << b;
}

// ---------------------------------------------------------------------------
// Slot maps per element case
// ---------------------------------------------------------------------------

TEST(UmatConversionSlots, NtensPerCase) {
  EXPECT_EQ(u::ntens(u::element_case::solid3d), 6u);
  EXPECT_EQ(u::ntens(u::element_case::plane_strain), 4u);
  EXPECT_EQ(u::ntens(u::element_case::axisymmetric), 4u);
  EXPECT_EQ(u::ntens(u::element_case::plane_stress), 3u);
}

TEST(UmatConversionSlots, PlaneStrainWidensToPrefix) {
  const T host[4] = {11.0, 22.0, 33.0, 12.0};
  T out[6];
  u::widen_vector<T>(host, u::element_case::plane_strain, out);
  const T expect[6] = {11.0, 22.0, 33.0, 12.0, 0.0, 0.0};
  for (std::size_t i = 0; i < 6; ++i) EXPECT_NEAR(out[i], expect[i], tol) << i;
}

TEST(UmatConversionSlots, AxisymmetricKeepsSuppliedHoopComponent) {
  // Unlike plane strain, the 33 slot is genuinely nonzero here and must survive.
  const T host[4] = {11.0, 22.0, 99.0, 12.0};
  T out[6];
  u::widen_vector<T>(host, u::element_case::axisymmetric, out);
  EXPECT_NEAR(out[2], 99.0, tol);
}

/// Plane stress is NOT a prefix map: host slot 2 is the 12 component, so it
/// must land in canonical slot 3, leaving slot 2 for the iterated eps_33.
TEST(UmatConversionSlots, PlaneStressMapIsNotAPrefix) {
  const T host[3] = {11.0, 22.0, 12.0};
  T out[6];
  u::widen_vector<T>(host, u::element_case::plane_stress, out, 99.0);
  const T expect[6] = {11.0, 22.0, 99.0, 12.0, 0.0, 0.0};
  for (std::size_t i = 0; i < 6; ++i) EXPECT_NEAR(out[i], expect[i], tol) << i;
}

TEST(UmatConversionSlots, WidenNarrowRoundTripsForEveryCase) {
  const u::element_case cases[] = {
      u::element_case::solid3d, u::element_case::plane_strain,
      u::element_case::axisymmetric, u::element_case::plane_stress};

  for (auto c : cases) {
    const auto n = u::ntens(c);
    T host[6], back[6], wide[6];
    for (std::size_t k = 0; k < n; ++k) host[k] = static_cast<T>(10 * (k + 1));
    u::widen_vector<T>(host, c, wide);
    u::narrow_vector<T>(wide, c, back);
    for (std::size_t k = 0; k < n; ++k)
      EXPECT_NEAR(back[k], host[k], tol)
          << "case " << static_cast<int>(c) << " slot " << k;
  }
}

TEST(UmatConversionSlots, PlaneStrainTangentIsLeadingSubBlock) {
  const auto C = make_major_asymmetric();
  T c36[36], host[16];
  u::tangent_to_buffer<T>(C, c36);
  u::narrow_matrix<T>(c36, u::element_case::plane_strain, host);

  for (std::size_t a = 0; a < 4; ++a)
    for (std::size_t b = 0; b < 4; ++b)
      EXPECT_NEAR(host[a + b * 4], c36[a * 6 + b], 1e-12) << a << "," << b;
}

// ---------------------------------------------------------------------------
// Plane-stress condensation
// ---------------------------------------------------------------------------

/// Condensing the isotropic 3D tangent must reproduce the textbook plane-stress
/// matrix, in engineering-strain convention:
///
///     [ E/(1-v^2)     v E/(1-v^2)   0 ]
///     [ v E/(1-v^2)   E/(1-v^2)     0 ]
///     [ 0             0             G ]
TEST(UmatConversionPlaneStress, CondensationMatchesAnalyticMatrix) {
  const T K = 166.67, G = 76.92;
  const T E = 9.0 * K * G / (3.0 * K + G);
  const T nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G));
  const T f = E / (1.0 - nu * nu);

  const auto C = make_isotropic(K, G);
  T c36[36], host[9];
  u::tangent_to_buffer<T>(C, c36);
  u::narrow_matrix<T>(c36, u::element_case::plane_stress, host);

  EXPECT_NEAR(host[0 + 0 * 3], f, 1e-9);
  EXPECT_NEAR(host[1 + 1 * 3], f, 1e-9);
  EXPECT_NEAR(host[0 + 1 * 3], nu * f, 1e-9);
  EXPECT_NEAR(host[1 + 0 * 3], nu * f, 1e-9);
  EXPECT_NEAR(host[2 + 2 * 3], G, 1e-9);
  EXPECT_NEAR(host[0 + 2 * 3], 0.0, 1e-9);
  EXPECT_NEAR(host[2 + 0 * 3], 0.0, 1e-9);
  EXPECT_NEAR(host[1 + 2 * 3], 0.0, 1e-9);
  EXPECT_NEAR(host[2 + 1 * 3], 0.0, 1e-9);
}

/// The condensed tangent must annihilate sigma_33: applying it to an in-plane
/// strain and recovering eps_33 from the condensation relation must give a
/// state with zero out-of-plane stress.
TEST(UmatConversionPlaneStress, CondensedResponseHasZeroSigma33) {
  const T K = 166.67, G = 76.92;
  const auto C = make_isotropic(K, G);
  T c36[36];
  u::tangent_to_buffer<T>(C, c36);

  // In-plane engineering strain.
  const T in_plane[3] = {0.003, -0.001, 0.002};

  // eps_33 that zeroes sigma_33: -(C_3a * e_a) / C_33
  const auto map = u::slot_map(u::element_case::plane_stress);
  T num = 0.0;
  for (std::size_t a = 0; a < 3; ++a)
    num += c36[u::canonical_33 * 6 + map[a]] * in_plane[a];
  const T eps33 = -num / c36[u::canonical_33 * 6 + u::canonical_33];

  T wide[6];
  u::widen_vector<T>(in_plane, u::element_case::plane_stress, wide, eps33);
  const auto eps = u::strain_from_buffer<T>(wide);
  tensor2 sig;
  sig = tmech::dcontract(C, eps);
  EXPECT_NEAR(sig(2, 2), 0.0, 1e-10);

  // And the condensed matrix must reproduce the in-plane stresses.
  T host[9], sig_buf[6], sig_host[3];
  u::narrow_matrix<T>(c36, u::element_case::plane_stress, host);
  u::stress_to_buffer<T>(sig, sig_buf);
  u::narrow_vector<T>(sig_buf, u::element_case::plane_stress, sig_host);

  for (std::size_t a = 0; a < 3; ++a) {
    T acc = 0.0;
    for (std::size_t b = 0; b < 3; ++b) acc += host[a + b * 3] * in_plane[b];
    EXPECT_NEAR(acc, sig_host[a], 1e-10) << "component " << a;
  }
}

TEST(UmatConversionPlaneStress, ThrowsWhenOutOfPlaneStiffnessIsZero) {
  T c36[36] = {0};
  for (std::size_t i = 0; i < 6; ++i) c36[i * 6 + i] = 1.0;
  c36[u::canonical_33 * 6 + u::canonical_33] = 0.0;  // no 33 stiffness
  T host[9];
  EXPECT_THROW(u::narrow_matrix<T>(c36, u::element_case::plane_stress, host),
               std::runtime_error);
}

// ---------------------------------------------------------------------------
// Co-rotational helpers
// ---------------------------------------------------------------------------

/// 90 degrees about the z axis, in Abaqus's column-major DROT(3,3) layout.
void rot_z90_buffer(T out[9]) {
  const T R[3][3] = {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) out[i + 3 * j] = R[i][j];
}

TEST(UmatConversionRotation, ReadsDrotColumnMajor) {
  T buf[9];
  rot_z90_buffer(buf);
  const auto R = u::rotation_from_buffer<T>(buf);
  EXPECT_NEAR(R(0, 1), -1.0, tol);
  EXPECT_NEAR(R(1, 0), 1.0, tol);
  EXPECT_NEAR(R(2, 2), 1.0, tol);
  EXPECT_NEAR(R(0, 0), 0.0, tol);
}

TEST(UmatConversionRotation, DetectsIdentity) {
  T ident[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  EXPECT_TRUE(u::is_identity_rotation(u::rotation_from_buffer<T>(ident)));
  T buf[9];
  rot_z90_buffer(buf);
  EXPECT_FALSE(u::is_identity_rotation(u::rotation_from_buffer<T>(buf)));
}

TEST(UmatConversionRotation, RotatesRankTwoByNinetyDegrees) {
  T buf[9];
  rot_z90_buffer(buf);
  const auto R = u::rotation_from_buffer<T>(buf);

  tensor2 a;
  a.fill(0.0);
  a(0, 0) = 1.0;  a(1, 1) = 2.0;  a(2, 2) = 3.0;

  const auto b = u::rotate(a, R);
  // A z-rotation of 90 degrees swaps the 11 and 22 components.
  EXPECT_NEAR(b(0, 0), 2.0, 1e-12);
  EXPECT_NEAR(b(1, 1), 1.0, 1e-12);
  EXPECT_NEAR(b(2, 2), 3.0, 1e-12);
}

/// Pins the SENSE of the rotation: R a R^T, not R^T a R.
///
/// The 90-degree diagonal fixture above cannot tell the two apart — both swap
/// the 11 and 22 entries — and neither can an invariant check, which any
/// orthogonal map satisfies. This uses a non-degenerate angle and a tensor with
/// a nonzero shear component, and computes the expectation from the closed-form
/// rotation algebra rather than from u::rotate, so the test cannot agree with a
/// self-consistently wrong implementation.
///
/// Getting this backwards would counter-rotate every tensor state variable
/// under NLGEOM=YES, with the error growing as rotation accumulates.
TEST(UmatConversionRotation, PinsTheSenseOfTheRotation) {
  constexpr T theta = 0.5235987755982988;  // 30 degrees
  const T c = std::cos(theta), sn = std::sin(theta);

  // R = [[c,-s,0],[s,c,0],[0,0,1]], column-major for DROT.
  T buf[9];
  for (auto& v : buf) v = 0.0;
  buf[0 + 3 * 0] = c;   buf[0 + 3 * 1] = -sn;
  buf[1 + 3 * 0] = sn;  buf[1 + 3 * 1] = c;
  buf[2 + 3 * 2] = 1.0;
  const auto R = u::rotation_from_buffer<T>(buf);

  tensor2 a;
  a.fill(0.0);
  a(0, 0) = 1.0;  a(1, 1) = 2.0;  a(2, 2) = 5.0;
  a(0, 1) = a(1, 0) = 3.0;

  const auto b = u::rotate(a, R);

  // (R a R^T)_ij = R_ip a_pq R_jq, written out for a z-rotation.
  const T e11 = c * c * a(0, 0) - 2 * c * sn * a(0, 1) + sn * sn * a(1, 1);
  const T e22 = sn * sn * a(0, 0) + 2 * sn * c * a(0, 1) + c * c * a(1, 1);
  const T e12 = c * sn * a(0, 0) + (c * c - sn * sn) * a(0, 1) - sn * c * a(1, 1);

  EXPECT_NEAR(b(0, 0), e11, 1e-12);
  EXPECT_NEAR(b(1, 1), e22, 1e-12);
  EXPECT_NEAR(b(0, 1), e12, 1e-12);
  EXPECT_NEAR(b(1, 0), e12, 1e-12);
  EXPECT_NEAR(b(2, 2), 5.0, 1e-12);

  // Guard the guard: this fixture must be able to tell the two senses apart,
  // unlike the 90-degree diagonal one. Stated as the property that matters —
  // the observed component differs from what R^T a R would have produced — so
  // there is no magnitude threshold to pick arbitrarily.
  const T e12_transposed =
      -c * sn * a(0, 0) + (c * c - sn * sn) * a(0, 1) + sn * c * a(1, 1);
  EXPECT_GT(std::abs(b(0, 1) - e12_transposed), 1e-6)
      << "fixture cannot distinguish R a R^T from R^T a R";
}

TEST(UmatConversionRotation, RotationPreservesInvariants) {
  T buf[9];
  rot_z90_buffer(buf);
  const auto R = u::rotation_from_buffer<T>(buf);
  const auto a = make_sym();
  const auto b = u::rotate(a, R);

  EXPECT_NEAR(tmech::trace(b), tmech::trace(a), 1e-12);
  EXPECT_NEAR(tmech::dcontract(b, b), tmech::dcontract(a, a), 1e-12);
}

TEST(UmatConversionRotation, RankFourRotationAgreesWithRankTwo) {
  T buf[9];
  rot_z90_buffer(buf);
  const auto R = u::rotation_from_buffer<T>(buf);

  const auto C = make_isotropic(166.67, 76.92);
  const auto Cr = u::rotate(C, R);
  // An isotropic tangent is invariant under any rotation.
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          EXPECT_NEAR(Cr(i, j, k, l), C(i, j, k, l), 1e-9) << i << j << k << l;

  // Rotating C : a must equal (rotated C) : (rotated a).
  const auto a = make_sym();
  tensor2 Ca;
  Ca = tmech::dcontract(C, a);
  tensor2 rhs;
  rhs = tmech::dcontract(Cr, u::rotate(a, R));
  const auto lhs = u::rotate(Ca, R);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) EXPECT_NEAR(lhs(i, j), rhs(i, j), 1e-9);
}

TEST(UmatConversionTangent, MinorSymmetryPredicate) {
  EXPECT_TRUE(u::is_minor_symmetric(make_isotropic(166.67, 76.92)));
  EXPECT_TRUE(u::is_minor_symmetric(make_major_asymmetric()));  // minor-sym only

  const auto I = tmech::eye<T, 3, 2>();
  tensor4 bad;
  bad = tmech::otimesu(I, I);
  EXPECT_FALSE(u::is_minor_symmetric(bad));
}

}  // namespace
