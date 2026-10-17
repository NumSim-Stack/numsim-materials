#ifndef NUMSIM_MATERIALS_CORE_LOG_STRAIN_H
#define NUMSIM_MATERIALS_CORE_LOG_STRAIN_H

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <Eigen/Dense>
#include <tmech/tmech.h>

/// Logarithmic (Hencky) strain ln V = 1/2 ln(F F^T) and its derivative with respect to F, from an
/// eigendecomposition of b = F F^T with the divided-difference formula, which stays correct through
/// repeated and nearly repeated eigenvalues (tmech's isotropic tensor functions do not, tmech#37).
namespace numsim::materials {

template <typename T>
struct log_strain_result {
  tmech::tensor<T, 3, 2> strain;     ///< ln V
  tmech::tensor<T, 3, 4> derivative; ///< d ln V / dF
};

template <typename T>
log_strain_result<T> log_strain(tmech::tensor<T, 3, 2> const &F) {
  using tensor2 = tmech::tensor<T, 3, 2>;
  using tensor4 = tmech::tensor<T, 3, 4>;
  tensor2 const b{F * tmech::trans(F)};
  Eigen::Matrix<T, 3, 3> B;
  for (std::size_t i{0}; i < 3; ++i)
    for (std::size_t j{0}; j < 3; ++j)
      B(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = b(i, j);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3>> solver(B);
  if (solver.info() != Eigen::Success || !(solver.eigenvalues()(0) > 0))
    throw std::domain_error("log_strain: F F^T must be positive definite");

  std::array<T, 3> lambda;
  std::array<tensor2, 3> E; // eigenprojectors n_i (x) n_i
  for (std::size_t i{0}; i < 3; ++i) {
    lambda[i] = solver.eigenvalues()(static_cast<Eigen::Index>(i));
    tmech::tensor<T, 3, 1> n;
    for (std::size_t k{0}; k < 3; ++k)
      n(k) = solver.eigenvectors()(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(i));
    E[i] = tmech::otimes(n, n);
  }
  auto const f = [](T x) { return std::log(x) / 2; };  // ln V = 1/2 ln b
  auto const df = [](T x) { return 1 / (2 * x); };

  log_strain_result<T> out;
  tensor4 dL_db; // d ln V / db
  for (std::size_t i{0}; i < 3; ++i) {
    out.strain += f(lambda[i]) * E[i];
    dL_db += df(lambda[i]) * tmech::otimes(E[i], E[i]);
    for (std::size_t j{0}; j < 3; ++j) {
      if (j == i)
        continue;
      T const mean{(lambda[i] + lambda[j]) / 2};
      T const g{std::abs(lambda[i] - lambda[j]) > T{1e-8} * mean ? (f(lambda[i]) - f(lambda[j])) / (lambda[i] - lambda[j]) : df(mean)};
      dL_db += g / 2 * (tmech::otimesu(E[i], E[j]) + tmech::otimesl(E[i], E[j]));
    }
  }
  auto const I{tmech::eye<T, 3, 2>()};
  tensor4 const db_dF{tmech::otimesu(I, F) + tmech::otimesl(F, I)}; // db_ij / dF_kl = d_ik F_jl + F_il d_jk
  out.derivative = tmech::dcontract(dL_db, db_dF);
  return out;
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_LOG_STRAIN_H
