#ifndef NUMSIM_MATERIALS_REPORT_LOAD_DRIVER_H
#define NUMSIM_MATERIALS_REPORT_LOAD_DRIVER_H

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <cstddef>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/umat/material_point_evaluator.h"

/// Mixed strain/stress control of one material point, for reports and checks.
/// Slots follow Abaqus (11, 22, 33, 12, 13, 23) with engineering shear strains.
namespace numsim::materials::report {

/// true = the stress of that slot is prescribed, false = its strain.
using control_mask = std::array<bool, 6>;

namespace control {
inline constexpr control_mask strain{false, false, false, false, false, false};
inline constexpr control_mask uniaxial_stress{false, true, true, true, true, true};
inline constexpr control_mask stress{true, true, true, true, true, true};
} // namespace control

template <typename Traits>
class load_driver {
public:
  using value_type = typename Traits::value_type;
  using evaluator = umat::material_point_evaluator<Traits>;
  using vector6 = tmech::tensor<value_type, 6, 1>;
  using matrix6 = tmech::tensor<value_type, 6, 2>;

  struct options {
    value_type tolerance{1e-10}; ///< on |P(sigma - sigma*)| relative to |sigma|
    int max_iter{25};
  };

  struct step_result {
    vector6 strain;
    vector6 stress;
    matrix6 tangent;
    std::vector<value_type> residuals; ///< |P(sigma - sigma*)| before each correction
    bool converged{false};
  };

  explicit load_driver(evaluator &ev, options opts = {})
      : m_ev(ev), m_opts(opts), m_statev(ev.nstatv()), m_statev_n(ev.nstatv()) {}

  /// Prescribed strain slots come from `strain`, prescribed stress slots from `stress`.
  step_result step(vector6 const &strain, vector6 const &stress, control_mask const &mask, value_type time,
                   value_type dtime) {
    auto const P{projector(mask)};
    auto const I{tmech::eye<value_type, 6, 2>()};
    m_statev_n = m_statev;
    m_strain_n_prev = m_strain_n;
    m_time = time;
    m_dtime = dtime;

    step_result res;
    res.strain = mv(I - P, strain) + mv(P, m_strain_n);
    for (int it{0}; it <= m_opts.max_iter; ++it) {
      evaluate(res.strain, m_statev_n.data(), res.stress, res.tangent);
      vector6 const r{mv(P, res.stress - stress)};
      res.residuals.push_back(tmech::norm(r));
      auto const scale{std::max(tmech::norm(res.stress), tmech::norm(mv(P, stress)))};
      if (res.residuals.back() <= m_opts.tolerance * scale) {
        res.converged = true;
        break;
      }
      matrix6 const A{mm(mm(P, res.tangent), P) + (I - P)};
      res.strain = tmech::eval(res.strain - solve(A, r));
    }
    if (res.converged) {
      m_ev.store_statev(m_statev.data());
      m_strain_n = res.strain;
    }
    return res;
  }

  /// Relative Frobenius error of the tangent of `res` against central differences of the stress.
  value_type tangent_error(step_result const &res, value_type h = 1e-7) {
    matrix6 fd;
    vector6 sp, sm;
    matrix6 unused;
    for (std::size_t j{0}; j < 6; ++j) {
      vector6 e;
      e(j) = h;
      evaluate(res.strain + e, m_statev_n.data(), sp, unused);
      evaluate(res.strain - e, m_statev_n.data(), sm, unused);
      fd += tmech::otimes(vector6{(sp - sm) / (2 * h)}, vector6{e / h});
    }
    return tmech::norm(fd - res.tangent) / tmech::norm(res.tangent);
  }

  [[nodiscard]] std::vector<value_type> const &statev() const noexcept { return m_statev; }

private:
  template <typename A, typename B> static auto mv(A const &a, B const &b) {
    return vector6{tmech::inner_product<tmech::sequence<2>, tmech::sequence<1>>(a, b)};
  }
  template <typename A, typename B> static auto mm(A const &a, B const &b) {
    return matrix6{tmech::inner_product<tmech::sequence<2>, tmech::sequence<1>>(a, b)};
  }

  /// Gaussian elimination with partial pivoting (tmech::inv covers 2x2 and 3x3 only).
  static vector6 solve(matrix6 A, vector6 b) {
    for (std::size_t k{0}; k < 6; ++k) {
      std::size_t p{k};
      for (std::size_t i{k + 1}; i < 6; ++i)
        if (std::abs(A(i, k)) > std::abs(A(p, k)))
          p = i;
      for (std::size_t j{0}; j < 6; ++j)
        std::swap(A(k, j), A(p, j));
      std::swap(b(k), b(p));
      for (std::size_t i{k + 1}; i < 6; ++i) {
        auto const f{A(i, k) / A(k, k)};
        for (std::size_t j{k}; j < 6; ++j)
          A(i, j) -= f * A(k, j);
        b(i) -= f * b(k);
      }
    }
    vector6 x;
    for (std::size_t k{6}; k-- > 0;) {
      auto s{b(k)};
      for (std::size_t j{k + 1}; j < 6; ++j)
        s -= A(k, j) * x(j);
      x(k) = s / A(k, k);
    }
    return x;
  }

  static matrix6 projector(control_mask const &mask) {
    matrix6 P;
    for (std::size_t i{0}; i < 6; ++i)
      P(i, i) = mask[i] ? 1 : 0;
    return P;
  }

  void evaluate(vector6 const &strain, value_type const *statev, vector6 &stress, matrix6 &tangent) {
    std::array<value_type, 6> old6, new6, sig6;
    std::array<value_type, 36> tan36;
    tmech::adaptor<value_type, 6, 1, tmech::full<6>>(old6.data()) = m_strain_n_prev;
    tmech::adaptor<value_type, 6, 1, tmech::full<6>>(new6.data()) = strain;
    m_ev.evaluate_canonical(statev, old6.data(), new6.data(), m_time, m_dtime, sig6.data(), tan36.data());
    stress = tmech::adaptor<value_type, 6, 1, tmech::full<6>>(sig6.data());
    tangent = tmech::trans(tmech::adaptor<value_type, 6, 2, tmech::full<6>>(tan36.data())); // column-major
  }

  evaluator &m_ev;
  options m_opts;
  std::vector<value_type> m_statev, m_statev_n;
  vector6 m_strain_n, m_strain_n_prev;
  value_type m_time{0}, m_dtime{0};
};

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_LOAD_DRIVER_H
