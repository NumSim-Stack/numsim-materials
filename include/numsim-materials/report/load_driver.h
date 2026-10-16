#ifndef NUMSIM_MATERIALS_REPORT_LOAD_DRIVER_H
#define NUMSIM_MATERIALS_REPORT_LOAD_DRIVER_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/umat/material_point_evaluator.h"
#include "numsim-materials/umat/tensor_conversion.h"

/// Mixed strain/stress control of one material point, for reports and checks.
namespace numsim::materials::report {

/// Per Abaqus slot (11, 22, 33, 12, 13, 23): true = stress prescribed, false = strain.
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
  using tensor2 = tmech::tensor<value_type, 3, 2>;
  using tensor4 = tmech::tensor<value_type, 3, 4>;

  struct options {
    value_type tolerance{1e-10}; ///< on |P:(sigma - sigma*)| relative to |sigma|
    int max_iter{25};
  };

  struct step_result {
    tensor2 strain;
    tensor2 stress;
    tensor4 tangent;
    std::vector<value_type> residuals; ///< |P:(sigma - sigma*)| before each correction
    bool converged{false};
  };

  explicit load_driver(evaluator &ev, options opts = {})
      : m_ev(ev), m_opts(opts), m_statev(ev.nstatv()), m_statev_n(ev.nstatv()) {}

  /// Prescribed strain components come from `strain`, prescribed stress components from `stress`.
  step_result step(tensor2 const &strain, tensor2 const &stress, control_mask const &mask, value_type time,
                   value_type dtime) {
    auto const P{projector(mask)};
    auto const &Is{sym_identity()};
    // Iskw makes A regular on the full 9x9 space; residual and correction stay symmetric.
    tensor4 const Q{Is - P + skew_identity()};
    m_statev_n = m_statev;
    m_strain_n_prev = m_strain_n;
    m_time = time;
    m_dtime = dtime;

    step_result res;
    res.strain = tmech::dcontract(tensor4{Is - P}, strain) + tmech::dcontract(P, m_strain_n);
    for (int it{0}; it <= m_opts.max_iter; ++it) {
      evaluate(res.strain, res.stress, res.tangent);
      tensor2 const r{tmech::dcontract(P, tensor2{res.stress - stress})};
      res.residuals.push_back(tmech::norm(r));
      auto const scale{std::max(tmech::norm(res.stress), tmech::norm(tensor2{tmech::dcontract(P, stress)}))};
      if (res.residuals.back() <= m_opts.tolerance * scale) {
        res.converged = true;
        break;
      }
      tensor4 const A{tmech::dcontract(tensor4{tmech::dcontract(P, res.tangent)}, P) + Q};
      res.strain = tmech::eval(res.strain - tmech::dcontract(tensor4{tmech::invf(A)}, r));
    }
    if (res.converged) {
      m_ev.store_statev(m_statev.data());
      m_strain_n = res.strain;
    }
    return res;
  }

  /// Relative error of the tangent of `res` against central differences of the stress.
  value_type tangent_error(step_result const &res, value_type h = 1e-7) {
    tensor4 fd;
    tensor2 sp, sm;
    tensor4 unused;
    for (auto const &E : sym_basis()) {
      evaluate(tensor2{res.strain + h * E}, sp, unused);
      evaluate(tensor2{res.strain - h * E}, sm, unused);
      fd += tmech::otimes(tensor2{(sp - sm) / (2 * h)}, E);
    }
    return tmech::norm(tensor4{fd - res.tangent}) / tmech::norm(res.tangent);
  }

  [[nodiscard]] std::vector<value_type> const &statev() const noexcept { return m_statev; }

private:
  /// Orthonormal basis of symmetric tensors, in Abaqus slot order.
  static std::array<tensor2, 6> const &sym_basis() {
    static auto const basis = [] {
      constexpr std::array<std::pair<std::size_t, std::size_t>, 6> ij{{{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}}};
      std::array<tensor2, 6> b;
      for (std::size_t a{0}; a < 6; ++a) {
        auto const [i, j]{ij[a]};
        b[a](i, j) = b[a](j, i) = i == j ? value_type{1} : value_type{1} / std::sqrt(value_type{2});
      }
      return b;
    }();
    return basis;
  }

  static tensor4 const &sym_identity() {
    static tensor4 const Is{[] {
      auto const I{tmech::eye<value_type, 3, 2>()};
      return tensor4{(tmech::otimesu(I, I) + tmech::otimesl(I, I)) * value_type{0.5}};
    }()};
    return Is;
  }

  static tensor4 const &skew_identity() {
    static tensor4 const Iskw{[] {
      auto const I{tmech::eye<value_type, 3, 2>()};
      return tensor4{(tmech::otimesu(I, I) - tmech::otimesl(I, I)) * value_type{0.5}};
    }()};
    return Iskw;
  }

  /// Projector onto the stress-controlled components.
  static tensor4 projector(control_mask const &mask) {
    tensor4 P;
    for (std::size_t a{0}; a < 6; ++a)
      if (mask[a])
        P += tmech::otimes(sym_basis()[a], sym_basis()[a]);
    return P;
  }

  void evaluate(tensor2 const &strain, tensor2 &stress, tensor4 &tangent) {
    value_type old6[umat::canonical_width], new6[umat::canonical_width], sig6[umat::canonical_width];
    value_type tan36[umat::canonical_width * umat::canonical_width];
    umat::strain_to_buffer(m_strain_n_prev, old6);
    umat::strain_to_buffer(strain, new6);
    m_ev.evaluate_canonical(m_statev_n.data(), old6, new6, m_time, m_dtime, sig6, tan36);
    stress = umat::stress_from_buffer(sig6);
    tangent = umat::tangent_from_buffer(tan36);
  }

  evaluator &m_ev;
  options m_opts;
  std::vector<value_type> m_statev, m_statev_n;
  tensor2 m_strain_n, m_strain_n_prev;
  value_type m_time{0}, m_dtime{0};
};

} // namespace numsim::materials::report

#endif // NUMSIM_MATERIALS_REPORT_LOAD_DRIVER_H
