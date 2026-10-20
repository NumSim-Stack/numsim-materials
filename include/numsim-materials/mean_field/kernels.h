#ifndef NUMSIM_MATERIALS_MEAN_FIELD_KERNELS_H
#define NUMSIM_MATERIALS_MEAN_FIELD_KERNELS_H

#include <map>
#include <memory>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/isotropize.h"
#include "numsim-materials/core/object_spec.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/mean_field/kernel.h"
#include "numsim-materials/mean_field/shapes.h"

namespace numsim::materials::mean_field {

namespace detail {

/// The dilute concentration of a phase C_r in a reference medium: (I + S : C_ref^-1 : (C_r - C_ref))^-1.
template <typename T>
tmech::tensor<T, 3, 4> dilute(tmech::tensor<T, 3, 4> const &S, tmech::tensor<T, 3, 4> const &C_ref_inv, tmech::tensor<T, 3, 4> const &C_ref,
                              tmech::tensor<T, 3, 4> const &C) {
  return tmech::inv(sym_identity<T, 3>() + tmech::dcontract(S, tmech::tensor<T, 3, 4>{tmech::dcontract(C_ref_inv, tmech::tensor<T, 3, 4>{C - C_ref})}));
}

} // namespace detail

/// Equal strain: A_r = I.
template <typename Traits>
class voigt final : public kernel_base<voigt<Traits>, Traits> {
public:
  using base = kernel_base<voigt<Traits>, Traits>;
  using typename kernel<Traits>::value_type;
  using typename kernel<Traits>::tensor4;
  explicit voigt(typename base::parameter_handler p) : base(std::move(p)) {}
  [[nodiscard]] kernel_needs needs() const noexcept override { return {false, false}; }
  void concentrations(std::span<phase_view<value_type> const>, std::span<tensor4> A) override {
    for (auto &a : A)
      a = sym_identity<value_type, 3>();
  }
};

/// Equal stress weighted with equal strain: A_r = (1 - w) I + w C_r^-1 : C_R, C_R = (sum_r c_r C_r^-1)^-1;
/// Reuss w = 1, Hill w = 1/2.
template <typename Traits>
class reuss_hill_base {
protected:
  using value_type = typename Traits::value_type;
  using tensor4 = tmech::tensor<value_type, 3, 4>;
  static void compute(value_type w, std::span<phase_view<value_type> const> phases, std::span<tensor4> A) {
    std::vector<tensor4> C_inv(phases.size());
    tensor4 compliance;
    for (std::size_t r{0}; r < phases.size(); ++r) {
      C_inv[r] = tmech::inv(*phases[r].tangent);
      compliance += phases[r].fraction * C_inv[r];
    }
    tensor4 const C_R{tmech::inv(compliance)};
    for (std::size_t r{0}; r < phases.size(); ++r)
      A[r] = (1 - w) * sym_identity<value_type, 3>() + w * tmech::dcontract(C_inv[r], C_R);
  }
};

template <typename Traits>
class reuss final : public kernel_base<reuss<Traits>, Traits>, reuss_hill_base<Traits> {
public:
  using base = kernel_base<reuss<Traits>, Traits>;
  using typename kernel<Traits>::value_type;
  using typename kernel<Traits>::tensor4;
  explicit reuss(typename base::parameter_handler p) : base(std::move(p)) {}
  [[nodiscard]] kernel_needs needs() const noexcept override { return {true, false}; }
  void concentrations(std::span<phase_view<value_type> const> phases, std::span<tensor4> A) override {
    reuss_hill_base<Traits>::compute(value_type{1}, phases, A);
  }
};

template <typename Traits>
class hill final : public kernel_base<hill<Traits>, Traits>, reuss_hill_base<Traits> {
public:
  using base = kernel_base<hill<Traits>, Traits>;
  using typename kernel<Traits>::value_type;
  using typename kernel<Traits>::tensor4;
  explicit hill(typename base::parameter_handler p) : base(std::move(p)) {}
  [[nodiscard]] kernel_needs needs() const noexcept override { return {true, false}; }
  void concentrations(std::span<phase_view<value_type> const> phases, std::span<tensor4> A) override {
    reuss_hill_base<Traits>::compute(value_type{0.5}, phases, A);
  }
};

/// Every inclusion alone in the matrix (the first phase): A_r = T_r from the phase's Eshelby tensor and
/// A_m = (I - sum_r c_r T_r) / c_m, so that the phase strains average to the composite strain.
template <typename Traits>
class dilute final : public kernel_base<dilute<Traits>, Traits> {
public:
  using base = kernel_base<dilute<Traits>, Traits>;
  using typename kernel<Traits>::value_type;
  using typename kernel<Traits>::tensor4;
  explicit dilute(typename base::parameter_handler p) : base(std::move(p)) {}
  [[nodiscard]] kernel_needs needs() const noexcept override { return {true, true}; }
  void concentrations(std::span<phase_view<value_type> const> phases, std::span<tensor4> A) override {
    auto const &Cm{*phases[0].tangent};
    tensor4 const Cm_inv{tmech::inv(Cm)};
    tensor4 rest{sym_identity<value_type, 3>()};
    for (std::size_t r{1}; r < phases.size(); ++r) {
      A[r] = detail::dilute(*phases[r].eshelby, Cm_inv, Cm, *phases[r].tangent);
      rest -= phases[r].fraction * A[r];
    }
    A[0] = rest / phases[0].fraction;
  }
};

/// The inclusions in the matrix strain: A_m = (c_m I + sum_r c_r T_r)^-1, A_r = T_r : A_m.
template <typename Traits>
class mori_tanaka final : public kernel_base<mori_tanaka<Traits>, Traits> {
public:
  using base = kernel_base<mori_tanaka<Traits>, Traits>;
  using typename kernel<Traits>::value_type;
  using typename kernel<Traits>::tensor4;
  explicit mori_tanaka(typename base::parameter_handler p) : base(std::move(p)) {}
  [[nodiscard]] kernel_needs needs() const noexcept override { return {true, true}; }
  void concentrations(std::span<phase_view<value_type> const> phases, std::span<tensor4> A) override {
    auto const &Cm{*phases[0].tangent};
    tensor4 const Cm_inv{tmech::inv(Cm)};
    tensor4 sum{phases[0].fraction * sym_identity<value_type, 3>()};
    for (std::size_t r{1}; r < phases.size(); ++r) {
      A[r] = detail::dilute(*phases[r].eshelby, Cm_inv, Cm, *phases[r].tangent);
      sum += phases[r].fraction * A[r];
    }
    A[0] = tmech::inv(sum);
    for (std::size_t r{1}; r < phases.size(); ++r)
      A[r] = tmech::dcontract(tensor4{A[r]}, A[0]);
  }
};

/// Every phase embedded in the effective medium C: T_r = (I + S_r(nu) : C^-1 : (C_r - C))^-1,
/// A_r = T_r : (sum_s c_s T_s)^-1, C = sum_r c_r C_r : A_r, iterated to "tolerance" with relaxed steps from
/// the last converged C (Voigt at first). S_r comes from the phase's shape at the Poisson's ratio of the
/// isotropised C (exact for an isotropic C).
///
/// Parameters: "shapes" {phase: {"type": sphere | cylinder | spheroid | ellipsoid, ...}} for every phase,
/// "relaxation" (0.5, in (0, 1]), "tolerance" (1e-10), "max_iterations" (100).
template <typename Traits>
class self_consistent final : public kernel_base<self_consistent<Traits>, Traits> {
public:
  using base = kernel_base<self_consistent<Traits>, Traits>;
  using typename kernel<Traits>::value_type;
  using typename kernel<Traits>::tensor4;
  using spec = object_spec<typename base::parameter_handler>;

  explicit self_consistent(typename base::parameter_handler p)
      : base(std::move(p)), m_relaxation(base::template get<value_type>("relaxation")), m_tolerance(base::template get<value_type>("tolerance")),
        m_max_iterations(base::template get<int>("max_iterations")) {
    if (!(m_relaxation > 0 && m_relaxation <= 1))
      throw std::invalid_argument("self_consistent: relaxation must lie in (0, 1]");
    if (!(m_tolerance > 0))
      throw std::invalid_argument("self_consistent: tolerance must be positive");
    if (m_max_iterations < 1)
      throw std::invalid_argument("self_consistent: max_iterations must be at least one");
  }

  static typename base::input_parameter_controller parameters() {
    typename base::input_parameter_controller para;
    required<std::map<std::string, spec>>(para, "shapes");
    optional<value_type>(para, "relaxation", value_type{0.5});
    optional<value_type>(para, "tolerance", value_type{1e-10});
    optional<int>(para, "max_iterations", 100);
    return para;
  }

  [[nodiscard]] kernel_needs needs() const noexcept override { return {true, false}; }

  void bind(std::vector<std::string> const &names) override {
    auto const &shapes{base::template get<std::map<std::string, spec>>("shapes")};
    m_shapes.clear();
    for (auto const &n : names) {
      auto const it{shapes.find(n)};
      if (it == shapes.end())
        throw std::invalid_argument("self_consistent: no shape for phase '" + n + "'");
      m_shapes.push_back(shape_registry<Traits>().make(it->second));
    }
  }

  void concentrations(std::span<phase_view<value_type> const> phases, std::span<tensor4> A) override {
    if (m_shapes.size() != phases.size())
      throw std::logic_error("self_consistent: bind() the phase names first");
    tensor4 C{m_C.value_or(tensor4{})};
    if (!m_C)
      for (auto const &p : phases)
        C += p.fraction * *p.tangent;
    std::vector<tensor4> T(phases.size());
    value_type residual{0};
    for (int it{0}; it < m_max_iterations; ++it) {
      tensor4 const C_inv{tmech::inv(C)};
      auto const [K, G]{isotropize(C)};
      value_type const nu{(3 * K - 2 * G) / (2 * (3 * K + G))};
      tensor4 T_sum;
      for (std::size_t r{0}; r < phases.size(); ++r) {
        T[r] = detail::dilute(m_shapes[r]->eshelby(nu), C_inv, C, *phases[r].tangent);
        T_sum += phases[r].fraction * T[r];
      }
      tensor4 const T_inv{tmech::inv(T_sum)};
      tensor4 C_new;
      for (std::size_t r{0}; r < phases.size(); ++r) {
        A[r] = tmech::dcontract(T[r], T_inv);
        C_new += phases[r].fraction * tmech::dcontract(*phases[r].tangent, A[r]);
      }
      residual = tmech::norm(tensor4{C_new - C}) / tmech::norm(C_new);
      if (residual <= m_tolerance) {
        m_C = C_new;
        return;
      }
      C = (1 - m_relaxation) * C + m_relaxation * C_new;
    }
    throw std::runtime_error(std::format("self_consistent: no convergence in {} iterations (residual {:g}, tolerance {:g})",
                                         m_max_iterations, residual, m_tolerance));
  }

private:
  value_type m_relaxation, m_tolerance;
  int m_max_iterations;
  std::vector<std::unique_ptr<shape<Traits>>> m_shapes;
  std::optional<tensor4> m_C; ///< the last converged medium: a warm start, not history
};

} // namespace numsim::materials::mean_field

#endif // NUMSIM_MATERIALS_MEAN_FIELD_KERNELS_H
