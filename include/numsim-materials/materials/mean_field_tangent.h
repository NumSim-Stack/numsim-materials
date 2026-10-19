#ifndef NUMSIM_MATERIALS_MEAN_FIELD_TANGENT_H
#define NUMSIM_MATERIALS_MEAN_FIELD_TANGENT_H

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/tensor_identities.h"

namespace numsim::materials {

/// Mean-field effective stiffness of phases (phase nodes producing "tangent", "volume_fraction" and
/// "eshelby"): dilute, Mori-Tanaka or self-consistent. A tangent producer: outputs "tangent", the strain
/// concentration tensors "concentration_<phase>" (matrix included) and "residual"; the stress is a
/// linear_stress on the tangent. The Eshelby tensors come in as properties, so their Poisson's ratio is
/// whatever node produces it: the matrix moduli for dilute and Mori-Tanaka, the isotropised effective
/// tangent of this node for self-consistent (an isotropic_moduli on "tangent", consumed by the Eshelby nodes
/// over a local edge). Self-consistent is then one relaxed fixed-point step per update; "residual" is the
/// relative change of the tangent and the caller repeats update() until it is small.
///
/// Parameters: "matrix" (a phase; its "eshelby" is read for self-consistent only), "phases", "kernel"
/// (dilute | mori_tanaka | self_consistent), "relaxation" (self-consistent step weight).
template <typename Traits>
class mean_field_tangent final : public material_base<mean_field_tangent<Traits>, Traits> {
public:
  using base = material_base<mean_field_tangent<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  static_assert(Dim == 3, "mean_field_tangent: the Eshelby tensors are three-dimensional");
  using tensor4 = tmech::tensor<value_type, 3, 4>;

  enum class kernel_kind { dilute, mori_tanaka, self_consistent };

  template <typename... Args>
  explicit mean_field_tangent(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_tangent(base::template add_output<tensor4>("tangent", &mean_field_tangent::compute)),
        m_residual(base::template add_output<value_type>("residual")),
        m_kernel(parse_kernel(base::template get_parameter<std::string>("kernel"))),
        m_relaxation(base::template get_parameter<value_type>("relaxation")) {
    auto const &matrix{base::template get_parameter<std::string>("matrix")};
    auto const &names{base::template get_parameter<std::vector<std::string>>("phases")};
    if (names.empty())
      throw std::invalid_argument("mean_field_tangent '" + base::name() + "': needs at least one phase");
    if (!(m_relaxation > 0 && m_relaxation <= 1))
      throw std::invalid_argument("mean_field_tangent '" + base::name() + "': relaxation must lie in (0, 1]");
    m_Cm = &base::template add_input<tensor4>(matrix, "tangent", EdgeKind::Global);
    m_cm = &base::template add_input<value_type>(matrix, "volume_fraction", EdgeKind::Global);
    if (m_kernel == kernel_kind::self_consistent)
      m_Sm = &base::template add_input<tensor4>(matrix, "eshelby", EdgeKind::Global);
    m_A_matrix = &base::template add_output<tensor4>("concentration_" + matrix);
    for (auto const &n : names) {
      m_C.push_back(&base::template add_input<tensor4>(n, "tangent", EdgeKind::Global));
      m_c.push_back(&base::template add_input<value_type>(n, "volume_fraction", EdgeKind::Global));
      m_S.push_back(&base::template add_input<tensor4>(n, "eshelby", EdgeKind::Global));
      m_A.push_back(&base::template add_output<tensor4>("concentration_" + n));
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "matrix");
    required<std::vector<std::string>>(para, "phases");
    optional<std::string>(para, "kernel", std::string{"mori_tanaka"});
    optional<value_type>(para, "relaxation", value_type{0.5});
    return para;
  }

  /// One callback for every output.
  void compute() {
    auto const &Is{sym_identity<value_type, 3>()};
    auto const n{m_C.size()};
    value_type const cm{m_cm->get()};
    value_type total{cm};
    for (auto const *c : m_c)
      total += c->get();
    if (std::abs(total - 1) > value_type{1e-10})
      throw std::invalid_argument("mean_field_tangent '" + base::name() + "': the volume fractions must sum to one");
    auto const &Cm{m_Cm->get()};

    // dilute concentrations in the matrix: T_i = (Isym + S_i : Cm^-1 : (C_i - Cm))^-1
    tensor4 const Cm_inv{tmech::inv(Cm)};
    std::vector<tensor4> T(n);
    for (std::size_t i{0}; i < n; ++i)
      T[i] = partial(m_S[i]->get(), Cm_inv, tensor4{m_C[i]->get() - Cm});
    tensor4 sum{cm * Is};
    for (std::size_t i{0}; i < n; ++i)
      sum += m_c[i]->get() * T[i];
    switch (m_kernel) {
    case kernel_kind::dilute:
      *m_A_matrix = Is;
      for (std::size_t i{0}; i < n; ++i)
        *m_A[i] = T[i];
      m_tangent = effective(Cm);
      m_residual = 0;
      break;
    case kernel_kind::mori_tanaka:
      *m_A_matrix = tmech::inv(sum);
      for (std::size_t i{0}; i < n; ++i)
        *m_A[i] = tmech::dcontract(T[i], *m_A_matrix);
      m_tangent = effective(Cm);
      m_residual = 0;
      break;
    case kernel_kind::self_consistent: {
      // every phase embedded in the previous effective medium C: T_r = (Isym + S_r : C^-1 : (C_r - C))^-1,
      // A_r = T_r : (sum_s c_s T_s)^-1, C_new = sum_r c_r C_r : A_r; the Mori-Tanaka estimate starts it
      if (m_first) {
        tensor4 const A_m{tmech::inv(sum)};
        for (std::size_t i{0}; i < n; ++i)
          *m_A[i] = tmech::dcontract(T[i], A_m);
        m_tangent = effective(Cm);
      }
      tensor4 const C{m_tangent};
      tensor4 const C_inv{tmech::inv(C)};
      tensor4 const Tm{partial(m_Sm->get(), C_inv, tensor4{Cm - C})};
      tensor4 T_sum{cm * Tm};
      for (std::size_t i{0}; i < n; ++i) {
        T[i] = partial(m_S[i]->get(), C_inv, tensor4{m_C[i]->get() - C});
        T_sum += m_c[i]->get() * T[i];
      }
      tensor4 const T_inv{tmech::inv(T_sum)};
      *m_A_matrix = tmech::dcontract(Tm, T_inv);
      tensor4 C_new{cm * tmech::dcontract(Cm, *m_A_matrix)};
      for (std::size_t i{0}; i < n; ++i) {
        *m_A[i] = tmech::dcontract(T[i], T_inv);
        C_new += m_c[i]->get() * tmech::dcontract(m_C[i]->get(), *m_A[i]);
      }
      m_residual = tmech::norm(tensor4{C_new - C}) / tmech::norm(C_new);
      m_tangent = (1 - m_relaxation) * C + m_relaxation * C_new;
      break;
    }
    }
    m_first = false;
  }

private:
  /// (Isym + S : C_ref^-1 : dC)^-1
  static tensor4 partial(tensor4 const &S, tensor4 const &C_ref_inv, tensor4 const &dC) {
    return tmech::inv(sym_identity<value_type, 3>() + tmech::dcontract(S, tensor4{tmech::dcontract(C_ref_inv, dC)}));
  }

  /// C = Cm + sum_i c_i (C_i - Cm) : A_i
  tensor4 effective(tensor4 const &Cm) const noexcept {
    tensor4 C{Cm};
    for (std::size_t i{0}; i < m_C.size(); ++i)
      C += m_c[i]->get() * tmech::dcontract(tensor4{m_C[i]->get() - Cm}, *m_A[i]);
    return C;
  }

  static kernel_kind parse_kernel(std::string const &k) {
    if (k == "dilute")
      return kernel_kind::dilute;
    if (k == "mori_tanaka")
      return kernel_kind::mori_tanaka;
    if (k == "self_consistent")
      return kernel_kind::self_consistent;
    throw std::invalid_argument("mean_field_tangent: unknown kernel '" + k + "' (dilute, mori_tanaka, self_consistent)");
  }

  tensor4 &m_tangent;
  value_type &m_residual;
  kernel_kind m_kernel;
  value_type const &m_relaxation;
  bool m_first{true}; ///< self-consistent: the first update has no previous effective medium
  input_property<tensor4, property_traits> const *m_Cm{nullptr};
  input_property<value_type, property_traits> const *m_cm{nullptr};
  input_property<tensor4, property_traits> const *m_Sm{nullptr}; ///< self-consistent only
  tensor4 *m_A_matrix{nullptr};
  std::vector<input_property<tensor4, property_traits> const *> m_C, m_S;
  std::vector<input_property<value_type, property_traits> const *> m_c;
  std::vector<tensor4 *> m_A;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_MEAN_FIELD_TANGENT_H
