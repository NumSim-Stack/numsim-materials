#ifndef NUMSIM_MATERIALS_MEAN_FIELD_STRAIN_H
#define NUMSIM_MATERIALS_MEAN_FIELD_STRAIN_H

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/tensor_identities.h"

namespace numsim::materials {

/// Strain localisation of a mean-field scheme (voigt, reuss, hill, dilute, mori_tanaka): from the composite
/// strain increment and the phases' tangents, fractions and (dilute, Mori-Tanaka) Eshelby tensors it produces
/// one strain per phase,
/// "strain_<phase>" = strain_n + A_r : d_strain (history), and the concentrations "concentration_<phase>".
/// The phase materials consume those strains; mean_field_stress averages what they produce. A nonlinear
/// phase closes a cycle (its tangent depends on its strain): its phase node reads the tangent over a local
/// edge, so one update() is one pass and the residual of mean_field_stress says when the passes agree.
///
/// Parameters: "strain_source" (history "strain"), "matrix" and "phases" (phase nodes; the matrix needs no
/// Eshelby tensor, the rule-of-mixture kernels none at all; Voigt reads no tangents either), "kernel"
/// (voigt | reuss | hill | dilute | mori_tanaka).
template <typename Traits>
class mean_field_strain final : public material_base<mean_field_strain<Traits>, Traits> {
public:
  using base = material_base<mean_field_strain<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  static_assert(Dim == 3, "mean_field_strain: the Eshelby tensors are three-dimensional");
  using tensor2 = tmech::tensor<value_type, 3, 2>;
  using tensor4 = tmech::tensor<value_type, 3, 4>;

  enum class kernel_kind { voigt, reuss, hill, dilute, mori_tanaka };

  template <typename... Args>
  explicit mean_field_strain(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_kernel(parse_kernel(base::template get_parameter<std::string>("kernel"))),
        m_strain(base::template add_input_history<tensor2>(
            connection_source{base::template get_parameter<std::string>("strain_source"), "strain"}, EdgeKind::Global)) {
    auto const &matrix{base::template get_parameter<std::string>("matrix")};
    auto const &names{base::template get_parameter<std::vector<std::string>>("phases")};
    if (names.empty())
      throw std::invalid_argument("mean_field_strain '" + base::name() + "': needs at least one phase");
    bool const tangents{m_kernel != kernel_kind::voigt};
    bool const eshelby{m_kernel == kernel_kind::dilute || m_kernel == kernel_kind::mori_tanaka};
    if (tangents)
      m_Cm = &base::template add_input<tensor4>(matrix, "tangent", EdgeKind::Global);
    m_cm = &base::template add_input<value_type>(matrix, "volume_fraction", EdgeKind::Global);
    m_A_matrix = &base::template add_output<tensor4>("concentration_" + matrix);
    m_eps_matrix = &base::template add_history_output<tensor2>("strain_" + matrix, &mean_field_strain::compute);
    for (auto const &n : names) {
      m_C.push_back(tangents ? &base::template add_input<tensor4>(n, "tangent", EdgeKind::Global) : nullptr);
      m_c.push_back(&base::template add_input<value_type>(n, "volume_fraction", EdgeKind::Global));
      m_S.push_back(eshelby ? &base::template add_input<tensor4>(n, "eshelby", EdgeKind::Global) : nullptr);
      m_A.push_back(&base::template add_output<tensor4>("concentration_" + n));
      m_eps.push_back(&base::template add_history_output<tensor2>("strain_" + n));
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "strain_source");
    required<std::string>(para, "matrix");
    required<std::vector<std::string>>(para, "phases");
    optional<std::string>(para, "kernel", std::string{"mori_tanaka"});
    return para;
  }

  /// One callback for every output: the concentrations of the tangents as they stand, then the strains.
  void compute() {
    auto const &Is{sym_identity<value_type, 3>()};
    auto const n{m_C.size()};
    value_type const cm{m_cm->get()};
    value_type total{cm};
    for (auto const *c : m_c)
      total += c->get();
    if (std::abs(total - 1) > value_type{1e-10})
      throw std::invalid_argument("mean_field_strain '" + base::name() + "': the volume fractions must sum to one");
    switch (m_kernel) {
    case kernel_kind::voigt: // equal strain
      *m_A_matrix = Is;
      for (std::size_t i{0}; i < n; ++i)
        *m_A[i] = Is;
      break;
    case kernel_kind::reuss:
    case kernel_kind::hill: { // equal stress: A_r = C_r^-1 : C_R, C_R = (sum c_r C_r^-1)^-1; Hill the mean with Voigt
      tensor4 const Cm_inv{tmech::inv(m_Cm->get())};
      std::vector<tensor4> C_inv(n);
      tensor4 compliance{cm * Cm_inv};
      for (std::size_t i{0}; i < n; ++i) {
        C_inv[i] = tmech::inv(m_C[i]->get());
        compliance += m_c[i]->get() * C_inv[i];
      }
      tensor4 const C_R{tmech::inv(compliance)};
      value_type const w{m_kernel == kernel_kind::hill ? value_type{0.5} : value_type{1}};
      *m_A_matrix = (1 - w) * Is + w * tmech::dcontract(Cm_inv, C_R);
      for (std::size_t i{0}; i < n; ++i)
        *m_A[i] = (1 - w) * Is + w * tmech::dcontract(C_inv[i], C_R);
      break;
    }
    case kernel_kind::dilute:
    case kernel_kind::mori_tanaka: {
      auto const &Cm{m_Cm->get()};
      tensor4 const Cm_inv{tmech::inv(Cm)};
      std::vector<tensor4> T(n); // dilute: T_i = (Isym + S_i : Cm^-1 : (C_i - Cm))^-1
      for (std::size_t i{0}; i < n; ++i)
        T[i] = tmech::inv(Is + tmech::dcontract(m_S[i]->get(), tensor4{tmech::dcontract(Cm_inv, tensor4{m_C[i]->get() - Cm})}));
      if (m_kernel == kernel_kind::dilute) {
        *m_A_matrix = Is;
        for (std::size_t i{0}; i < n; ++i)
          *m_A[i] = T[i];
      } else {
        tensor4 sum{cm * Is};
        for (std::size_t i{0}; i < n; ++i)
          sum += m_c[i]->get() * T[i];
        *m_A_matrix = tmech::inv(sum);
        for (std::size_t i{0}; i < n; ++i)
          *m_A[i] = tmech::dcontract(T[i], *m_A_matrix);
      }
      break;
    }
    }
    tensor2 const d_eps{m_strain.new_value() - m_strain.old_value()};
    m_eps_matrix->new_value() = m_eps_matrix->old_value() + tmech::dcontract(*m_A_matrix, d_eps);
    for (std::size_t i{0}; i < n; ++i)
      m_eps[i]->new_value() = m_eps[i]->old_value() + tmech::dcontract(*m_A[i], d_eps);
  }

private:
  static kernel_kind parse_kernel(std::string const &k) {
    if (k == "voigt")
      return kernel_kind::voigt;
    if (k == "reuss")
      return kernel_kind::reuss;
    if (k == "hill")
      return kernel_kind::hill;
    if (k == "dilute")
      return kernel_kind::dilute;
    if (k == "mori_tanaka")
      return kernel_kind::mori_tanaka;
    throw std::invalid_argument("mean_field_strain: unknown kernel '" + k + "' (voigt, reuss, hill, dilute, mori_tanaka)");
  }

  kernel_kind m_kernel;
  input_history<tensor2, property_traits> const &m_strain;
  input_property<tensor4, property_traits> const *m_Cm{nullptr};
  input_property<value_type, property_traits> const *m_cm{nullptr};
  tensor4 *m_A_matrix{nullptr};
  history_property<tensor2> *m_eps_matrix{nullptr};
  std::vector<input_property<tensor4, property_traits> const *> m_C, m_S;
  std::vector<input_property<value_type, property_traits> const *> m_c;
  std::vector<tensor4 *> m_A;
  std::vector<history_property<tensor2> *> m_eps;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_MEAN_FIELD_STRAIN_H
