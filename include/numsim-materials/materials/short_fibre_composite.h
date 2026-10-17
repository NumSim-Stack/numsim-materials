#ifndef NUMSIM_MATERIALS_SHORT_FIBRE_COMPOSITE_H
#define NUMSIM_MATERIALS_SHORT_FIBRE_COMPOSITE_H

#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/lebedev.h"
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/orientation.h"
#include "numsim-materials/core/tensor_identities.h"

namespace numsim::materials {

/// Mori-Tanaka stiffness of fibres in an isotropic matrix, averaged over the fibre orientation
/// distribution (ACG with the second-order orientation tensor): the orientation average of the dilute
/// concentration inside Mori-Tanaka ("concentration") or of the unidirectional stiffness ("stiffness").
/// Matrix and fibre are phase nodes (produced tangent and fraction; the fibre's "eshelby" is its shape
/// along x1); everything is formed once in the fibre frame and rotated onto the quadrature directions. A
/// tangent producer: outputs "tangent", "concentration_matrix", "concentration_fibre".
///
/// Parameters: "matrix", "fibre" (phase names), "orientation" [a11, a22, a33, a12, a13, a23], "averaging",
/// "precision" (Lebedev degree).
template <typename Traits>
class short_fibre_composite final : public material_base<short_fibre_composite<Traits>, Traits> {
public:
  using base = material_base<short_fibre_composite<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  static_assert(Dim == 3, "short_fibre_composite: three-dimensional");
  using tensor2 = tmech::tensor<value_type, 3, 2>;
  using tensor4 = tmech::tensor<value_type, 3, 4>;

  enum class averaging_kind { concentration, stiffness };

  template <typename... Args>
  explicit short_fibre_composite(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_tangent(base::template add_output<tensor4>("tangent", &short_fibre_composite::compute)),
        m_A_matrix(base::template add_output<tensor4>("concentration_matrix")),
        m_A_fibre(base::template add_output<tensor4>("concentration_fibre")),
        m_averaging(parse_averaging(base::template get_parameter<std::string>("averaging"))),
        m_Cm(base::template add_input<tensor4>(base::template get_parameter<std::string>("matrix"), "tangent", EdgeKind::Global)),
        m_cm(base::template add_input<value_type>(base::template get_parameter<std::string>("matrix"), "volume_fraction", EdgeKind::Global)),
        m_Cf(base::template add_input<tensor4>(base::template get_parameter<std::string>("fibre"), "tangent", EdgeKind::Global)),
        m_c(base::template add_input<value_type>(base::template get_parameter<std::string>("fibre"), "volume_fraction", EdgeKind::Global)),
        m_S(base::template add_input<tensor4>(base::template get_parameter<std::string>("fibre"), "eshelby", EdgeKind::Global)) {
    auto const &a{base::template get_parameter<std::vector<value_type>>("orientation")};
    auto const precision{base::template get_parameter<int>("precision")};
    if (a.size() != 6)
      throw std::invalid_argument("short_fibre_composite '" + base::name() + "': orientation takes [a11, a22, a33, a12, a13, a23]");
    if (precision < 0)
      throw std::invalid_argument("short_fibre_composite '" + base::name() + "': precision must be a Lebedev degree");
    (void)lebedev::rule(static_cast<std::size_t>(precision)); // validated here: an aligned distribution never uses it
    tensor2 const A{symmetric_tensor<value_type>(std::span<value_type const, 6>{a.data(), 6})};
    for (auto const &[n, w] : orientation::nodes(orientation::fit_acg(A), static_cast<std::size_t>(precision))) {
      tensor2 const R{rotation_to(n)}; // the rank-4 rotation as two fixed tensors per direction
      m_Q.push_back(tensor4{tmech::otimesu(R, R)});
      m_Qt.push_back(tensor4{tmech::otimesu(tmech::trans(R), tmech::trans(R))});
      m_w.push_back(w);
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "matrix");
    required<std::string>(para, "fibre");
    required<std::vector<value_type>>(para, "orientation");
    optional<std::string>(para, "averaging", std::string{"concentration"});
    optional<int>(para, "precision", 23);
    return para;
  }

  /// One callback for every output.
  void compute() {
    auto const &Is{sym_identity<value_type, 3>()};
    auto const &Cm{m_Cm.get()};
    auto const &Cf{m_Cf.get()};
    value_type const c{m_c.get()}, cm{m_cm.get()};
    if (std::abs(cm + c - 1) > value_type{1e-10} || !(c < 1))
      throw std::invalid_argument("short_fibre_composite '" + base::name() + "': the two fractions must sum to one with the fibres below one");
    tensor4 const dC{Cf - Cm};
    tensor4 const T1{tmech::inv(Is + tmech::dcontract(m_S.get(), tensor4{tmech::dcontract(tmech::inv(Cm), dC)}))}; // dilute, fibre frame
    if (m_averaging == averaging_kind::concentration) {
      // <T> and <(Cf - Cm) : T> over the directions, then Mori-Tanaka with the averaged concentration
      tensor4 const D1{tmech::dcontract(dC, T1)};
      tensor4 T_avg, D_avg;
      for (std::size_t q{0}; q < m_w.size(); ++q) {
        T_avg += m_w[q] * rotated(T1, q);
        D_avg += m_w[q] * rotated(D1, q);
      }
      m_A_matrix = tmech::inv(cm * Is + c * T_avg);
      m_A_fibre = tmech::dcontract(T_avg, m_A_matrix);
      m_tangent = Cm + c * tmech::dcontract(D_avg, m_A_matrix);
    } else {
      // the unidirectional Mori-Tanaka stiffness and concentration, averaged over the directions
      tensor4 const A1{tmech::dcontract(T1, tensor4{tmech::inv(cm * Is + c * T1)})};
      tensor4 const C1{Cm + c * tmech::dcontract(dC, A1)};
      m_A_fibre = tensor4{};
      m_tangent = tensor4{};
      for (std::size_t q{0}; q < m_w.size(); ++q) {
        m_A_fibre += m_w[q] * rotated(A1, q);
        m_tangent += m_w[q] * rotated(C1, q);
      }
      m_A_matrix = (Is - c * m_A_fibre) / cm;
    }
  }

private:
  /// X of the fibre frame in direction q: R X R^T on every index.
  [[nodiscard]] tensor4 rotated(tensor4 const &X, std::size_t q) const noexcept {
    return tmech::dcontract(tensor4{tmech::dcontract(m_Q[q], X)}, m_Qt[q]);
  }

  static averaging_kind parse_averaging(std::string const &s) {
    if (s == "concentration")
      return averaging_kind::concentration;
    if (s == "stiffness")
      return averaging_kind::stiffness;
    throw std::invalid_argument("short_fibre_composite: unknown averaging '" + s + "' (concentration, stiffness)");
  }

  tensor4 &m_tangent;
  tensor4 &m_A_matrix;
  tensor4 &m_A_fibre;
  averaging_kind m_averaging;
  input_property<tensor4, property_traits> const &m_Cm;
  input_property<value_type, property_traits> const &m_cm;
  input_property<tensor4, property_traits> const &m_Cf;
  input_property<value_type, property_traits> const &m_c;
  input_property<tensor4, property_traits> const &m_S; ///< the fibre shape along x1
  std::vector<tensor4> m_Q, m_Qt; ///< otimesu(R, R) and its transpose per quadrature direction
  std::vector<value_type> m_w;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_SHORT_FIBRE_COMPOSITE_H
