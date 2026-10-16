#ifndef NUMSIM_MATERIALS_SHORT_FIBRE_COMPOSITE_H
#define NUMSIM_MATERIALS_SHORT_FIBRE_COMPOSITE_H

#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/elastic_constants.h"
#include "numsim-materials/core/eshelby.h"
#include "numsim-materials/core/isotropize.h"
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/orientation.h"
#include "numsim-materials/core/tensor_identities.h"

namespace numsim::materials {

/// Mori-Tanaka stiffness of spheroidal fibres in an isotropic matrix, averaged over the fibre
/// orientation distribution (ACG with the second-order orientation tensor): either the
/// orientation average of the dilute concentration inside Mori-Tanaka ("concentration") or the
/// average of the unidirectional stiffness ("stiffness"). Outputs "tangent", "stress",
/// "concentration_matrix", "concentration_fibre".
///
/// Parameters: "matrix", "fibre" (isotropic materials providing "tangent"), "volume_fraction",
/// "aspect_ratio" (length over diameter), "orientation" [a11, a22, a33, a12, a13, a23],
/// "averaging", "precision" (Lebedev degree), "strain_source", "tangent_property".
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
  using vector = tmech::tensor<value_type, 3, 1>;

  enum class averaging_kind { concentration, stiffness };

  template <typename... Args>
  explicit short_fibre_composite(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_stress(base::template add_output<tensor2>("stress", &short_fibre_composite::update)),
        m_tangent(base::template add_output<tensor4>("tangent")),
        m_A_matrix(base::template add_output<tensor4>("concentration_matrix")),
        m_A_fibre(base::template add_output<tensor4>("concentration_fibre")),
        m_c(base::template get_parameter<value_type>("volume_fraction")),
        m_aspect_ratio(base::template get_parameter<value_type>("aspect_ratio")),
        m_averaging(parse_averaging(base::template get_parameter<std::string>("averaging"))),
        m_precision(static_cast<std::size_t>(base::template get_parameter<int>("precision"))),
        m_Cm(base::template add_input<tensor4>(base::template get_parameter<std::string>("matrix"),
                                               base::template get_parameter<std::string>("tangent_property"), EdgeKind::Global)),
        m_Cf(base::template add_input<tensor4>(base::template get_parameter<std::string>("fibre"),
                                               base::template get_parameter<std::string>("tangent_property"), EdgeKind::Global)),
        m_strain(base::template add_input<tensor2>(base::template get_parameter<std::string>("strain_source"), "strain",
                                                   EdgeKind::Global)) {
    auto const &a{base::template get_parameter<std::vector<value_type>>("orientation")};
    if (a.size() != 6)
      throw std::invalid_argument("short_fibre_composite '" + base::name() + "': orientation takes [a11, a22, a33, a12, a13, a23]");
    if (!(m_c >= 0 && m_c <= 1) || !(m_aspect_ratio > 0))
      throw std::invalid_argument("short_fibre_composite '" + base::name() + "': needs a fraction in [0, 1] and a positive aspect ratio");
    (void)lebedev::rule(m_precision); // validates the precision
    tensor2 const A{symmetric_tensor<value_type>(std::span<value_type const, 6>{a.data(), 6})};
    m_nodes = orientation::nodes(orientation::fit_acg(A), m_precision);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("matrix").template add<is_required>();
    para.template insert<std::string>("fibre").template add<is_required>();
    para.template insert<value_type>("volume_fraction").template add<is_required>();
    para.template insert<value_type>("aspect_ratio").template add<is_required>();
    para.template insert<std::vector<value_type>>("orientation").template add<is_required>();
    para.template insert<std::string>("averaging").template add<set_default>(std::string{"concentration"});
    para.template insert<int>("precision").template add<set_default>(15);
    para.template insert<std::string>("strain_source").template add<is_required>();
    para.template insert<std::string>("tangent_property").template add<set_default>(std::string{"tangent"});
    return para;
  }

  /// One callback for every output.
  void update() {
    auto const &Is{sym_identity<value_type, 3>()};
    auto const &Cm{m_Cm.get()};
    auto const &Cf{m_Cf.get()};
    value_type const nu{poisson_of(Cm)};
    if (nu != m_cached_nu) { // the Eshelby tensors depend on the matrix only
      m_S.clear();
      tensor4 const S1{eshelby::ellipsoid(nu, m_aspect_ratio, value_type{1}, value_type{1})}; // along x1
      for (auto const &[n, w] : m_nodes)
        m_S.push_back(rotate(S1, rotation_to(n)));
      m_cached_nu = nu;
    }
    tensor4 const Cm_inv{tmech::inv(Cm)};
    tensor4 const dC{Cf - Cm};
    tensor4 const M{tmech::dcontract(Cm_inv, dC)};
    value_type const cm{1 - m_c};
    if (m_averaging == averaging_kind::concentration) {
      tensor4 T_avg;
      for (std::size_t q{0}; q < m_nodes.size(); ++q)
        T_avg += m_nodes[q].second * tmech::inv(Is + tmech::dcontract(m_S[q], M));
      m_A_matrix = tmech::inv(cm * Is + m_c * T_avg);
      m_A_fibre = tmech::dcontract(T_avg, m_A_matrix);
    } else {
      m_A_fibre = tensor4{};
      for (std::size_t q{0}; q < m_nodes.size(); ++q) {
        tensor4 const Tq{tmech::inv(Is + tmech::dcontract(m_S[q], M))};
        m_A_fibre += m_nodes[q].second * tmech::dcontract(Tq, tensor4{tmech::inv(cm * Is + m_c * Tq)});
      }
      m_A_matrix = (Is - m_c * m_A_fibre) / cm;
    }
    m_tangent = Cm + m_c * tmech::dcontract(dC, m_A_fibre);
    m_stress = tmech::dcontract(m_tangent, m_strain.get());
  }

private:
  value_type poisson_of(tensor4 const &C) const {
    auto const [K, G]{isotropize<value_type, 3>(C)};
    if (tmech::norm(tensor4{C - isotropic_stiffness<value_type, 3>(K, G)}) > value_type{1e-8} * tmech::norm(C))
      throw std::invalid_argument("short_fibre_composite '" + base::name() + "': the matrix must be isotropic");
    return poisson_ratio(K, G);
  }

  static averaging_kind parse_averaging(std::string const &s) {
    if (s == "concentration")
      return averaging_kind::concentration;
    if (s == "stiffness")
      return averaging_kind::stiffness;
    throw std::invalid_argument("short_fibre_composite: unknown averaging '" + s + "' (concentration, stiffness)");
  }

  tensor2 &m_stress;
  tensor4 &m_tangent;
  tensor4 &m_A_matrix;
  tensor4 &m_A_fibre;
  value_type const &m_c;
  value_type const &m_aspect_ratio;
  averaging_kind m_averaging;
  std::size_t m_precision;
  input_property<tensor4, property_traits> const &m_Cm;
  input_property<tensor4, property_traits> const &m_Cf;
  input_property<tensor2, property_traits> const &m_strain;
  std::vector<std::pair<vector, value_type>> m_nodes;
  std::vector<tensor4> m_S;
  value_type m_cached_nu{std::numeric_limits<value_type>::quiet_NaN()};
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_SHORT_FIBRE_COMPOSITE_H
