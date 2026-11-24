#ifndef NUMSIM_MATERIALS_SETH_HILL_STRAIN_H
#define NUMSIM_MATERIALS_SETH_HILL_STRAIN_H

#include <cmath>
#include <string>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials {

namespace detail {
/// f(x) = (x^(m/2) - 1)/m of an eigenvalue x of C, ln(x)/2 for m = 0; expm1 keeps small m accurate.
template <typename T>
struct seth_hill_function {
  T m;
  T apply(T x) const { return m == 0 ? std::log(x) / 2 : std::expm1(m / 2 * std::log(x)) / m; }
  T derivative(T x) const { return std::pow(x, m / 2 - 1) / 2; }
  T second_derivative(T x) const { return (m / 2 - 1) * std::pow(x, m / 2 - 2) / 2; }
};
} // namespace detail

/// Lagrangian strain of the Seth-Hill family E = (C^(m/2) - I)/m, C = F^T F ("deformation_gradient_source"): m = 0
/// Hencky (ln C / 2), 1 Biot, 2 Green-Lagrange, -2 Almansi-type. Outputs "strain", "d_strain_d_right_cauchy_green"
/// (dE/dC) and, on demand, "d2_strain_d_right_cauchy_green2" (d2E/dC2). For anisotropic laws (docs/finite-strain.md).
template <typename Traits>
class seth_hill_strain final : public material_base<seth_hill_strain<Traits>, Traits> {
public:
  using base = material_base<seth_hill_strain<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;
  using tensor6 = tmech::tensor<value_type, Dim, 6>;

  template <typename... Args>
  explicit seth_hill_strain(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_E(base::template add_output<tensor2>("strain")),
        m_dE(base::template add_output<tensor4>("d_strain_d_right_cauchy_green")),
        m_d2E(base::template add_output<tensor6>("d2_strain_d_right_cauchy_green2")),
        m_m(base::template get_parameter<value_type>("m")),
        m_F(base::template add_input<tensor2>(base::template get_parameter<std::string>("deformation_gradient_source"),
                                              "deformation_gradient", EdgeKind::Global)) {
    // no partials(): F is not symmetric and the declared derivatives cover symmetric tensors only
    base::add_callback(&seth_hill_strain::compute, m_E, m_dE, m_d2E).reads(m_F).on_demand(m_d2E);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<value_type>(para, "m");
    required<std::string>(para, "deformation_gradient_source");
    return para;
  }

  void compute() {
    auto const &F{m_F.get()};
    auto const I{tmech::eye<value_type, Dim, 2>()};
    tensor2 const C{tmech::trans(F) * F};
    if (m_m == 2) { // Green-Lagrange: no eigendecomposition, d2E/dC2 = 0
      m_E = value_type{0.5} * (C - I);
      m_dE = value_type{0.25} * (tmech::otimesu(I, I) + tmech::otimesl(I, I));
      if (base::is_read(m_d2E))
        m_d2E.fill(0);
      return;
    }
    auto E{tmech::isotropic_tensor_function(C, detail::seth_hill_function<value_type>{m_m})};
    m_E = E;
    m_dE = E.derivative();
    if (base::is_read(m_d2E))
      m_d2E = E.second_derivative();
  }

private:
  tensor2 &m_E;
  tensor4 &m_dE;
  tensor6 &m_d2E;
  value_type const &m_m;
  input_property<tensor2, property_traits> const &m_F;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_SETH_HILL_STRAIN_H
