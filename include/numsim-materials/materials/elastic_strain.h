#ifndef NUMSIM_MATERIALS_ELASTIC_STRAIN_H
#define NUMSIM_MATERIALS_ELASTIC_STRAIN_H

#include <string>
#include <vector>

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/tensor_identities.h"

namespace numsim::materials {

/// Elastic strain eps_e = eps - sum_i eps*_i: the strain ("strain_source", "strain_property") less the
/// eigenstrains ("eigenstrain_sources", "material" or "material::property", property "eigenstrain"; none:
/// the strain passes through). Output "strain".
template <typename Traits>
class elastic_strain final : public material_base<elastic_strain<Traits>, Traits> {
public:
  using base = material_base<elastic_strain<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;
  using input2 = input_property<tensor2, property_traits>;

  template <typename... Args>
  explicit elastic_strain(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_strain(base::template add_output<tensor2>("strain")),
        m_eps(base::template add_input<tensor2>(base::template get_parameter<std::string>("strain_source"),
                                                base::template get_parameter<std::string>("strain_property"), EdgeKind::Global)) {
    if (base::m_parameter_handler.contains("eigenstrain_sources"))
      for (auto const &source : base::template get_parameter<std::vector<std::string>>("eigenstrain_sources")) {
        auto const split{source.find("::")};
        m_eigenstrains.push_back(&base::template add_input<tensor2>(
            source.substr(0, split), split == std::string::npos ? std::string{"eigenstrain"} : source.substr(split + 2),
            EdgeKind::Global));
      }
    base::add_callback(&elastic_strain::compute, m_strain).reads(m_eps, m_eigenstrains).partials(m_strain, m_eps, m_eigenstrains);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "strain_source");
    optional<std::string>(para, "strain_property", std::string{"strain"});
    para.template insert<std::vector<std::string>>("eigenstrain_sources"); // optional: none passes the strain through
    return para;
  }

  void compute() {
    tensor2 elastic{m_eps.get()};
    for (auto const *eigenstrain : m_eigenstrains)
      elastic = tensor2{elastic - eigenstrain->get()};
    m_strain = elastic;
    if (!base::want_partials())
      return;
    if (base::want_partial(m_strain, m_eps))
      base::partial(m_strain, m_eps) = sym_identity<value_type, Dim>();
    for (auto const *eigenstrain : m_eigenstrains)
      if (base::want_partial(m_strain, *eigenstrain))
        base::partial(m_strain, *eigenstrain) = value_type{-1} * sym_identity<value_type, Dim>();
  }

private:
  tensor2 &m_strain;
  input2 const &m_eps;
  std::vector<input2 const *> m_eigenstrains;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_ELASTIC_STRAIN_H
