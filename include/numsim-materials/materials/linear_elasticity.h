#ifndef LINEAR_ELASTICITY_H
#define LINEAR_ELASTICITY_H

#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/tensor_identities.h"

namespace numsim::materials {

template <typename Traits>
class linear_elasticity
    : public material_base<linear_elasticity<Traits>, Traits> {
public:
  using base = material_base<linear_elasticity<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using base::Dim;
  using tensor2 = tmech::tensor<value_type, Dim, 2>;
  using tensor4 = tmech::tensor<value_type, Dim, 4>;


  /**
   * @brief Constructs a linear elasticity instance with the given arguments.
   *
   * This constructor initializes reaction parameters, registers properties
   * and functions, and queries required external properties and functions.
   *
   * @tparam Args Variadic template for forwarding constructor arguments.
   * @param args Arguments to initialize the base class.
   */
  template <typename... Args>
  explicit linear_elasticity(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_sig(base::template add_output<tensor2>("stress")),
        m_C(base::template add_output<tensor4>("tangent")),
        m_K(base::template get_parameter<value_type>("K")),
        m_G(base::template get_parameter<value_type>("G")),
        m_eps_name(base::template get_parameter<std::string>("strain_producer_name")),
        m_eps(base::template add_input<tensor2>(m_eps_name, base::template get_parameter<std::string>("strain_property"), EdgeKind::Global)),
        m_eigenstrain(eigenstrain_input())
  {
    update_tangent(); // readable before the first update
    base::add_callback(&linear_elasticity::update_tangent, m_C).reads();
    base::add_callback(&linear_elasticity::update_stress, m_sig)
        .reads(m_C, m_eps, m_eigenstrain)
        .partials(m_sig, m_C, m_eps, m_eigenstrain);
    m_dsig_deps = base::partial_of(m_sig, m_eps);
    m_dsig_dC = base::partial_of(m_sig, m_C);
    if (m_eigenstrain)
      m_dsig_deig = base::partial_of(m_sig, *m_eigenstrain);
  }

  /**
   * @brief Defines the parameters required by this class.
   *
   * This function specifies the expected parameters, marking some as required
   * and providing default values for others.
   *
   * @return An input parameter controller defining the parameter structure.
   */
  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<value_type>(para, "K");
    required<value_type>(para, "G");
    required<std::string>(para, "strain_producer_name");
    optional<std::string>(para, "strain_property", std::string{"strain"});
    optional<std::string>(para, "eigenstrain_source", std::string{});
    return para;
  }

  /// sigma = C : (eps - eps_eig) with the optional eigenstrain (thermal, chemical).
  void update_stress(){
    if (m_eigenstrain)
      m_sig = tmech::dcontract(m_C, tensor2{m_eps.get() - m_eigenstrain->get()});
    else
      m_sig = tmech::dcontract(m_C, m_eps.get());
    if (base::want_partials())
      update_partials();
  }

  /// d sigma / d eps = C, d sigma / d eps_eig = -C, d sigma / d C = Isym (x) (eps - eps_eig).
  void update_partials() {
    if (m_dsig_deps.wanted())
      m_dsig_deps.value() = m_C;
    if (m_eigenstrain && m_dsig_deig.wanted())
      m_dsig_deig.value() = value_type{-1} * m_C;
    if (!m_dsig_dC.wanted())
      return;
    if (m_eigenstrain)
      m_dsig_dC.value() = tmech::otimes(sym_identity<value_type, Dim>(), tensor2{m_eps.get() - m_eigenstrain->get()});
    else
      m_dsig_dC.value() = tmech::otimes(sym_identity<value_type, Dim>(), m_eps.get());
  }

  void update_tangent(){
    m_C = 3*m_K*vol_projector<value_type, Dim>() + 2*m_G*dev_projector<value_type, Dim>();
  }

private:
  const input_property<tensor2, property_traits>* eigenstrain_input() {
    auto const &source{base::template get_parameter<std::string>("eigenstrain_source")};
    return source.empty() ? nullptr : &base::template add_input<tensor2>(source, "eigenstrain", EdgeKind::Global);
  }

  /// Produced properties
  tensor2 &m_sig;
  tensor4 &m_C;

  /// Parameters
  const value_type &m_K;
  const value_type &m_G;
  const std::string &m_eps_name;

  /// Consumed properties
  const input_property<tensor2, property_traits> &m_eps;
  const input_property<tensor2, property_traits>* m_eigenstrain; ///< null without an eigenstrain_source

  typename base::template partial_handle<tensor2, tensor2> m_dsig_deps, m_dsig_deig; ///< m_dsig_deig: with an eigenstrain
  typename base::template partial_handle<tensor2, tensor4> m_dsig_dC;
};

}
#endif // LINEAR_ELASTICITY_H
