#ifndef NUMSIM_MATERIALS_BENCHMARK_J2_RESIDUAL_SYSTEM_H
#define NUMSIM_MATERIALS_BENCHMARK_J2_RESIDUAL_SYSTEM_H

// J2 with linear isotropic hardening as a residual system of the graph, for the benchmarks only.
// j2_flow + j2_stress: the system node owns dlambda, a flow node computes the yield residual and the plastic
// strain, a stress node reads the plastic strain. j2_return: one node writes the stress and the residual from
// dlambda (sigma = sigma_trial - 2 G dlambda N), optionally reading an eigenstrain.

#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/tensor_identities.h"
#include "numsim-materials/driver/external_state_source.h"
#include "numsim-materials/materials/constant_scalar.h"
#include "numsim-materials/materials/linear_elasticity.h"
#include "numsim-materials/materials/plasticity_utils.h"
#include "numsim-materials/materials/thermal_eigenstrain.h"
#include "numsim-materials/solvers/system.h"
#include <tmech/tmech.h>

#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace numsim::materials::benchmark {

/// Yield residual r(dlambda) and the history it implies; an elastic trial state turns r into dlambda = 0.
template <typename Traits> class j2_flow final : public numsim::materials::material_base<j2_flow<Traits>, Traits> {
public:
  using base = numsim::materials::material_base<j2_flow<Traits>, Traits>;
  using T = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using t2 = tmech::tensor<T, 3, 2>;
  using t4 = tmech::tensor<T, 3, 4>;

  template <typename... Args>
  explicit j2_flow(Args &&...args)
      : base(std::forward<Args>(args)...), m_r(base::template add_output<T>("residual")),
        m_eps_p(base::template add_history_output<t2>("plastic_strain")),
        m_kappa(base::template add_history_output<T>("equivalent_plastic_strain")),
        m_K(base::template get_parameter<T>("K")), m_G(base::template get_parameter<T>("G")),
        m_sigma_0(base::template get_parameter<T>("sigma_0")), m_H(base::template get_parameter<T>("H")),
        m_strain(base::template add_input<t2>(base::template get_parameter<std::string>("strain_source"), "strain")),
        m_dl(base::template add_input<T>(base::template get_parameter<std::string>("system_source"), "dlambda")) {
    base::add_callback(&j2_flow::compute, m_r, m_eps_p, m_kappa)
        .reads(m_strain, m_dl)
        .partials(m_r, m_strain, m_dl)
        .partials(m_eps_p, m_strain, m_dl)
        .fills_partials(&j2_flow::fill_partials);
    m_dr_dl = base::partial_of(m_r, m_dl);
    m_dr_deps = base::partial_of(m_r, m_strain);
    m_dp_dl = base::partial_of(m_eps_p, m_dl);
    m_dp_deps = base::partial_of(m_eps_p, m_strain);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    numsim::materials::required<std::string>(para, "strain_source");
    numsim::materials::required<std::string>(para, "system_source");
    numsim::materials::required<T>(para, "K");
    numsim::materials::required<T>(para, "G");
    numsim::materials::required<T>(para, "sigma_0");
    numsim::materials::required<T>(para, "H");
    return para;
  }

  void compute() {
    T const dl{m_dl.get()};
    T const kappa_n{m_kappa.old_value()};
    t2 const s{T(2) * m_G * tmech::dev(m_strain.get() - m_eps_p.old_value())};
    m_q = std::sqrt(T(1.5) * tmech::dcontract(s, s));
    m_plastic = m_q - m_sigma_0 - m_H * kappa_n > T(0);
    m_kappa.new_value() = kappa_n + dl;
    if (!m_plastic) {
      m_r = dl;
      m_eps_p.new_value() = m_eps_p.old_value();
      return;
    }
    m_N = (T(1.5) / m_q) * s;
    m_r = m_q - T(3) * m_G * dl - m_sigma_0 - m_H * (kappa_n + dl);
    m_eps_p.new_value() = m_eps_p.old_value() + dl * m_N;
  }

  /// The requested partials at the state compute() left: Newton asks for d r / d dlambda only.
  void fill_partials() {
    if (m_dr_dl.wanted())
      m_dr_dl.value() = m_plastic ? -T(3) * m_G - m_H : T(1);
    if (m_dr_deps.wanted())
      m_dr_deps.value() = m_plastic ? t2{(T(2) * m_G) * m_N} : t2{};
    if (m_dp_dl.wanted())
      m_dp_dl.value() = m_plastic ? m_N : t2{};
    if (!m_dp_deps.wanted())
      return;
    if (m_plastic)
      m_dp_deps.value() =
          (T(2) * m_G * m_dl.get() / m_q) * (T(1.5) * numsim::materials::dev_projector<T, 3>() - tmech::otimes(m_N, m_N));
    else
      m_dp_deps.value() = t4{};
  }

private:
  T &m_r;
  numsim::materials::history_property<t2> &m_eps_p;
  numsim::materials::history_property<T> &m_kappa;
  T const &m_K, &m_G, &m_sigma_0, &m_H;
  numsim::materials::input_property<t2, numsim::materials::property_traits> const &m_strain;
  numsim::materials::input_property<T, numsim::materials::property_traits> const &m_dl;
  T m_q{};
  t2 m_N{};
  bool m_plastic{false};
  typename base::template partial_handle<T, T> m_dr_dl;
  typename base::template partial_handle<T, t2> m_dr_deps;
  typename base::template partial_handle<t2, T> m_dp_dl;
  typename base::template partial_handle<t2, t2> m_dp_deps;
};

/// sigma = C_e : (eps - plastic strain).
template <typename Traits> class j2_stress final : public numsim::materials::material_base<j2_stress<Traits>, Traits> {
public:
  using base = numsim::materials::material_base<j2_stress<Traits>, Traits>;
  using T = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using t2 = tmech::tensor<T, 3, 2>;
  using t4 = tmech::tensor<T, 3, 4>;

  template <typename... Args>
  explicit j2_stress(Args &&...args)
      : base(std::forward<Args>(args)...), m_stress(base::template add_output<t2>("stress")),
        m_strain(base::template add_input<t2>(base::template get_parameter<std::string>("strain_source"), "strain")),
        m_eps_p(base::template add_input<t2>(base::template get_parameter<std::string>("flow_source"), "plastic_strain")),
        m_C(numsim::materials::plasticity_detail::make_isotropic_tangent<T, 3>(base::template get_parameter<T>("K"),
                                                                                base::template get_parameter<T>("G"))) {
    base::add_callback(&j2_stress::compute, m_stress)
        .reads(m_strain, m_eps_p)
        .partials(m_stress, m_strain, m_eps_p)
        .fills_partials(&j2_stress::fill_partials);
    m_ds_deps = base::partial_of(m_stress, m_strain);
    m_ds_dp = base::partial_of(m_stress, m_eps_p);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    numsim::materials::required<std::string>(para, "strain_source");
    numsim::materials::required<std::string>(para, "flow_source");
    numsim::materials::required<T>(para, "K");
    numsim::materials::required<T>(para, "G");
    return para;
  }

  void compute() { m_stress = tmech::dcontract(m_C, t2{m_strain.get() - m_eps_p.get()}); }

  void fill_partials() {
    if (m_ds_deps.wanted())
      m_ds_deps.value() = m_C;
    if (m_ds_dp.wanted())
      m_ds_dp.value() = t4{-m_C};
  }

private:
  t2 &m_stress;
  numsim::materials::input_property<t2, numsim::materials::property_traits> const &m_strain;
  numsim::materials::input_property<t2, numsim::materials::property_traits> const &m_eps_p;
  t4 const m_C;
  typename base::template partial_handle<t2, t2> m_ds_deps;
  typename base::template partial_handle<t2, t2> m_ds_dp;
};

/// J2 radial return as the one member of a residual system: sigma = sigma_trial - 2G dlambda N and the yield
/// residual r(dlambda), with the partials of both. An optional eigenstrain enters as eps - eps_eig, so its
/// partials are those of the strain with the opposite sign.
template <typename Traits> class j2_return final : public numsim::materials::material_base<j2_return<Traits>, Traits> {
public:
  using base = numsim::materials::material_base<j2_return<Traits>, Traits>;
  using T = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using t2 = tmech::tensor<T, 3, 2>;
  using t4 = tmech::tensor<T, 3, 4>;
  using input_t2 = numsim::materials::input_property<t2, numsim::materials::property_traits>;

  template <typename... Args>
  explicit j2_return(Args &&...args)
      : base(std::forward<Args>(args)...), m_stress(base::template add_output<t2>("stress")),
        m_r(base::template add_output<T>("residual")), m_eps_p(base::template add_history_output<t2>("plastic_strain")),
        m_kappa(base::template add_history_output<T>("equivalent_plastic_strain")),
        m_K(base::template get_parameter<T>("K")), m_G(base::template get_parameter<T>("G")),
        m_sigma_0(base::template get_parameter<T>("sigma_0")), m_H(base::template get_parameter<T>("H")),
        m_strain(base::template add_input<t2>(base::template get_parameter<std::string>("strain_source"), "strain")),
        m_dl(base::template add_input<T>(base::template get_parameter<std::string>("system_source"), "dlambda")),
        m_eig(eigenstrain_input()),
        m_C(numsim::materials::plasticity_detail::make_isotropic_tangent<T, 3>(m_K, m_G)) {
    base::add_callback(&j2_return::compute, m_stress, m_r, m_eps_p, m_kappa)
        .reads(m_strain, m_dl, m_eig)
        .partials(m_stress, m_strain, m_dl, m_eig)
        .partials(m_r, m_strain, m_dl, m_eig)
        .fills_partials(&j2_return::fill_partials);
    m_ds_de = base::partial_of(m_stress, m_strain);
    m_ds_dl = base::partial_of(m_stress, m_dl);
    m_dr_de = base::partial_of(m_r, m_strain);
    m_dr_dl = base::partial_of(m_r, m_dl);
    if (m_eig) {
      m_ds_deig = base::partial_of(m_stress, *m_eig);
      m_dr_deig = base::partial_of(m_r, *m_eig);
    }
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    numsim::materials::required<std::string>(para, "strain_source");
    numsim::materials::required<std::string>(para, "system_source");
    numsim::materials::optional<std::string>(para, "eigenstrain_source", std::string{});
    numsim::materials::required<T>(para, "K");
    numsim::materials::required<T>(para, "G");
    numsim::materials::required<T>(para, "sigma_0");
    numsim::materials::required<T>(para, "H");
    return para;
  }

  void compute() {
    t2 const eps_e{m_eig ? t2{m_strain.get() - m_eig->get() - m_eps_p.old_value()} : t2{m_strain.get() - m_eps_p.old_value()}};
    T const dl{m_dl.get()};
    T const kappa_n{m_kappa.old_value()};
    t2 const s{T(2) * m_G * tmech::dev(eps_e)};
    m_q = std::sqrt(T(1.5) * tmech::dcontract(s, s));
    m_plastic = m_q - m_sigma_0 - m_H * kappa_n > T(0);
    m_kappa.new_value() = kappa_n + dl;
    t2 const trial{s + (m_K * tmech::trace(eps_e)) * tmech::eye<T, 3, 2>()};
    if (!m_plastic) {
      m_r = dl;
      m_eps_p.new_value() = m_eps_p.old_value();
      m_stress = trial;
      return;
    }
    m_N = (T(1.5) / m_q) * s;
    m_r = m_q - T(3) * m_G * dl - m_sigma_0 - m_H * (kappa_n + dl);
    m_eps_p.new_value() = m_eps_p.old_value() + dl * m_N;
    m_stress = trial - (T(2) * m_G * dl) * m_N;
  }

  /// The partials at the state compute() left; Newton asks for d r / d dlambda only.
  void fill_partials() {
    if (m_dr_dl.wanted())
      m_dr_dl.value() = m_plastic ? -T(3) * m_G - m_H : T(1);
    if (m_ds_dl.wanted())
      m_ds_dl.value() = m_plastic ? t2{(-T(2) * m_G) * m_N} : t2{};
    t2 const dr_de{m_plastic ? t2{(T(2) * m_G) * m_N} : t2{}};
    if (m_dr_de.wanted())
      m_dr_de.value() = dr_de;
    if (m_eig && m_dr_deig.wanted())
      m_dr_deig.value() = -dr_de;
    bool const de{m_ds_de.wanted()}, deig{m_eig && m_ds_deig.wanted()};
    if (!de && !deig)
      return;
    // d sigma / d eps at fixed dlambda: C - 2G dlambda dN/deps
    T const a{m_plastic ? T(6) * m_G * m_G * m_dl.get() / m_q : T(0)};
    t4 const ds{m_plastic ? t4{m_C - a * numsim::materials::dev_projector<T, 3>() + (T(2) * a / T(3)) * tmech::otimes(m_N, m_N)} : m_C};
    if (de)
      m_ds_de.value() = ds;
    if (deig)
      m_ds_deig.value() = -ds;
  }

private:
  input_t2 const *eigenstrain_input() {
    auto const &source{base::template get_parameter<std::string>("eigenstrain_source")};
    return source.empty() ? nullptr : &base::template add_input<t2>(source, "eigenstrain");
  }

  t2 &m_stress;
  T &m_r;
  numsim::materials::history_property<t2> &m_eps_p;
  numsim::materials::history_property<T> &m_kappa;
  T const &m_K, &m_G, &m_sigma_0, &m_H;
  input_t2 const &m_strain;
  numsim::materials::input_property<T, numsim::materials::property_traits> const &m_dl;
  input_t2 const *m_eig;
  t4 const m_C;
  T m_q{};
  t2 m_N{};
  bool m_plastic{false};
  typename base::template partial_handle<t2, t2> m_ds_de, m_ds_deig;
  typename base::template partial_handle<t2, T> m_ds_dl;
  typename base::template partial_handle<T, t2> m_dr_de, m_dr_deig;
  typename base::template partial_handle<T, T> m_dr_dl;
};

using system_policy = numsim::materials::material_policy_default;
using system_context = numsim::materials::material_context<system_policy>;

/// "strain" -> system "plastic" (dlambda) <-> "flow" -> stress "j2".
inline std::function<void(system_context &)> j2_system_build(double K, double G, double sigma_0, double H) {
  return [=](system_context &ctx) {
    namespace nm = numsim::materials;
    system_policy::ParameterHandler p;
    p.insert<std::string>("name", "strain");
    ctx.create<nm::external_strain_source<system_policy>>(p);
    p.clear();
    p.insert<std::string>("name", "plastic");
    p.insert<double>("tolerance", 1e-8);
    p.insert<std::vector<nm::system_unknown>>("unknowns", {{"dlambda", nm::unknown_kind::scalar}});
    p.insert<std::vector<nm::system_equation>>("equations", {{"flow::residual", "dlambda"}});
    ctx.create<nm::residual_system<system_policy>>(p);
    p.clear();
    p.insert<std::string>("name", "flow");
    p.insert<std::string>("strain_source", "strain");
    p.insert<std::string>("system_source", "plastic");
    p.insert<double>("K", K);
    p.insert<double>("G", G);
    p.insert<double>("sigma_0", sigma_0);
    p.insert<double>("H", H);
    ctx.create<j2_flow<system_policy>>(p);
    p.clear();
    p.insert<std::string>("name", "j2");
    p.insert<std::string>("strain_source", "strain");
    p.insert<std::string>("flow_source", "flow");
    p.insert<double>("K", K);
    p.insert<double>("G", G);
    ctx.create<j2_stress<system_policy>>(p);
  };
}

/// "strain" -> system "plastic" (dlambda) <-> return "j2" (stress, residual); with alpha != 0 also
/// "temperature" -> "alpha" -> thermal_eigenstrain "thermal" -> j2.
inline std::function<void(system_context &)> j2_return_build(double K, double G, double sigma_0, double H, double alpha = 0,
                                                              double T_ref = 0) {
  return [=](system_context &ctx) {
    namespace nm = numsim::materials;
    system_policy::ParameterHandler p;
    p.insert<std::string>("name", "strain");
    ctx.create<nm::external_strain_source<system_policy>>(p);
    if (alpha != 0) {
      p.clear();
      p.insert<std::string>("name", "temperature");
      ctx.create<nm::external_scalar_source<system_policy>>(p);
      p.clear();
      p.insert<std::string>("name", "alpha");
      p.insert<double>("value", alpha);
      ctx.create<nm::constant_scalar<system_policy>>(p);
      p.clear();
      p.insert<std::string>("name", "thermal");
      p.insert<std::string>("alpha_source", "alpha::value");
      p.insert<double>("reference_temperature", T_ref);
      ctx.create<nm::thermal_eigenstrain<system_policy>>(p);
    }
    p.clear();
    p.insert<std::string>("name", "plastic");
    p.insert<double>("tolerance", 1e-8);
    p.insert<std::vector<nm::system_unknown>>("unknowns", {{"dlambda", nm::unknown_kind::scalar}});
    p.insert<std::vector<nm::system_equation>>("equations", {{"j2::residual", "dlambda"}});
    ctx.create<nm::residual_system<system_policy>>(p);
    p.clear();
    p.insert<std::string>("name", "j2");
    p.insert<std::string>("strain_source", "strain");
    p.insert<std::string>("system_source", "plastic");
    if (alpha != 0)
      p.insert<std::string>("eigenstrain_source", "thermal");
    p.insert<double>("K", K);
    p.insert<double>("G", G);
    p.insert<double>("sigma_0", sigma_0);
    p.insert<double>("H", H);
    ctx.create<j2_return<system_policy>>(p);
  };
}

/// "strain" -> linear_elasticity "elastic".
inline std::function<void(system_context &)> linear_elastic_build(double K, double G) {
  return [=](system_context &ctx) {
    namespace nm = numsim::materials;
    system_policy::ParameterHandler p;
    p.insert<std::string>("name", "strain");
    ctx.create<nm::external_strain_source<system_policy>>(p);
    p.clear();
    p.insert<std::string>("name", "elastic");
    p.insert<std::string>("strain_producer_name", "strain");
    p.insert<std::string>("strain_property", "strain");
    p.insert<double>("K", K);
    p.insert<double>("G", G);
    ctx.create<nm::linear_elasticity<system_policy>>(p);
  };
}

} // namespace numsim::materials::benchmark

#endif // NUMSIM_MATERIALS_BENCHMARK_J2_RESIDUAL_SYSTEM_H
