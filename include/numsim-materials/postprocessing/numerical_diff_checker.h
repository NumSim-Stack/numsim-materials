#ifndef NUMSIM_MATERIALS_NUMERICAL_DIFF_CHECKER_H
#define NUMSIM_MATERIALS_NUMERICAL_DIFF_CHECKER_H

#include <cmath>
#include <print>
#include <string>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/material_context.h"

namespace numsim::materials {

// --- Sequence and symmetry type deduction ---

template<typename Output, typename Input>
struct diff_traits {
  using sequence_type = void;
  using sym_direction = void;
  using sym_result = std::tuple<>;
};

template<typename T, std::size_t Dim>
struct diff_traits<tmech::tensor<T,Dim,2>, tmech::tensor<T,Dim,2>> {
  using sequence_type = tmech::sequence<1,2,3,4>;
  using sym_direction = std::tuple<tmech::sequence<1,2>, tmech::sequence<2,1>>;
  using sym_result = std::tuple<>;
};

// --- Result type deduction ---

template<typename Output, typename Input>
struct diff_result;

template<>
struct diff_result<double, double> { using type = double; };

template<typename T, std::size_t Dim>
struct diff_result<tmech::tensor<T,Dim,2>, tmech::tensor<T,Dim,2>> {
  using type = tmech::tensor<T,Dim,4>;
};

template<typename T, std::size_t Dim>
struct diff_result<double, tmech::tensor<T,Dim,2>> {
  using type = tmech::tensor<T,Dim,2>;
};

template<typename T, std::size_t Dim>
struct diff_result<tmech::tensor<T,Dim,2>, double> {
  using type = tmech::tensor<T,Dim,2>;
};

// --- Component-wise error reporting ---

namespace detail {

template<typename T>
void report_mismatches(double analytical, double numerical, T tol) {
  auto diff = std::abs(analytical - numerical);
  auto ref = std::max(std::abs(analytical), std::abs(numerical));
  auto rel = (ref > T{1e-30}) ? diff / ref : diff;
  if (rel > tol)
    std::println("    scalar: analytical={:.6e} numerical={:.6e} diff={:.6e} rel={:.6e}",
                 analytical, numerical, diff, rel);
}

template<typename T, std::size_t Dim>
void report_mismatches(const tmech::tensor<T,Dim,2>& analytical,
                       const tmech::tensor<T,Dim,2>& numerical, T tol) {
  for (std::size_t i = 0; i < Dim; ++i)
    for (std::size_t j = 0; j < Dim; ++j) {
      auto diff = std::abs(analytical(i,j) - numerical(i,j));
      auto ref = std::max(std::abs(analytical(i,j)), std::abs(numerical(i,j)));
      auto rel = (ref > T{1e-30}) ? diff / ref : diff;
      if (rel > tol)
        std::println("    ({},{}) analytical={:.6e} numerical={:.6e} rel={:.6e}",
                     i, j, analytical(i,j), numerical(i,j), rel);
    }
}

template<typename T, std::size_t Dim>
void report_mismatches(const tmech::tensor<T,Dim,4>& analytical,
                       const tmech::tensor<T,Dim,4>& numerical, T tol) {
  for (std::size_t i = 0; i < Dim; ++i)
    for (std::size_t j = 0; j < Dim; ++j)
      for (std::size_t k = 0; k < Dim; ++k)
        for (std::size_t l = 0; l < Dim; ++l) {
          auto diff = std::abs(analytical(i,j,k,l) - numerical(i,j,k,l));
          auto ref = std::max(std::abs(analytical(i,j,k,l)), std::abs(numerical(i,j,k,l)));
          auto rel = (ref > T{1e-30}) ? diff / ref : diff;
          if (rel > tol)
            std::println("    ({},{},{},{}) analytical={:.6e} numerical={:.6e} rel={:.6e}",
                         i, j, k, l, analytical(i,j,k,l), numerical(i,j,k,l), rel);
        }
}

template<typename T>
T frobenius_norm(double a) { return std::abs(a); }

template<typename T, std::size_t Dim, std::size_t Rank>
T frobenius_norm(const tmech::tensor<T,Dim,Rank>& t) {
  T sum{0};
  auto kernel = [&](auto... idx) { sum += t(idx...) * t(idx...); };
  using loop = typename tmech::detail::meta_for_loop_deep<Dim, Rank-1>::type;
  loop::loop(kernel);
  return std::sqrt(sum);
}

} // namespace detail

// --- Numerical differentiation checker material ---

/// Checks an analytical derivative against numerical central differences.
///
/// Template parameters:
///   Traits  — material policy
///   Output  — type of the function output (double, tensor2, ...)
///   Input   — type of the function input to perturb (double, tensor2, ...)
///
/// Parameters:
///   "context"           — material_context<Traits>* (via parameter_handler)
///   "output_source"     — "material::property" of f(x)
///   "input_source"      — "material::property" of x
///   "analytical_source" — "material::property" of analytical df/dx
///   "history_sources"   — vector<string> of histories to revert
///   "epsilon"           — perturbation size (default 1e-7)
///   "use_symmetric"     — int: 1 to use symmetric kernel (default 0)
///   "report_threshold"  — value_type: report components with rel error above this (default 0.01)
///
/// Outputs:
///   "error"     — scalar: Frobenius norm of (analytical - numerical)
///   "rel_error" — scalar: relative error
template<typename Traits, typename Output, typename Input>
class numerical_diff_checker final
    : public material_base<numerical_diff_checker<Traits, Output, Input>, Traits> {
public:
  using base = material_base<numerical_diff_checker<Traits, Output, Input>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using context_type = material_context<Traits>;
  using result_type = typename diff_result<Output, Input>::type;
  using traits_info = diff_traits<Output, Input>;

  template<typename... Args>
  explicit numerical_diff_checker(Args&&... args)
      : base(std::forward<Args>(args)...),
        m_error(base::template add_output<value_type>("error", &numerical_diff_checker::compute)),
        m_rel_error(base::template add_output<value_type>("rel_error")),
        m_ctx(base::template get_parameter<context_type*>("context")),
        m_output_src(connection_source::parse(
            base::template get_parameter<std::string>("output_source"))),
        m_input_src(connection_source::parse(
            base::template get_parameter<std::string>("input_source"))),
        m_analytical_src(connection_source::parse(
            base::template get_parameter<std::string>("analytical_source"))),
        m_epsilon(base::template get_parameter<value_type>("epsilon")),
        m_report_threshold(base::template get_parameter<value_type>("report_threshold")),
        m_history_strs(base::template get_parameter<std::vector<std::string>>("history_sources"))
  {
    base::template add_input<result_type>(
        m_analytical_src.material, m_analytical_src.property, EdgeKind::Global);
    base::template add_input<Output>(
        m_output_src.material, m_output_src.property, EdgeKind::Global);

    for (auto& s : m_history_strs)
      m_history_srcs.push_back(connection_source::parse(s));
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    para.template insert<std::string>("output_source").template add<is_required>();
    para.template insert<std::string>("input_source").template add<is_required>();
    para.template insert<std::string>("analytical_source").template add<is_required>();
    para.template insert<std::vector<std::string>>("history_sources")
        .template add<set_default>(std::vector<std::string>{});
    para.template insert<value_type>("epsilon")
        .template add<set_default>(value_type{1e-7});
    para.template insert<value_type>("report_threshold")
        .template add<set_default>(value_type{0.01});
    return para;
  }

  void compute() {
    // 1. Copy input
    auto& input_ref = m_ctx->template get_mutable<Input>(
        m_input_src.material, m_input_src.property);
    const Input input_saved{input_ref};

    // 2. Collect history properties
    std::vector<property_base*> hist_props;
    for (auto& src : m_history_srcs) {
      auto* p = m_ctx->find_property(src.material, src.property);
      if (p) hist_props.push_back(p);
    }

    // 3. Read analytical derivative
    const result_type analytical{m_ctx->template get<result_type>(
        m_analytical_src.material, m_analytical_src.property)};

    // 4. Build perturbation lambda
    auto func = [&](const Input& trial) -> Output {
      input_ref = trial;
      for (auto* hp : hist_props) hp->revert();
      m_ctx->update_property(m_output_src.material, m_output_src.property);
      return m_ctx->template get<Output>(m_output_src.material, m_output_src.property);
    };

    // 5. Numerical differentiation
    result_type numerical = compute_standard(func, input_saved);

    // 6. Restore
    input_ref = input_saved;
    for (auto* hp : hist_props) hp->revert();
    m_ctx->update_property(m_output_src.material, m_output_src.property);

    // 7. Compute error
    const result_type diff{analytical - numerical};
    m_error = detail::frobenius_norm<value_type>(diff);
    auto norm = detail::frobenius_norm<value_type>(analytical);
    m_rel_error = (norm > value_type{1e-30}) ? m_error / norm : m_error;

    // 8. Report mismatching components
    if (m_rel_error > m_report_threshold) {
      std::println("  numerical_diff_checker '{}': mismatching components (rel > {:.2e}):",
                   base::name(), m_report_threshold);
      detail::report_mismatches(analytical, numerical, m_report_threshold);
    }
  }

private:
  // Standard (non-symmetric) numerical differentiation
  template<typename Func>
  result_type compute_standard(Func& func, const Input& x) {
    using seq = typename traits_info::sequence_type;
    if constexpr (std::is_same_v<seq, void>) {
      return tmech::num_diff_central(func, x, m_epsilon);
    } else {
      return tmech::num_diff_central<seq>(func, x, m_epsilon);
    }
  }

  value_type& m_error;
  value_type& m_rel_error;
  context_type* m_ctx;
  connection_source m_output_src;
  connection_source m_input_src;
  connection_source m_analytical_src;
  const value_type& m_epsilon;
  const value_type& m_report_threshold;
  const std::vector<std::string>& m_history_strs;
  std::vector<connection_source> m_history_srcs;
};

// --- Convenience aliases ---

template<typename Traits>
using tangent_checker = numerical_diff_checker<Traits,
    tmech::tensor<typename Traits::value_type, Traits::Dim, 2>,
    tmech::tensor<typename Traits::value_type, Traits::Dim, 2>>;

template<typename Traits>
using gradient_checker = numerical_diff_checker<Traits,
    typename Traits::value_type,
    tmech::tensor<typename Traits::value_type, Traits::Dim, 2>>;

template<typename Traits>
using scalar_derivative_checker = numerical_diff_checker<Traits,
    typename Traits::value_type,
    typename Traits::value_type>;

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_NUMERICAL_DIFF_CHECKER_H
