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

// --- Sequence deduction ---

template<typename Output, typename Input>
struct diff_sequence { using type = void; };

template<typename T, std::size_t Dim>
struct diff_sequence<tmech::tensor<T,Dim,2>, tmech::tensor<T,Dim,2>> {
  using type = tmech::sequence<1,2,3,4>;
};

// scalar wrt scalar, scalar wrt tensor, tensor wrt scalar: no sequence needed (void)

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

// --- Frobenius norm helper ---

namespace detail {
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
}

// --- Numerical differentiation checker material ---

/// Checks an analytical derivative against numerical central differences.
///
/// Template parameters:
///   Traits  — material policy
///   Output  — type of the function output (double, tensor2, ...)
///   Input   — type of the function input to perturb (double, tensor2, ...)
///
/// Parameters:
///   "context"           — material_context<Traits>* (passed via parameter_handler)
///   "output_source"     — "material::property" of f(x)
///   "input_source"      — "material::property" of x
///   "analytical_source" — "material::property" of analytical df/dx
///   "history_sources"   — vector<string> of "material::property" histories to revert
///   "epsilon"           — perturbation size (default 1e-7)
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
  using sequence_type = typename diff_sequence<Output, Input>::type;

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
        m_history_strs(base::template get_parameter<std::vector<std::string>>("history_sources"))
  {
    // Register inputs so the topo sort places the checker after the checked material
    base::template add_input<result_type>(
        m_analytical_src.material, m_analytical_src.property, EdgeKind::Global);
    base::template add_input<Output>(
        m_output_src.material, m_output_src.property, EdgeKind::Global);

    // Parse history sources
    for (auto& s : m_history_strs)
      m_history_srcs.push_back(connection_source::parse(s));
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    // "context" is NOT declared here — it's a pointer type that can't go through
    // the input_parameters variant. It's read directly from parameter_handler.
    para.template insert<std::string>("output_source").template add<is_required>();
    para.template insert<std::string>("input_source").template add<is_required>();
    para.template insert<std::string>("analytical_source").template add<is_required>();
    para.template insert<std::vector<std::string>>("history_sources")
        .template add<set_default>(std::vector<std::string>{});
    para.template insert<value_type>("epsilon")
        .template add<set_default>(value_type{1e-7});
    return para;
  }

  void compute() {
    // 1. Copy input value
    auto& input_ref = m_ctx->template get_mutable<Input>(
        m_input_src.material, m_input_src.property);
    const Input input_saved{input_ref};

    // 2. Collect history property pointers and save their state
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
      for (auto* hp : hist_props) hp->revert();  // new = old
      m_ctx->update_property(m_output_src.material, m_output_src.property);
      return m_ctx->template get<Output>(m_output_src.material, m_output_src.property);
    };

    // 5. Numerical differentiation
    result_type numerical;
    if constexpr (std::is_same_v<sequence_type, void>) {
      numerical = tmech::num_diff_central(func, input_saved, m_epsilon);
    } else {
      numerical = tmech::num_diff_central<sequence_type>(func, input_saved, m_epsilon);
    }

    // 6. Restore input
    input_ref = input_saved;

    // 7. Restore histories by re-evaluating with original input
    for (auto* hp : hist_props) hp->revert();
    m_ctx->update_property(m_output_src.material, m_output_src.property);

    // 8. Compute error
    const result_type diff{analytical - numerical};
    m_error = detail::frobenius_norm<value_type>(diff);
    auto norm = detail::frobenius_norm<value_type>(analytical);
    m_rel_error = (norm > value_type{1e-30}) ? m_error / norm : m_error;
  }

private:
  value_type& m_error;
  value_type& m_rel_error;
  context_type* m_ctx;
  connection_source m_output_src;
  connection_source m_input_src;
  connection_source m_analytical_src;
  const value_type& m_epsilon;
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
