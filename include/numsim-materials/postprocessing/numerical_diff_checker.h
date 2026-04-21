#ifndef NUMSIM_MATERIALS_NUMERICAL_DIFF_CHECKER_H
#define NUMSIM_MATERIALS_NUMERICAL_DIFF_CHECKER_H

#include <cassert>
#include <cmath>
#include <print>
#include <string>
#include <type_traits>
#include <unordered_set>
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
  // Symmetry of input (strain): ε_ij = ε_ji
  using sym_direction = std::tuple<tmech::sequence<1,2>, tmech::sequence<2,1>>;
  // Result symmetry: both minor symmetries C_ijkl = C_jikl = C_ijlk = C_jilk
  using sym_result = std::tuple<tmech::sequence<1,2,3,4>, tmech::sequence<2,1,3,4>,
                                tmech::sequence<1,2,4,3>, tmech::sequence<2,1,4,3>>;
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

// --- Helpers ---

namespace detail {

template<typename T>
auto compute_norm(const T& x) {
  if constexpr (std::is_arithmetic_v<T>) return std::abs(x);
  else return tmech::norm(x);
}

// --- Component-wise error reporting ---

template<typename T>
void report_mismatches(T analytical, T numerical, T tol) {
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

} // namespace detail

// --- Numerical differentiation checker material ---

/// Diagnostic material that checks an analytical derivative against numerical
/// central differences.
///
/// This is a privileged material: it requires a material_context pointer to
/// perturb inputs and re-evaluate the property graph. Normal materials
/// communicate only through the graph; this one operates ON the graph.
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
    base::template add_input<Input>(
        m_input_src.material, m_input_src.property, EdgeKind::Global);
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
    assert(m_ctx->is_finalized() && "numerical_diff_checker requires finalized context");

    // 1. Cache property pointers (once, after finalize)
    if (!m_cached) {
      m_input_prop = m_ctx->find_property(m_input_src.material, m_input_src.property);
      if (!m_input_prop)
        throw std::runtime_error(
            "numerical_diff_checker '" + std::string(base::name()) +
            "': input source '" + m_input_src.material + "::" + m_input_src.property + "' not found");
      for (auto& src : m_history_srcs) {
        auto* p = m_ctx->find_property(src.material, src.property);
        if (!p)
          throw std::runtime_error(
              "numerical_diff_checker '" + std::string(base::name()) +
              "': history source '" + src.material + "::" + src.property + "' not found");
        m_hist_props.push_back(p);
      }
      m_exclude = {m_input_prop};
      m_cached = true;
    }

    // 2. Copy input
    auto& input_ref = m_ctx->template get_mutable<Input>(
        m_input_src.material, m_input_src.property);
    const Input input_saved{input_ref};

    // 3. Read analytical derivative
    const result_type analytical{m_ctx->template get<result_type>(
        m_analytical_src.material, m_analytical_src.property)};

    // 4. Build perturbation lambda — uses update_property with exclusion
    //    to skip the input producer's callback (e.g., a stepper)
    auto func = [&](const Input& trial) -> Output {
      input_ref = trial;
      for (auto* hp : m_hist_props) hp->revert();
      m_ctx->update_property(m_output_src.material, m_output_src.property, m_exclude);
      return m_ctx->template get<Output>(m_output_src.material, m_output_src.property);
    };

    // 5. Numerical differentiation (symmetric for tensor inputs)
    //    Wrapped in try/catch to guarantee state restoration on failure.
    result_type numerical;
    try {
      if constexpr (requires_symmetric_diff()) {
        using sym_dir = typename traits_info::sym_direction;
        using sym_res = typename traits_info::sym_result;
        numerical = tmech::num_diff_sym_central<sym_dir, sym_res>(func, input_saved, m_epsilon);
      } else {
        numerical = compute_standard(func, input_saved);
      }
    } catch (...) {
      restore_state(input_ref, input_saved);
      throw;
    }

    // 6. Restore graph to pre-perturbation state
    restore_state(input_ref, input_saved);

    // 7. Compute error
    const result_type diff{analytical - numerical};
    m_error = detail::compute_norm(diff);
    auto norm = detail::compute_norm(analytical);
    m_rel_error = (norm > value_type{1e-30}) ? m_error / norm : m_error;

    // 8. Report mismatching components
    if (m_rel_error > m_report_threshold) {
      std::println("  numerical_diff_checker '{}': mismatching components (rel > {:.2e}):",
                   base::name(), m_report_threshold);
      detail::report_mismatches(analytical, numerical, m_report_threshold);
    }
  }

private:
  static constexpr bool requires_symmetric_diff() {
    return !std::is_same_v<typename traits_info::sym_direction, void>;
  }

  template<typename Func>
  result_type compute_standard(Func& func, const Input& x) {
    using seq = typename traits_info::sequence_type;
    if constexpr (std::is_same_v<seq, void>) {
      return tmech::num_diff_central(func, x, m_epsilon);
    } else {
      return tmech::num_diff_central<seq>(func, x, m_epsilon);
    }
  }

  void restore_state(Input& input_ref, const Input& input_saved) {
    input_ref = input_saved;
    for (auto* hp : m_hist_props) hp->revert();
    m_ctx->update_property(m_output_src.material, m_output_src.property, m_exclude);
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
  const property_base* m_input_prop{nullptr};
  std::unordered_set<const property_base*> m_exclude;
  std::vector<property_base*> m_hist_props;
  bool m_cached{false};
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
