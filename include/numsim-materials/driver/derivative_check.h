#ifndef NUMSIM_MATERIALS_DRIVER_DERIVATIVE_CHECK_H
#define NUMSIM_MATERIALS_DRIVER_DERIVATIVE_CHECK_H

#include <cmath>
#include <exception>
#include <print>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <tmech/tmech.h>
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
  // Result storage: num_diff_sym_central builds tuples as (k,l,i,j) where
  // (k,l) are input indices and (i,j) are output indices. Sequences <3,4,1,2>
  // remap to result(i,j,k,l) = C_ijkl. Minor symmetries applied via all 4 permutations.
  using sym_result = std::tuple<tmech::sequence<3,4,1,2>, tmech::sequence<4,3,1,2>,
                                tmech::sequence<3,4,2,1>, tmech::sequence<4,3,2,1>>;
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

/// Central differences of f at x; symmetric perturbations for a symmetric tensor input.
template<typename Output, typename Input, typename F, typename Value>
typename diff_result<Output, Input>::type central_difference(F& f, const Input& x, Value h) {
  using traits_info = diff_traits<Output, Input>;
  if constexpr (!std::is_same_v<typename traits_info::sym_direction, void>)
    return tmech::num_diff_sym_central<typename traits_info::sym_direction, typename traits_info::sym_result>(f, x, h);
  else if constexpr (std::is_same_v<typename traits_info::sequence_type, void>)
    return tmech::num_diff_central(f, x, h);
  else
    return tmech::num_diff_central<typename traits_info::sequence_type>(f, x, h);
}

} // namespace detail

/// A numerical derivative next to the analytical one it checks.
template<typename Result, typename Value>
struct derivative_check {
  Result numerical;
  Result analytical;
  Value error;     ///< norm(analytical - numerical)
  Value rel_error; ///< error / norm(analytical), or error when that norm vanishes
};

/// d(output)/d(input) by central differences along the host path: per evaluation revert(), set_input(x'),
/// update(). Evaluated at x first (analytical) and last; on a throw the context is reverted with x set.
template<typename Output, typename Traits, typename Input, typename SetInput>
derivative_check<typename diff_result<Output, Input>::type, typename Traits::value_type>
check_derivative(material_context<Traits>& ctx, const std::string& output, const std::string& analytical,
                 const Input& x, SetInput&& set_input, typename Traits::value_type h = 1e-7) {
  using result_type = typename diff_result<Output, Input>::type;
  auto const out{connection_source::parse(output)};
  auto const ana{connection_source::parse(analytical)};
  Input const x0{x}; // x may be the bound input itself
  auto evaluate{[&](const Input& trial) -> Output {
    ctx.revert();
    set_input(trial);
    ctx.update();
    return ctx.template get<Output>(out.material, out.property);
  }};
  try {
    evaluate(x0);
    result_type const derivative{ctx.template get<result_type>(ana.material, ana.property)};
    // tmech's differentiation is noexcept: a failure is kept and rethrown after it
    std::exception_ptr failure;
    auto perturbed{[&](const Input& trial) -> Output {
      if (!failure)
        try {
          return evaluate(trial);
        } catch (...) {
          failure = std::current_exception();
        }
      return Output{};
    }};
    result_type const numerical{detail::central_difference<Output>(perturbed, x0, h)};
    if (failure)
      std::rethrow_exception(failure);
    evaluate(x0);
    auto const error{detail::compute_norm(result_type{derivative - numerical})};
    auto const norm{detail::compute_norm(derivative)};
    return {numerical, derivative, error, norm > 1e-30 ? error / norm : error};
  } catch (...) {
    ctx.revert();
    set_input(x0);
    throw;
  }
}

/// Prints the components whose relative error exceeds `threshold`.
template<typename Result, typename Value>
void report_mismatches(std::string_view name, const derivative_check<Result, Value>& check, Value threshold) {
  if (check.rel_error <= threshold)
    return;
  std::println("  {}: mismatching components (rel > {:.2e}):", name, threshold);
  detail::report_mismatches(check.analytical, check.numerical, threshold);
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DRIVER_DERIVATIVE_CHECK_H
