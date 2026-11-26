#ifndef NUMSIM_MATERIALS_CORE_DERIVATIVE_CHECK_H
#define NUMSIM_MATERIALS_CORE_DERIVATIVE_CHECK_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <print>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/history_property.h"
#include "numsim-materials/core/input_types.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/property.h"
#include "numsim-materials/core/property_traits.h"

namespace numsim::materials {

/// d(output)/d(input) along one direction of the input, as the output's components (row-major).
template <class T>
struct derivative_direction {
  std::string label; ///< the input component: "x", "(i)" or "(i,j)"
  std::vector<T> numerical;
  std::vector<T> analytical;
};

/// What material_context::check_derivative() found.
template <class T>
struct derivative_check {
  std::vector<derivative_direction<T>> directions;
  T error{0};     ///< over all directions: the norm of analytical - numerical
  T rel_error{0}; ///< error / the norm of the analytical, or error when that vanishes
  T asymmetry{0}; ///< a symmetric input: the norm of the analytical's A_ij - A_ji (i < j), part of error
};

/// The parameters of material_context::check_derivative().
template <class Traits>
typename Traits::InputParameterController derivative_check_parameters() {
  using value_type = typename Traits::value_type;
  typename Traits::InputParameterController para;
  required<std::string>(para, "output");     // "material::property" differentiated
  required<std::string>(para, "input");      // "material::property" perturbed: host-bound, its codec sets the directions
  required<std::string>(para, "analytical"); // "material::property": d(output)/d(input), output indices first
  optional<value_type>(para, "step", value_type{1e-7});
  optional<std::string>(para, "step_kind", std::string{"absolute"}); // or "relative": step * max(|x_k|, 1)
  optional<std::string>(para, "scheme", std::string{"central"});     // or "forward"
  // index pairs (i, j) of a tensor input, indices of a vector input; empty: every component
  optional<std::vector<std::size_t>>(para, "components", {});
  optional<value_type>(para, "report_threshold", value_type{1e-2}); // prints the mismatches above it
  return para;
}

namespace detail::derivative {

template <class T>
using value_types = std::tuple<T, tmech::tensor<T, 3, 1>, tmech::tensor<T, 3, 2>, tmech::tensor<T, 3, 4>>;

template <class T, class V>
std::vector<T> flatten(V const& v) {
  if constexpr (std::is_arithmetic_v<V>)
    return {v};
  else
    return {v.raw_data(), v.raw_data() + V::size()};
}

template <class T, class V>
V unflatten(std::vector<T> const& x) {
  if constexpr (std::is_arithmetic_v<V>) {
    return x.front();
  } else {
    V v;
    std::ranges::copy(x, v.raw_data());
    return v;
  }
}

template <class T, class F>
void for_each_type(F&& f) {
  [&]<class... V>(std::tuple<V...>*) { (f(std::type_identity<V>{}), ...); }(static_cast<value_types<T>*>(nullptr));
}

/// The components of `prop`'s value (the new one of a history), when it holds one of value_types.
template <class T>
std::optional<std::vector<T>> read(property_base const* prop) {
  std::optional<std::vector<T>> out;
  for_each_type<T>([&]<class V>(std::type_identity<V>) {
    if (out)
      return;
    if (auto const* p = dynamic_cast<property<V> const*>(prop))
      out = flatten<T>(p->get());
    else if (auto const* h = dynamic_cast<history_property<V, property_traits> const*>(prop))
      out = flatten<T>(h->new_value());
  });
  return out;
}

/// A direction of the input: its unit components (row-major) and its label.
template <class T>
struct direction {
  std::string label;
  std::vector<T> unit;
  std::size_t first; ///< a component it moves, for a relative step
};

/// The directions of an input of `count` components laid out by a codec of `width` slots (0: no codec).
template <class T>
std::vector<direction<T>> directions(std::string const& key, std::size_t count, std::size_t width,
                                     std::vector<std::size_t> const& components) {
  auto const unit{[&](std::initializer_list<std::size_t> at) {
    std::vector<T> e(count, T{0});
    for (auto const k : at)
      e[k] = T{1};
    return e;
  }};
  auto const index{[&](std::size_t i) {
    if (i > 2)
      throw std::invalid_argument("check_derivative(): component index " + std::to_string(i) + " of '" + key + "' is not 0, 1 or 2");
    return i;
  }};
  std::vector<direction<T>> out;
  if (count == 1) {
    if (!components.empty())
      throw std::invalid_argument("check_derivative(): '" + key + "' is a scalar: no components to select");
    out.push_back({"x", unit({0}), 0});
  } else if (count == 3 && width == 3) {
    std::vector<std::size_t> all{0, 1, 2};
    for (auto const i : components.empty() ? all : components)
      if (auto const label{"(" + std::to_string(index(i)) + ")"}; !std::ranges::contains(out, label, &direction<T>::label))
        out.push_back({label, unit({i}), i});
  } else if (count == 9 && (width == 6 || width == 9)) {
    bool const symmetric{width == 6};
    std::vector<std::size_t> pairs{components};
    if (pairs.empty())
      for (std::size_t i{0}; i < 3; ++i)
        for (std::size_t j{symmetric ? i : 0}; j < 3; ++j)
          pairs.insert(pairs.end(), {i, j});
    if (pairs.size() % 2 != 0)
      throw std::invalid_argument("check_derivative(): the components of '" + key + "' are index pairs (i, j)");
    for (std::size_t k{0}; k < pairs.size(); k += 2) {
      auto i{index(pairs[k])}, j{index(pairs[k + 1])};
      if (symmetric && i > j)
        std::swap(i, j); // (j, i) is the same direction
      auto const label{"(" + std::to_string(i) + "," + std::to_string(j) + ")"};
      if (std::ranges::contains(out, label, &direction<T>::label))
        continue;
      out.push_back({label, symmetric ? unit({3 * i + j, 3 * j + i}) : unit({3 * i + j}), 3 * i + j});
    }
  } else {
    throw std::invalid_argument("check_derivative(): '" + key + "' has no codec to perturb it by: declare it as a history " +
                                "of a scalar, a vector (codecs::vector), a symmetric (codecs::strain) or a full tensor " +
                                "(codecs::two_point)");
  }
  return out;
}

/// Writes components into `prop`: the new value of a history, else the value (marking the graph as written).
template <class T, class Context>
std::function<void(std::vector<T> const&)> writer(Context& ctx, property_base* prop) {
  std::function<void(std::vector<T> const&)> out;
  for_each_type<T>([&]<class V>(std::type_identity<V>) {
    if (out)
      return;
    if (auto* h = dynamic_cast<history_property<V, property_traits>*>(prop))
      out = [h](std::vector<T> const& x) { h->new_value() = unflatten<T, V>(x); };
    else if (dynamic_cast<property<V>*>(prop))
      out = [&ctx, prop](std::vector<T> const& x) {
        ctx.template get_mutable<V>(prop->traits().id.owner, prop->traits().id.name) = unflatten<T, V>(x);
      };
  });
  return out;
}

template <class T>
T norm(std::vector<T> const& v) {
  T sum{0};
  for (auto const x : v)
    sum += x * x;
  return std::sqrt(sum);
}

template <class T>
void report(std::string const& output, std::string const& input, derivative_check<T> const& check, T threshold) {
  std::println("  check_derivative d({})/d({}): relative error {:.3e} above {:.2e}, asymmetry {:.3e}", output, input,
               check.rel_error, threshold, check.asymmetry);
  for (auto const& d : check.directions)
    for (std::size_t k{0}; k < d.numerical.size(); ++k) {
      auto const diff{std::abs(d.analytical[k] - d.numerical[k])};
      auto const ref{std::max(std::abs(d.analytical[k]), std::abs(d.numerical[k]))};
      if (diff > threshold * ref && diff > T{1e-30})
        std::println("    {} output[{}]: analytical={:.6e} numerical={:.6e}", d.label, k, d.analytical[k], d.numerical[k]);
    }
}

template <class Traits, class Context, class Engine>
derivative_check<typename Traits::value_type> run(Context& ctx, Engine& engine, typename Traits::ParameterHandler parameters) {
  using T = typename Traits::value_type;
  static auto const controller{derivative_check_parameters<Traits>()};
  check_parameters(controller, parameters);
  auto const& output_key{parameters.template get<std::string>("output")};
  auto const& input_key{parameters.template get<std::string>("input")};
  auto const& analytical_key{parameters.template get<std::string>("analytical")};
  auto const step{parameters.template get<T>("step")};
  auto const& step_kind{parameters.template get<std::string>("step_kind")};
  auto const& scheme{parameters.template get<std::string>("scheme")};
  if (step_kind != "absolute" && step_kind != "relative")
    throw std::invalid_argument("check_derivative(): step_kind '" + step_kind + "' is neither 'absolute' nor 'relative'");
  if (scheme != "central" && scheme != "forward")
    throw std::invalid_argument("check_derivative(): scheme '" + scheme + "' is neither 'central' nor 'forward'");
  if (!(step > 0) || !std::isfinite(step))
    throw std::invalid_argument("check_derivative(): the step must be positive and finite");
  auto const threshold{parameters.template get<T>("report_threshold")};
  if (!(threshold >= 0))
    throw std::invalid_argument("check_derivative(): the report_threshold must not be negative");

  auto const find{[&](std::string const& key) {
    auto const src{connection_source::parse(key)};
    auto* prop{ctx.find_property(src.material, src.property)};
    if (!prop)
      throw std::runtime_error("check_derivative(): property '" + key + "' not found");
    return prop;
  }};
  auto const value_of{[&](std::string const& key, property_base const* prop) {
    auto value{read<T>(prop)};
    if (!value)
      throw std::invalid_argument("check_derivative(): '" + key + "' holds no scalar or rank 1, 2 or 4 tensor");
    return *value;
  }};
  auto* input{find(input_key)};
  auto const* output{find(output_key)};
  auto const* analytical{find(analytical_key)};
  if (!engine.host_bound(input))
    throw std::logic_error("check_derivative(): '" + input_key + "' is filled by a callback, which update() reruns over " +
                           "the perturbation: perturb a host-bound input (external_*_source) instead");
  if (!engine.depends_on(output, input))
    throw std::invalid_argument("check_derivative(): '" + output_key + "' does not depend on '" + input_key + "'");
  auto const x0{value_of(input_key, input)};
  std::size_t width{0};
  for (auto const* mat : ctx.materials())
    if (mat->name() == input->traits().id.owner)
      if (auto const* slot{mat->history_slot_of(input)})
        width = slot->width();
  auto const dirs{directions<T>(input_key, x0.size(), width, parameters.template get<std::vector<std::size_t>>("components"))};
  auto const write{writer<T>(ctx, input)};
  auto const outputs{value_of(output_key, output).size()};
  if (auto const n{value_of(analytical_key, analytical).size()}; n != outputs * x0.size())
    throw std::invalid_argument("check_derivative(): '" + analytical_key + "' has " + std::to_string(n) +
                                " components, d(output)/d(input) has " + std::to_string(outputs * x0.size()));

  auto const evaluate{[&](std::vector<T> const& x) {
    engine.revert_produced();
    write(x);
    ctx.update();
    return value_of(output_key, output);
  }};
  derivative_check<T> check;
  std::vector<T> skew;
  try {
    auto const f0{evaluate(x0)};
    auto const A{value_of(analytical_key, analytical)};
    for (auto const& d : dirs) {
      auto const h{step_kind == "relative" ? step * std::max(std::abs(x0[d.first]), T{1}) : step};
      auto const moved{[&](T s) {
        auto x{x0};
        for (std::size_t m{0}; m < x.size(); ++m)
          x[m] += s * h * d.unit[m];
        return x;
      }};
      auto const plus{evaluate(moved(1))};
      auto const base{scheme == "central" ? evaluate(moved(-1)) : f0};
      T const divisor{scheme == "central" ? 2 * h : h};
      derivative_direction<T> r{d.label, std::vector<T>(f0.size()), std::vector<T>(f0.size(), T{0})};
      for (std::size_t o{0}; o < f0.size(); ++o) {
        r.numerical[o] = (plus[o] - base[o]) / divisor;
        for (std::size_t m{0}; m < x0.size(); ++m)
          r.analytical[o] += A[o * x0.size() + m] * d.unit[m];
      }
      check.directions.push_back(std::move(r));
    }
    if (width == 6) // the directions see A_ij + A_ji only
      for (std::size_t o{0}; o < outputs; ++o)
        for (std::size_t i{0}; i < 3; ++i)
          for (std::size_t j{i + 1}; j < 3; ++j)
            skew.push_back(A[o * 9 + 3 * i + j] - A[o * 9 + 3 * j + i]);
    evaluate(x0);
  } catch (...) {
    engine.revert_produced();
    write(x0);
    throw;
  }

  std::vector<T> diff{skew}, ref;
  for (auto const& d : check.directions)
    for (std::size_t o{0}; o < d.numerical.size(); ++o) {
      diff.push_back(d.analytical[o] - d.numerical[o]);
      ref.push_back(d.analytical[o]);
    }
  check.asymmetry = norm(skew);
  check.error = norm(diff);
  auto const scale{norm(ref)};
  check.rel_error = scale > T{1e-30} ? check.error / scale : check.error;
  if (check.rel_error > threshold)
    report(output_key, input_key, check, threshold);
  return check;
}

} // namespace detail::derivative

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_DERIVATIVE_CHECK_H
