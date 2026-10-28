#ifndef NUMSIM_MATERIALS_CORE_PARTIAL_CHECK_H
#define NUMSIM_MATERIALS_CORE_PARTIAL_CHECK_H

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "numsim-materials/core/material_context.h"

namespace numsim::materials {

/// Compares every partial `material` declares with central differences of its own callback, one Mandel component
/// of the read at a time, at the state the context holds (after an update()). Returns one line per wrong partial;
/// the outputs and reads are restored.
template <typename Traits>
std::vector<std::string> check_partials(material_context<Traits>& ctx, std::string const& material,
                                        typename Traits::value_type tolerance = 1e-6) {
  using T = typename Traits::value_type;
  auto* mat{ctx.find(material)};
  if (!mat)
    throw std::invalid_argument("check_partials(): no material '" + material + "'");
  auto const& partials{mat->partials()};
  std::vector<property_base const*> callbacks;
  for (auto const& p : partials)
    if (!std::ranges::contains(callbacks, p.callback))
      callbacks.push_back(p.callback);

  std::vector<std::string> mismatches;
  for (auto const* callback : callbacks) {
    auto* runs{ctx.find_property(material, callback->traits().id.name)};
    auto const run{[runs] { runs->traits().update(); }};
    bool const requested{mat->partials_requested(callback)};
    mat->request_partials(callback, true);
    run();
    mat->request_partials(callback, requested);
    for (auto const& p : partials | std::views::filter([callback](auto const& q) { return q.callback == callback; })) {
      auto const& block{*p.block};
      std::vector<T> analytic(block.rows() * block.cols()), plus(block.rows()), minus(block.rows());
      block.pack(analytic.data());
      auto* source{p.input ? ctx.find_property(p.input->source_owner(), p.input->source_name())
                           : ctx.find_property(material, p.own->traits().id.name)};
      auto const read{source->traits().id.owner + "::" + source->traits().id.name};
      T scale{0};
      for (auto const a : analytic)
        scale = std::max(scale, std::abs(a));
      for (std::size_t k{0}; k < block.cols(); ++k) {
        T const h{std::cbrt(std::numeric_limits<T>::epsilon()) * std::max(T{1}, std::abs(block.read_component(*source, k)))};
        for (auto [step, values] : {std::pair{h, &plus}, std::pair{-h, &minus}}) {
          auto const restore{block.perturb(*source, k, step)};
          run();
          block.pack_output(values->data());
          restore();
        }
        T error{0}, size{scale}, output{0};
        for (std::size_t r{0}; r < block.rows(); ++r) {
          T const fd{(plus[r] - minus[r]) / (2 * h)};
          error = std::max(error, std::abs(fd - analytic[r + k * block.rows()]));
          size = std::max(size, std::abs(fd));
          output = std::max({output, std::abs(plus[r]), std::abs(minus[r])});
        }
        // relative to the largest entry, above the rounding of the differences
        if (error > tolerance * size + 10 * std::numeric_limits<T>::epsilon() * output / h) {
          mismatches.push_back(std::format("'{}::{}' wrt '{}': Mandel column {} differs from central differences by {:g} "
                                           "(largest entry {:g})",
                                           material, p.output->traits().id.name, read, k, error, size));
          break;
        }
      }
    }
    run();
  }
  return mismatches;
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_PARTIAL_CHECK_H
