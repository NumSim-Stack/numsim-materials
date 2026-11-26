#ifndef NUMSIM_MATERIALS_CORE_PARTIAL_CHECK_H
#define NUMSIM_MATERIALS_CORE_PARTIAL_CHECK_H

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "numsim-materials/core/material_context.h"

namespace numsim::materials {

namespace partial_check_detail {
/// Runs `f` when the scope ends, also while an exception unwinds it; what `f` throws is dropped.
template <typename F>
struct on_exit {
  F f;
  ~on_exit() {
    try {
      f();
    } catch (...) {
    }
  }
};
template <typename F>
on_exit(F) -> on_exit<F>;

/// |v - sym(v)| / |v| of the rank-2 or rank-4 value `prop` holds, minor symmetry for rank 4; nothing for other values.
template <typename T, std::size_t Dim>
std::optional<T> asymmetry(property_base& prop) {
  auto const of{[&]<std::size_t R>() {
    using V = tmech::tensor<T, Dim, R>;
    auto const& v{derivatives::value_in<V>(prop)};
    std::array<T, derivatives::width<Dim>(R)> packed{};
    derivatives::pack<Dim, R>(v, packed.data());
    V symmetric;
    derivatives::unpack<Dim, R>(packed.data(), symmetric);
    T const size{tmech::norm(v)};
    return size > T{0} ? T{tmech::norm(V{v - symmetric}) / size} : T{0};
  }};
  switch (derivatives::property_rank<T, Dim>(prop).value_or(0)) {
  case 2: return of.template operator()<2>();
  case 4: return of.template operator()<4>();
  default: return std::nullopt;
  }
}
} // namespace partial_check_detail

/// Compares every partial `material` declares with central differences of its own callback, one Mandel component
/// of the read at a time (each entry of a read declared unsymmetric_output()), at the state the context holds (after
/// an update()), and perturbs every read declared independent_of() to see that no output of the callback moves.
/// Returns one line per wrong declaration; the outputs and reads are restored, also when a perturbed callback throws.
/// Also reports every rank-2 or rank-4 output and read whose value is not (minor-)symmetric and not declared
/// unsymmetric_output() by its producer.
template <typename Traits>
std::vector<std::string> check_partials(material_context<Traits>& ctx, std::string const& material,
                                        typename Traits::value_type tolerance = 1e-6) {
  using T = typename Traits::value_type;
  using partial_check_detail::on_exit;
  constexpr T eps{std::numeric_limits<T>::epsilon()};
  constexpr T scale_floor{1e-3}; // the scale of a read that is zero, or nearly
  constexpr T independence_step{1e-3};
  auto* mat{ctx.find(material)};
  if (!mat)
    throw std::invalid_argument("check_partials(): no material '" + material + "'");

  auto const run_of{[&](auto const* callback) {
    auto* runs{ctx.find_property(material, callback->traits().id.name)};
    return [runs] { runs->traits().update(); };
  }};
  auto const source_of{[&](auto const& d) {
    return d.input ? ctx.find_property(d.input->source_owner(), d.input->source_name())
                   : ctx.find_property(material, d.own->traits().id.name);
  }};
  auto const key_of{[](auto const* source) { return source->traits().id.owner + "::" + source->traits().id.name; }};
  // a read is unsymmetric as its producer declares it
  auto const unsymmetric{[&](auto const* source) {
    auto const* producer{ctx.find(source->traits().id.owner)};
    return producer && producer->is_unsymmetric(source);
  }};
  // the largest component of the read: one step for all of them
  auto const read_scale{[&](auto const& read, auto& source, bool full) {
    T scale{scale_floor};
    for (std::size_t k{0}; k < read.width(full); ++k)
      scale = std::max(scale, std::abs(read.read_component(source, k, full)));
    return scale;
  }};

  std::vector<std::string> mismatches;
  auto const undeclared{[&](property_base& value, bool declared) -> std::optional<std::string> {
    auto const distance{partial_check_detail::asymmetry<T, Traits::Dim>(value)};
    if (declared || !distance || !(*distance > tolerance))
      return std::nullopt;
    bool const rank2{derivatives::property_rank<T, Traits::Dim>(value) == 2};
    return std::format("is not {} (relative distance {:g} to its symmetric part) and not declared unsymmetric_output()",
                       rank2 ? "symmetric" : "minor-symmetric", *distance);
  }};
  if (auto const own{ctx.properties().data().find(material)}; own != ctx.properties().data().end()) {
    std::vector<std::pair<std::string, property_base*>> outputs;
    for (auto const& [name, prop] : own->second)
      outputs.emplace_back(name, prop.get());
    std::ranges::sort(outputs);
    for (auto const& [name, prop] : outputs)
      if (auto const what{undeclared(*prop, mat->is_unsymmetric(prop))})
        mismatches.push_back(std::format("'{}::{}' {}", material, name, *what));
  }
  for (auto const& input : mat->typed_inputs())
    if (auto* source{ctx.find_property(input->source_owner(), input->source_name())})
      if (auto const what{undeclared(*source, unsymmetric(source))})
        mismatches.push_back(std::format("'{}' reads '{}', which {} by its producer", material, key_of(source), *what));

  auto const& partials{mat->partials()};
  std::vector<property_base const*> callbacks;
  for (auto const& p : partials)
    if (!std::ranges::contains(callbacks, p.callback))
      callbacks.push_back(p.callback);

  for (auto const* callback : callbacks) {
    auto const run{run_of(callback)};
    on_exit rerun{run}; // the outputs at the state the check found
    bool const requested{mat->partials_requested(callback)};
    mat->request_partials(callback, true);
    mat->request_all_partials(true);
    {
      on_exit reset{[&] {
        mat->request_partials(callback, requested);
        mat->request_all_partials(false);
      }};
      run();
      if (auto* fill{mat->fill_of(callback)})
        mat->fill_partials(*fill, derivatives::every_reader);
    }
    auto of_callback{partials | std::views::filter([callback](auto const& q) { return q.callback == callback; })};
    // the kinds of the output and the read, and the partials packed before any perturbed rerun overwrites them
    std::vector<std::pair<bool, bool>> full;
    std::vector<std::vector<T>> packed;
    for (auto const& p : of_callback) {
      auto const& [out, in]{full.emplace_back(mat->is_unsymmetric(p.output), unsymmetric(source_of(p)))};
      p.block->pack(packed.emplace_back(p.block->rows(out) * p.block->cols(in)).data(), out, in);
    }
    for (auto const& [p, analytic, kinds] : std::views::zip(of_callback, packed, full)) {
      auto const [full_out, full_in]{kinds};
      auto const& block{*p.block};
      auto const rows{block.rows(full_out)};
      std::vector<T> plus(rows), minus(rows);
      auto* source{source_of(p)};
      auto const read{key_of(source)};
      if (T const asymmetry{block.asymmetry(full_out, full_in)}; asymmetry > tolerance) {
        mismatches.push_back(std::format("'{}::{}' wrt '{}': the partial is not minor-symmetric in the index pairs of its symmetric "
                                         "values (relative distance {:g} to their symmetric part)",
                                         material, p.output->traits().id.name, read, asymmetry));
        continue;
      }
      T scale{0};
      for (auto const a : analytic)
        scale = std::max(scale, std::abs(a));
      T const read_size{read_scale(block, *source, full_in)};
      T const h{std::sqrt(eps) * read_size};
      for (std::size_t k{0}; k < block.cols(full_in); ++k) {
        for (auto [step, values] : {std::pair{h, &plus}, std::pair{-h, &minus}}) {
          on_exit restore{block.perturb(*source, k, step, full_in)};
          run();
          block.pack_output(values->data(), full_out);
        }
        T error{0}, size{scale}, output{0};
        for (std::size_t r{0}; r < rows; ++r) {
          T const fd{(plus[r] - minus[r]) / (2 * h)};
          error = std::max(error, std::abs(fd - analytic[r + k * rows]));
          size = std::max(size, std::abs(fd));
          output = std::max({output, std::abs(plus[r]), std::abs(minus[r])});
        }
        // Rounding of the differences, for an output no larger than the analytic block can produce: a large
        // output must not hide a small partial.
        T const allowance{10 * eps * std::min(output, scale * (read_size + h)) / h};
        if (error > tolerance * size + allowance) {
          mismatches.push_back(std::format("'{}::{}' wrt '{}': {} column {} differs from central differences by {:g} "
                                           "(largest entry {:g})",
                                           material, p.output->traits().id.name, read, full_in ? "full-notation" : "Mandel", k,
                                           error, size));
          break;
        }
      }
    }
  }

  std::vector<property_base const*> independents;
  for (auto const& i : mat->independent_reads())
    if (i.probe && !std::ranges::contains(independents, i.callback))
      independents.push_back(i.callback);
  for (auto const* callback : independents) {
    auto const run{run_of(callback)};
    on_exit rerun{run};
    for (auto const& i : mat->independent_reads()) {
      if (i.callback != callback || !i.probe)
        continue;
      auto* source{source_of(i)};
      auto const read{key_of(source)};
      bool const full_in{unsymmetric(source)};
      T const h{independence_step * read_scale(*i.probe, *source, full_in)};
      run(); // the outputs at the state the check found
      auto const deviation{mat->watch_outputs(callback)};
      bool moved{false};
      for (std::size_t k{0}; k < i.probe->width(full_in) && !moved; ++k)
        for (T const step : {h, -h}) {
          on_exit restore{i.probe->perturb(*source, k, step, full_in)};
          run();
          auto const worst{deviation()};
          if (worst.relative > tolerance) {
            mismatches.push_back(std::format("'{}::{}' is declared independent_of '{}', but {} component {} of the read "
                                             "changes it by {:g} (relative)",
                                             material, worst.output->traits().id.name, read, full_in ? "full-notation" : "Mandel", k,
                                             worst.relative));
            moved = true;
            break;
          }
        }
    }
  }
  return mismatches;
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_PARTIAL_CHECK_H
