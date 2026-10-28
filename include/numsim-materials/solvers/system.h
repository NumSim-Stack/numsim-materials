#ifndef NUMSIM_MATERIALS_SOLVERS_SYSTEM_H
#define NUMSIM_MATERIALS_SOLVERS_SYSTEM_H

#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <tmech/tmech.h>
#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"
#include "numsim-materials/core/unknown_spec.h"

namespace numsim::materials {

/// A residual system set up at run time (#218), "type": "system": its unknowns are outputs of this node, its
/// residuals ("equations": [{"residual": "material::property", "for": unknown}]) outputs of any nodes reading them.
/// update() solves it by "method" newton (J from the partials), newton_fd or picard (x -= R; without unknowns
/// the group of converges()); "seed": "history" starts an unknown from its value at the end of the last step.
template <typename Traits>
class residual_system final : public material_base<residual_system<Traits>, Traits> {
public:
  using base = material_base<residual_system<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;
  using tensor2 = tmech::tensor<value_type, base::Dim, 2>;

  template <typename... Args>
  explicit residual_system(Args&&... args)
      : base(std::forward<Args>(args)...), m_tolerance(base::template get_parameter<value_type>("tolerance")),
        m_max_iter(base::template get_parameter<int>("max_iter")) {
    auto const& unknowns{base::template get_parameter<std::vector<system_unknown>>("unknowns")};
    auto const& equations{base::template get_parameter<std::vector<system_equation>>("equations")};
    auto const method{check(unknowns, equations)};
    std::vector<void const*> outputs;
    std::vector<value_type> seeds;
    std::vector<input_wire_base const*> residuals;
    for (auto const& u : unknowns) {
      auto const& residual{std::ranges::find(equations, u.name, &system_equation::unknown)->residual};
      if (u.kind == unknown_kind::scalar)
        outputs.push_back(add<value_type>(u, residual, residuals));
      else
        outputs.push_back(add<tensor2>(u, residual, residuals));
      seeds.push_back(static_cast<value_type>(u.seed));
      m_names.push_back(u.name);
    }
    if (unknowns.empty())
      for (auto const& e : equations)
        residuals.push_back(&input<value_type>(e.residual));
    base::declare_system(method, outputs, std::move(seeds), std::move(residuals), m_tolerance, m_max_iter, m_report);
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    optional<std::string>(para, "method", std::string{"newton"});
    optional<std::vector<system_unknown>>(para, "unknowns", {});
    required<std::vector<system_equation>>(para, "equations");
    optional<value_type>(para, "tolerance", value_type{1e-10});
    optional<int>(para, "max_iter", 50);
    return para;
  }

  /// The updates of x the last solve took.
  int iterations() const noexcept { return m_report.iterations; }

  /// The Jacobian blocks (row unknown, column unknown) without a path from the column unknown to the residual
  /// of the row: structurally zero (after finalize()).
  std::vector<block_ref> zero_blocks() const {
    return m_report.zero_blocks |
           std::views::transform([this](auto const& b) { return block_ref{m_names[b.first], m_names[b.second]}; }) |
           std::ranges::to<std::vector>();
  }

private:
  /// The unknown `u` (history or not) and the input of its residual; returns the output.
  template <typename V>
  void const* add(system_unknown const& u, std::string const& residual, std::vector<input_wire_base const*>& residuals) {
    residuals.push_back(&input<V>(residual));
    if (u.history)
      return &base::template add_history_output<V>(u.name);
    return &base::template add_output<V>(u.name);
  }

  template <typename V>
  input_property<V, property_traits> const& input(std::string const& residual) {
    auto const split{residual.find("::")};
    return base::template add_input<V>(residual.substr(0, split), residual.substr(split + 2), EdgeKind::Local);
  }

  [[noreturn]] void fail(std::string const& what) const {
    throw std::invalid_argument("system '" + base::name() + "': " + what);
  }

  /// Validates the parameters before any property is created; returns the method.
  system_method check(std::vector<system_unknown> const& unknowns, std::vector<system_equation> const& equations) const {
    auto const& name{base::template get_parameter<std::string>("method")};
    auto const method{name == "newton" ? system_method::newton : name == "newton_fd" ? system_method::newton_fd
                                         : name == "picard"  ? system_method::picard
                                                             : throw std::invalid_argument("system '" + base::name() + "': unknown method '" +
                                                                                           name + "' (newton, newton_fd or picard)")};
    if (unknowns.empty() && method != system_method::picard)
      fail("'unknowns' is empty; only picard iterates without unknowns");
    if (equations.empty())
      fail("'equations' is empty");
    if (!(m_tolerance >= 0) || m_max_iter < 0)
      fail("needs a tolerance >= 0 and max_iter >= 0");
    for (auto const& e : equations) {
      auto const split{e.residual.find("::")};
      if (split == std::string::npos || split == 0 || split + 2 >= e.residual.size())
        fail("residual '" + e.residual + "' is not \"material::property\"");
      if (unknowns.empty() && !e.unknown.empty())
        fail("the equation for '" + e.unknown + "': a system without unknowns has no \"for\"");
      if (!unknowns.empty() && !std::ranges::contains(unknowns, e.unknown, &system_unknown::name))
        fail("residual '" + e.residual + "' is for '" + e.unknown + "', which is not an unknown");
    }
    for (auto it{unknowns.begin()}; it != unknowns.end(); ++it) {
      if (it->name.empty())
        fail("an unknown has an empty name");
      if (std::ranges::contains(unknowns.begin(), it, it->name, &system_unknown::name))
        fail("unknown '" + it->name + "' is declared twice");
      if (it->kind == unknown_kind::sym_tensor && it->seed != 0)
        fail("unknown '" + it->name + "': the seed of a sym_tensor is 0 or \"history\"");
      auto const count{std::ranges::count(equations, it->name, &system_equation::unknown)};
      if (count != 1)
        fail("unknown '" + it->name + "' has " + std::to_string(count) + " equations, needs one");
    }
    return method;
  }

  value_type const& m_tolerance;
  int const& m_max_iter;
  std::vector<std::string> m_names;
  system_report m_report;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_SOLVERS_SYSTEM_H
