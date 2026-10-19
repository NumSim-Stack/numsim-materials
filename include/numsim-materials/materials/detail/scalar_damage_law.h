#ifndef NUMSIM_MATERIALS_DETAIL_SCALAR_DAMAGE_LAW_H
#define NUMSIM_MATERIALS_DETAIL_SCALAR_DAMAGE_LAW_H

#include <stdexcept>
#include <string>
#include <utility>

#include "numsim-materials/core/material_base.h"
#include "numsim-materials/core/parameter_helpers.h"

namespace numsim::materials::detail {

/// Common part of the scalar damage laws: kappa from "yield_source" ("kappa"), the threshold "kappa_0",
/// the cap "d_max", outputs "damage" and "d_damage" (d damage / d kappa). Derived::evaluate(kappa) returns
/// {d, dd} above the threshold; at the cap the damage is d_max and the derivative zero.
template <typename Derived, typename Traits>
class scalar_damage_law : public material_base<Derived, Traits> {
public:
  using base = material_base<Derived, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit scalar_damage_law(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_d_damage(base::template add_output<value_type>("d_damage")),
        m_kappa_0(base::template get_parameter<value_type>("kappa_0")),
        m_d_max(base::template get_parameter<value_type>("d_max")),
        m_kappa(base::template add_input<value_type>(base::template get_parameter<std::string>("yield_source"), "kappa", EdgeKind::Global)) {
    if (!(m_kappa_0 >= 0))
      throw std::invalid_argument(Derived::type_name() + std::string{" '"} + base::name() + "': kappa_0 must be non-negative");
    if (!(m_d_max > 0 && m_d_max <= 1))
      throw std::invalid_argument(Derived::type_name() + std::string{" '"} + base::name() + "': d_max must lie in (0, 1]");
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    required<std::string>(para, "yield_source");
    optional<value_type>(para, "d_max", value_type{0.999});
    return para;
  }

  void compute() noexcept {
    auto const kappa{m_kappa.get()};
    if (kappa <= m_kappa_0) {
      *m_damage = m_d_damage = value_type{0};
      return;
    }
    auto const [d, dd]{static_cast<Derived const &>(*this).evaluate(kappa)};
    *m_damage = d < m_d_max ? d : m_d_max;
    m_d_damage = d < m_d_max ? dd : value_type{0};
  }

protected:
  /// Called by the law's own constructor: binding the update needs the most-derived object.
  void register_output() { m_damage = &base::template add_output<value_type>("damage", &scalar_damage_law::compute); }

  /// The law's own parameter check, with the type name in the message.
  void require(bool condition, char const *what) const {
    if (!condition)
      throw std::invalid_argument(Derived::type_name() + std::string{" '"} + base::name() + "': " + what);
  }

  [[nodiscard]] value_type kappa_0() const noexcept { return m_kappa_0; }

private:
  value_type *m_damage{nullptr};
  value_type &m_d_damage;
  value_type const &m_kappa_0;
  value_type const &m_d_max;
  input_property<value_type, property_traits> const &m_kappa;
};

} // namespace numsim::materials::detail

#endif // NUMSIM_MATERIALS_DETAIL_SCALAR_DAMAGE_LAW_H
