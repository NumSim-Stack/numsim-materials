#ifndef NUMSIM_MATERIALS_DRIVER_LOAD_CONTROL_H
#define NUMSIM_MATERIALS_DRIVER_LOAD_CONTROL_H

#include <array>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <tmech/tmech.h>
#include "numsim-materials/core/tensor_identities.h"

namespace numsim::materials {

/// Which symmetric components are stress controlled; the rest are strain controlled. mixed() names each once.
template <typename T>
class load_control {
public:
  using tensor4 = tmech::tensor<T, 3, 4>;

  static load_control strain() { return load_control{}; }
  static load_control uniaxial_stress() { return mixed({"22", "33", "12", "13", "23"}); }
  static load_control mixed(std::initializer_list<std::string_view> stress_components) {
    return mixed(std::span{stress_components.begin(), stress_components.size()});
  }
  static load_control mixed(std::span<std::string_view const> stress_components) {
    load_control c;
    std::array<bool, 6> given{};
    for (auto const name : stress_components) {
      auto const a{materials::component_index(name)};
      if (std::exchange(given[a], true))
        throw std::invalid_argument("load_control: component '" + std::string{name} + "' is given twice");
      auto const &E{materials::symmetric_basis<T>()[a]};
      c.m_P += tmech::otimes(E, E);
    }
    return c;
  }

  /// Projector onto the stress-controlled components.
  [[nodiscard]] tensor4 const &stress_projector() const noexcept { return m_P; }

  /// Projector onto the strain-controlled components.
  [[nodiscard]] tensor4 strain_projector() const noexcept { return tensor4{sym_identity<T, 3>() - m_P}; }

private:
  load_control() = default;
  tensor4 m_P;
};

/// Which of the nine components of F are stress (P) controlled; the rest are prescribed. mixed() names each once.
template <typename T>
class deformation_control {
public:
  using tensor4 = tmech::tensor<T, 3, 4>;

  static deformation_control deformation() { return deformation_control{}; }
  static deformation_control uniaxial_stress() { return mixed({"22", "33", "12", "13", "23", "21", "31", "32"}); }
  static deformation_control mixed(std::initializer_list<std::string_view> stress_components) {
    return mixed(std::span{stress_components.begin(), stress_components.size()});
  }
  static deformation_control mixed(std::span<std::string_view const> stress_components) {
    deformation_control c;
    std::array<bool, 9> given{};
    for (auto const name : stress_components) {
      auto const a{materials::full_component_index(name)};
      if (std::exchange(given[a], true))
        throw std::invalid_argument("deformation_control: component '" + std::string{name} + "' is given twice");
      auto const &E{materials::full_basis<T>()[a]};
      c.m_M += tmech::otimes(E, E);
    }
    return c;
  }

  /// Projector onto the stress-controlled components.
  [[nodiscard]] tensor4 const &stress_projector() const noexcept { return m_M; }

  /// Projector onto the prescribed components of F.
  [[nodiscard]] tensor4 deformation_projector() const noexcept {
    auto const I{tmech::eye<T, 3, 2>()};
    return tensor4{tmech::otimesu(I, I) - m_M};
  }

private:
  deformation_control() = default;
  tensor4 m_M;
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_DRIVER_LOAD_CONTROL_H
