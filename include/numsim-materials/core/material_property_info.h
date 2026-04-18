#ifndef NUMSIM_MATERIALS_MATERIAL_PROPERTY_INFO_H
#define NUMSIM_MATERIALS_MATERIAL_PROPERTY_INFO_H

#include <expected>
#include <format>
#include <ranges>
#include <string>
#include <vector>
#include "numsim-materials/core/property.h"
#include "numsim-materials/core/material_interface.h"

namespace numsim::materials {

class material_property_info {
public:
  template<typename Traits>
  static std::expected<material_property_info, std::string>
  from_material(const material_interface<Traits>& material) noexcept {
    try {
      const auto& registry = material.get_property_registry();

      auto extract = [](const auto& props) {
        return props
               | std::views::transform([](const property_base* p) {
                   return &(p->traits());
                 })
               | std::ranges::to<std::vector<const property_traits*>>();
      };

      return material_property_info{
          material.name(),
          extract(registry.produced_properties())
      };
    } catch (const std::exception& e) {
      return std::unexpected(std::format("Failed to extract from '{}': {}", material.name(), e.what()));
    } catch (...) {
      return std::unexpected(std::format("Unknown error extracting from '{}'", material.name()));
    }
  }

  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] const auto& produced() const noexcept { return produced_; }

private:
  std::string name_;
  std::vector<const property_traits*> produced_;

  material_property_info(std::string_view name,
                         std::vector<const property_traits*> produced)
      : name_{name}, produced_{std::move(produced)} {}
};

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_MATERIAL_PROPERTY_INFO_H
