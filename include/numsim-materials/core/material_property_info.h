#ifndef MATERIAL_PROPERTY_INFO_H
#define MATERIAL_PROPERTY_INFO_H

#include <format>
#include <string>
#include <vector>
#include "numsim-materials/core/expected.h"
#include "numsim-materials/core/property.h"
#include "numsim-materials/core/material_interface.h"

namespace numsim::materials {

class material_property_info {
public:
  template<typename Traits>
  static expected<material_property_info, std::string>
  from_material(const material_interface<Traits>& material) noexcept {
    try {
      const auto& registry = material.get_property_registry();

      auto extract = [](const auto& props) {
        std::vector<const property_traits*> traits;
        for (const property_base* p : props) traits.push_back(&(p->traits()));
        return traits;
      };

      return material_property_info{
          material.name(),
          extract(registry.produced_properties())
      };
    } catch (const std::exception& e) {
      return unexpected(std::format("Failed to extract from '{}': {}", material.name(), e.what()));
    } catch (...) {
      return unexpected(std::format("Unknown error extracting from '{}'", material.name()));
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

#endif // MATERIAL_PROPERTY_INFO_H
