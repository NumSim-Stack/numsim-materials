#ifndef NUMSIM_MATERIALS_CORE_PROPERTY_VALUE_H
#define NUMSIM_MATERIALS_CORE_PROPERTY_VALUE_H

#include "numsim-materials/core/history_property.h"
#include "numsim-materials/core/property.h"

namespace numsim::materials {

/// Value of a plain property, or the new value of a history property; nullptr for another type.
template <typename T>
T const *property_value(property_base const *p) noexcept {
  if (auto const *plain{dynamic_cast<property<T> const *>(p)})
    return &plain->get();
  if (auto const *hist{dynamic_cast<history_property<T, property_traits> const *>(p)})
    return &hist->new_value();
  return nullptr;
}

} // namespace numsim::materials

#endif // NUMSIM_MATERIALS_CORE_PROPERTY_VALUE_H
