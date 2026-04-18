#ifndef NUMSIM_MATERIALS_PROPERTY_H
#define NUMSIM_MATERIALS_PROPERTY_H

#include <numsim-core/property_graph/property.h>

namespace numsim::materials {
  // Concrete aliases for the default property_traits
  using property_base = numsim_core::property_base<numsim_core::property_traits>;
  using property_base_ptr = numsim_core::property_base_ptr<numsim_core::property_traits>;
  template<typename T>
  using property = numsim_core::property<T, numsim_core::property_traits>;
  using numsim_core::make_property;
}

#endif // NUMSIM_MATERIALS_PROPERTY_H
