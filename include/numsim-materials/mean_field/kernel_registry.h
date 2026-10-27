#ifndef NUMSIM_MATERIALS_MEAN_FIELD_KERNEL_REGISTRY_H
#define NUMSIM_MATERIALS_MEAN_FIELD_KERNEL_REGISTRY_H

#include "numsim-materials/core/object_spec.h"
#include "numsim-materials/mean_field/kernel.h"
#include "numsim-materials/mean_field/kernels.h"
#include "numsim-materials/mean_field/shapes.h"

namespace numsim::materials::mean_field {

/// The kernels a deck can name by "type"; the six schemes built in, more by add<K>("name").
template <typename Traits>
class kernel_registry {
public:
  using type = spec_registry<kernel<Traits>, typename Traits::InputParameterController, typename Traits::ParameterHandler>;

  static type &instance() {
    static type registry{[] {
      type r{"kernel"};
      r.template add<voigt<Traits>>("voigt").template add<reuss<Traits>>("reuss").template add<hill<Traits>>("hill");
      r.template add<dilute<Traits>>("dilute").template add<mori_tanaka<Traits>>("mori_tanaka");
      r.template add<self_consistent<Traits>>("self_consistent");
      return r;
    }()};
    return registry;
  }
};

} // namespace numsim::materials::mean_field

#endif // NUMSIM_MATERIALS_MEAN_FIELD_KERNEL_REGISTRY_H
