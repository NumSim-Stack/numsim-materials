# numsim-materials

Property-graph material framework for FEM simulations. C++23, header-only.

## Architecture

Everything is a material — solvers, postprocessors, and actual materials all share the same base class and participate in a property dependency graph.

- **`add_output<T>("name", &callback)`** — produce a property
- **`add_input<T>(source, prop, EdgeKind)`** — consume a property (wired at `finalize()`)
- **`update_source()`** — re-evaluate upstream properties (for solver inner loops)
- **`ctx.update()`** — execute all properties in topological order
- **`ctx.update_property("mat", "prop")`** — selective update with upstream dependencies
- **`ctx.commit()` / `ctx.revert()`** — timestep history management

## Structure

```
include/numsim-materials/
├── core/              — framework: material_base, material_context, property_engine
├── materials/         — scalar_stepper, linear_elasticity, autocatalytic_reaction, ...
├── solvers/           — backward_euler (as a material)
├── postprocessing/    — property_console_logger, property_plot (Qt optional)
└── default_materials.h — factory registration
```

Core property types (`property_base`, `history_property`, `input_property`, etc.) live in [numsim-core](https://github.com/NumSim-Stack/numsim-core) for reuse across libraries.

## Quick Start

```cpp
#include <numsim-materials/core/material_context.h>
#include <numsim-materials/materials/scalar_stepper.h>
#include <numsim-materials/default_materials.h>

using policy = numsim::materials::material_policy_default;

numsim::materials::material_context<policy> ctx;
typename policy::ParameterHandler params;

params.insert<std::string>("name", "time");
params.insert<double>("increment", 0.1);
ctx.create<numsim::materials::scalar_stepper<policy>>(params);

ctx.finalize();
for (int i = 0; i < 100; ++i) {
    ctx.update();
    ctx.commit();
}
```

## Dependencies

- [numsim-core](https://github.com/NumSim-Stack/numsim-core) (fetched via CMake)
- [tmech](https://github.com/petlenz/tmech) (for tensor materials)
- Qt6 + QCustomPlot (optional, for live plotting)

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

With plotting:
```bash
cmake .. -DENABLE_PLOTTING=ON
```
