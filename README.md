# numsim-materials

Property-graph material framework for FEM simulations. C++23, header-only.

## Architecture

Everything is a material — solvers, postprocessors, and actual constitutive models all share the same base class and participate in a property dependency graph.

```
material_context (facade)
├── object_store      — creation & lifetime management
├── property_engine   — topological sort & execution
└── materials         — scalar_stepper, linear_elasticity, isotropic_damage, ...
```

**Core API:**
- `add_output<T>("name", &callback)` — produce a property
- `add_input<T>(source, prop, EdgeKind)` — consume a property (wired at `finalize()`)
- `ctx.update()` — execute all properties in topological order
- `ctx.update_property("mat", "prop")` — selective upstream update
- `ctx.commit()` / `ctx.revert()` — timestep history management
- `ctx.save_state("checkpoint.bin")` / `ctx.load_state(...)` — binary checkpoint/restart

## Materials

### Constitutive models
- `linear_elasticity` — isotropic K/G formulation
- `isotropic_damage` — decomposed: state function + yield + damage law + assembler
- `von_mises_state_function`, `strain_energy_state_function`, `vector_strain_state_function`
- `strain_threshold_yield`, `exponential_damage_law`, `linear_damage_law`

### Solvers & steppers
- `backward_euler` — Newton-Raphson solver (as a material)
- `scalar_stepper`, `tensor_component_stepper` — loading drivers

### Postprocessing
- `numerical_diff_checker` — verify analytical tangents against numerical central differences
- `property_console_logger`, `property_plot` (Qt6 optional)

## JSON-driven configuration

```cpp
#include <numsim-materials/io/json_material_factory.h>
#include <numsim-materials/default_materials.h>

numsim::materials::register_default_materials<policy>();

auto json = nlohmann::json::parse(R"({
  "materials": [
    {"type": "tensor_component_stepper_rank2", "name": "stepper", "increment": 0.001, "indices": [0, 0]},
    {"type": "linear_elasticity", "name": "elastic", "strain_producer_name": "stepper", "K": 166.67, "G": 76.92}
  ]
})");

for (auto& mat : json["materials"])
  numsim::materials::create_from_json(ctx, mat);
```

## Build

```bash
cmake -B build -DCMAKE_CXX_COMPILER=g++-14
cmake --build build -j$(nproc)
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `NUMSIM_BUILD_TESTS` | ON | Build GTest unit tests |
| `NUMSIM_BUILD_EXAMPLES` | ON | Build examples |
| `ENABLE_PLOTTING` | OFF | Qt6 + QCustomPlot live plotting |

### Run tests

```bash
cmake -B build -DNUMSIM_BUILD_TESTS=ON -DCMAKE_CXX_COMPILER=g++-14
cmake --build build
cd build && ctest --output-on-failure
```

## Dependencies

- C++23 (GCC 14+ or Clang 18+)
- [numsim-core](https://github.com/NumSim-Stack/numsim-core) — fetched automatically via CMake
- [tmech](https://github.com/petlenz/tmech) — tensor library (`find_package(tmech REQUIRED)`)
- [nlohmann/json](https://github.com/nlohmann/json) — for JSON configuration (optional)
- Qt6 + QCustomPlot — for live plotting (optional)

## Structure

```
include/numsim-materials/
├── core/              — material_base, material_context, property_engine, traits
├── materials/         — constitutive models, steppers, weights
├── solvers/           — backward_euler
├── postprocessing/    — logging, plotting, numerical_diff_checker
├── io/                — json_parameter_converter, json_material_factory
└── default_materials.h — factory registration + macros
```
