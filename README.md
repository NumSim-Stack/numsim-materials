# numsim-materials

Property-graph material framework for FEM simulations. C++23, header-only.

A constitutive model is assembled at run time from small, independently testable
materials wired together by name. Swapping a linear hardening law for an
exponential one, or a J2 return map for Drucker-Prager, is an edit to a JSON
deck rather than a recompile — and the assembled model runs as a UMAT in
Abaqus or CalculiX.

```json
{"materials": [
  {"type": "linear_isotropic_hardening", "name": "hard", "source": "j2", "K": 1000.0},
  {"type": "local_newton", "name": "solver", "max_iter": 50, "tolerance": 1e-10},
  {"type": "j2_plasticity", "name": "j2", "strain_source": "stepper",
   "hardening_source": "hard", "solver_source": "solver",
   "K": 166666.7, "G": 76923.1, "sigma_0": 200.0}
]}
```

`K` is the bulk modulus on `j2_plasticity` and the hardening modulus on
`linear_isotropic_hardening`. That collision is tracked in
[#73](https://github.com/NumSim-Stack/numsim-materials/issues/73).

## Documentation

| | |
|---|---|
| [docs/plasticity.md](docs/plasticity.md) | Using the plasticity materials: parameters, wiring, worked decks |
| [docs/plasticity-theory.md](docs/plasticity-theory.md) | J2 and Drucker-Prager from kinematics to the consistent tangent |

## Architecture

Everything is a material — solvers, postprocessors and constitutive models all
share the same base class and participate in a property dependency graph.

```
material_context (facade)
├── object_store      — creation & lifetime management
├── property_engine   — topological sort & execution
└── materials         — scalar_stepper, linear_elasticity, j2_plasticity, ...
```

**Core API:**
- `add_output<T>("name", &callback)` — produce a property
- `add_input<T>(source, prop, EdgeKind)` — consume a property (wired at `finalize()`)
- `ctx.update()` — execute all properties in topological order
- `ctx.update_property("mat", "prop")` — selective upstream update
- `ctx.commit()` / `ctx.revert()` — timestep history management
- `ctx.save_state("checkpoint.bin")` / `ctx.load_state(...)` — binary checkpoint/restart

`commit()` and `revert()` belong to the driver. Materials and solvers never call
them.

## Plasticity

Small-strain rate-independent plasticity with a consistent (algorithmic)
tangent. The return map, the yield surface and the hardening law are separate
materials, so a hardening law is written once and reused by every return map.

| Type | |
|---|---|
| `j2_plasticity` | von Mises radial return, closed-form consistent tangent |
| `j2_rk_plasticity` | J2 integrated with a Runge-Kutta tableau (explicit or DIRK) |
| `drucker_prager_plasticity` | Non-associated Drucker-Prager with an apex return |
| `linear_isotropic_hardening` | `σ_y = σ_0 + K·κ` |
| `exponential_isotropic_hardening` | `σ_y = σ_0 + K_inf·(1 − e^(−δκ))` |
| `local_newton` | Scalar Newton solver used by the return maps |
| `vector_newton` | Coupled multi-unknown Newton, for tensor-valued unknowns |
| `backward_euler` | Graph-driven time integration |

See [docs/plasticity-theory.md](docs/plasticity-theory.md) for the derivations,
including why the tangent collapses to the `4G²` form and when it does not.

## Running as a UMAT

The assembled graph can be driven from Abaqus or CalculiX through the standard
`UMAT` interface. One `material_context` is built per thread; the evaluator
itself is stateless, and history properties are mapped to and from `STATEV`
automatically.

```cpp
#include <numsim-materials/umat/umat_interface.h>

// Defines extern "C" umat_(...) for the Abaqus/CalculiX UMAT signature.
NUMSIM_MATERIALS_DEFINE_UMAT(policy)
```

CalculiX additionally offers a native external-behaviour hook, which takes PK2
stress and `∂S/∂E` rather than the UMAT convention:

```cpp
#include <numsim-materials/umat/calculix_interface.h>

NUMSIM_MATERIALS_DEFINE_CALCULIX_BEHAVIOUR(policy, my_behaviour, "MYMODEL")
```

The model itself is described by a JSON deck named from the Abaqus `*MATERIAL`
name, so one compiled shared library serves every model in the deck. Both
headers document their conventions — tensor ordering, the engineering-shear
`STATEV` convention, time rebasing and cutback signalling — at the top of the
file.

Small-strain materials are valid for large rotation with small strain; there is
no finite-strain formulation yet.

## Material catalogue

All 29 types registered by `register_default_materials<policy>()`.

**Elasticity and stress** — `linear_elasticity`, `linear_stress`,
`isotropic_tangent`, `weighted_sum`

**Plasticity** — `j2_plasticity`, `j2_rk_plasticity`,
`drucker_prager_plasticity`, `linear_isotropic_hardening`,
`exponential_isotropic_hardening`

**Damage** — `isotropic_damage`, `linear_damage_law`, `exponential_damage_law`,
`strain_threshold_yield`, `von_mises_state_function`,
`strain_energy_state_function`, `vector_strain_state_function`

**Chemistry** — `curing_rate`, `autocatalytic_reaction`

**Solvers** — `local_newton`, `vector_newton`, `backward_euler`

**Drivers and scalars** — `scalar_stepper`, `tensor_component_stepper_rank1`,
`tensor_component_stepper_rank2`, `constant_scalar`, `props_scalar`,
`scalar_identity_weight`, `scalar_complement_weight`

**Postprocessing** — `property_console_logger`, plus `numerical_diff_checker`
and `property_plot` (not factory-registered; Qt6 optional for the latter)

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

Material names must be unique: they are what input wires, material references
and property lookups resolve against.

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
| `NUMSIM_USE_SYSTEM_TMECH` | OFF | Use an installed tmech instead of fetching it |
| `NUMSIM_INSTALL_BUNDLED_DEPS` | ON | Install the fetched header-only dependencies alongside ours |

### Run tests

```bash
cmake -B build -DNUMSIM_BUILD_TESTS=ON -DCMAKE_CXX_COMPILER=g++-14
cmake --build build
cd build && ctest --output-on-failure
```

### Install and consume

```bash
cmake -B build -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build -j$(nproc)
cmake --install build
```

```cmake
find_package(numsim-materials REQUIRED)
target_link_libraries(app PRIVATE numsim-materials::numsim-materials)
```

With the default configuration the fetched header-only dependencies (tmech,
nlohmann/json) are installed alongside our own headers, so the package compiles
without further setup. Set `NUMSIM_INSTALL_BUNDLED_DEPS=OFF` to supply your own
instead — note that tmech must then be recent enough to provide the Mandel
adaptor used by `solvers/unknown_layout.h`.

## Dependencies

- C++23 (GCC 14+ or Clang 18+)
- [numsim-core](https://github.com/NumSim-Stack/numsim-core) — fetched automatically via CMake
- [tmech](https://github.com/petlenz/tmech) — tensor library; fetched automatically via CMake
  (set `NUMSIM_USE_SYSTEM_TMECH=ON` to use an installed copy instead, or
  `-DFETCHCONTENT_SOURCE_DIR_TMECH=/path/to/tmech` for a local checkout)
- [nlohmann/json](https://github.com/nlohmann/json) — for JSON configuration (optional)
- Eigen — dense linear algebra for `vector_newton`
- Qt6 + QCustomPlot — for live plotting (optional)

## Structure

```
include/numsim-materials/
├── core/              — material_base, material_context, property_engine, traits
├── materials/         — constitutive models, steppers, weights
├── solvers/           — local_newton, vector_newton, backward_euler, rk_integrator
├── umat/              — Abaqus/CalculiX UMAT layer, STATEV mapping, JSON models
├── postprocessing/    — logging, plotting, numerical_diff_checker
├── io/                — json_parameter_converter, json_material_factory
└── default_materials.h — factory registration + macros
```
