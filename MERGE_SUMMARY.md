# Merge Summary: v2 Features into v1 CRTP+Traits Architecture

## Goal

Bring v2's architectural improvements (typed input ports, solver-as-strategy, Kahn's topological sort, enhanced integrity checks) into the existing v1 CRTP+Traits material framework — without replacing v1 or changing constructor signatures. After integration, the v2 directory was removed.

## Constraints

- CRTP pattern (`material_base<Derived, Traits>`) preserved
- `input_parameter_controller` preserved
- Derived material constructor signatures unchanged (variadic forwarding `Args&&...`)
- All configuration through the parameter interface

---

## Phase 1: Typed Input Infrastructure

### New types in `type_traits_materials.h`

| Type | Purpose |
|------|---------|
| `input_wire_base` | Type-erased base for deferred input wiring at `finalize()` |
| `connection_source` | Parses `"material::property"` strings for config-driven wiring |
| `input_property<T, Traits>` | Typed read-only input to another material's `property<T>`. Wired at finalize via `dynamic_cast`-checked `wire()`. Asserts on null access. |
| `input_history<T, Traits>` | Typed read-only input to another material's `history_property<T>`. Provides `old_value()` / `new_value()`. |

### New methods on `material_interface<Traits>`

| Method | Purpose |
|--------|---------|
| `add_input<T>(material, property, EdgeKind)` | Register a typed input for deferred wiring |
| `add_input_history<T>(connection_source, EdgeKind)` | Register a typed history input |
| `wire_inputs()` | Wire all typed inputs to source properties (called at finalize) |
| `typed_inputs()` | Access input wires for dependency graph building |

### History management on `property_base`

| Method | Purpose |
|--------|---------|
| `is_history()` | Virtual — returns `true` for `history_property`, `false` for `property` |
| `commit()` | Virtual — `old = new` for history properties, no-op otherwise |
| `revert()` | Virtual — `new = old` for history properties, no-op otherwise |

Added `static_assert(!std::is_reference_v<T>)` on `history_property` to prevent reference-type instantiation. Added const-ref constructor and fixed `make_history_property` to use `std::decay_t<T>`.

---

## Phase 2: Solver-as-Strategy

### New file: `solver.h`

All types templated on scalar type `T`:

| Type | Purpose |
|------|---------|
| `nonlinear_system<T>` | Struct with `residual(T) -> T`, `jacobian(T) -> T`, `accept(T)` closures |
| `solver_base<T>` | Abstract solver strategy with virtual `solve(nonlinear_system<T>&, T initial_guess)` |
| `backward_euler<T>` | Damped Newton-Raphson with configurable tolerance, max iterations, zero-Jacobian guard |
| `forward_euler<T>` | Explicit single-step solver |

### Integration into material_interface

| Method | Purpose |
|--------|---------|
| `set_solver(solver_base<value_type>*)` | Inject a solver into the material |
| `solver()` | Access the assigned solver |
| `needs_solver()` | Virtual — materials override to signal they need a solver |

---

## Phase 3: Kahn's Topological Sort in `material_context`

### `material_context<Traits>` enhancements

| Feature | Detail |
|---------|--------|
| `compute_execution_order()` | Kahn's algorithm at material level. Only `Global` edges affect ordering; `Local` edges excluded. Edges sourced from both typed inputs and `wire_consumers` property dependencies. Deduplicated via helper. |
| `update()` | Runs all materials in topological order via virtual dispatch. Single dispatch path — no dual mechanism. |
| `commit()` / `revert()` | Iterates only `is_history()` properties (filtered at finalize). |
| `add_solver()` / `find_solver()` | Solver storage and lookup by name. |
| `check_integrity()` | Warns on unwired typed inputs and materials that `needs_solver()` but have none. |
| `dump()` | Prints execution order. |
| Destructor | Wraps `finalize()` in `try-catch` to prevent throwing from destructor. |

### Dropped: `Algebraic` EdgeKind

Removed unused `EdgeKind::Algebraic`. Only `Global` and `Local` remain.

---

## Phase 4: Material Updates

### All v1 materials: unified virtual `update()`

`material_interface::update()` declared `virtual void update() noexcept {}`. All materials now override it:

| Material | Change |
|----------|--------|
| `scalar_stepper` | Added `override`, removed manual `old = new` from `update()` (commit/revert now via `material_context`) |
| `autocatalytic_reaction` | Added `override`, removed manual `old = new` from `update_state()` |
| `scalar_euler_backward` | Added `override` |
| `tensor_component_stepper` | Added `override`, added `noexcept` |
| `linear_elasticity` | No `update()` method — uses `wire_updates` callbacks only |
| `weighted_sum` | Already had `override`, added `noexcept` |
| `scalar_identity_weight` | Already had `override`, added `noexcept` |
| `scalar_complement_weight` | Already had `override`, added `noexcept` |

### `autocatalytic_reaction`: nonlinear_system integration

Both solver-strategy and legacy paths now go through `nonlinear_system<value_type>`:

```
make_nonlinear_system()
  -> residual:  compute_residual(dz, z_n, dt)
  -> jacobian:  compute_jacobian(dz, z_n, dt)
  -> accept:    m_his.new_value() += dz
```

| Path | Flow |
|------|------|
| Solver-strategy | `update()` -> `make_nonlinear_system()` -> `m_solver->solve(sys, forward_euler_guess())` |
| Legacy callbacks | `update_nonlinear_function()` -> `make_nonlinear_system()` -> `sys.residual(dz)` / `sys.jacobian(dz)` -> store in properties for `scalar_euler_backward` |

Core computation extracted into member functions: `compute_residual()`, `compute_jacobian()`, `forward_euler_guess()`. All numeric literals use `value_type{}`.

### `scalar_stepper`: bug fix

Changed `m_his` from value member (`history_property<value_type> m_his{0}`) to reference (`history_property<value_type>& m_his`). The value member was a copy — updates modified the local copy, not the registry property.

### `scalar_weight`: connection_source usage

Changed parameter from `connection_source` type (not in variant) to `std::string` with `connection_source::parse()`.

---

## Phase 5: Lifetime Safety Fixes

| Fix | Detail |
|-----|--------|
| `material_property_info::name_` | Changed from `std::string_view` to `std::string` — prevents dangling when info outlives material |
| `material_interface::m_name` | Changed from `const std::string&` to `std::string` — prevents dangling reference to parameter handler internals. Initialized directly from `m_parameter_handler.get<std::string>("name")`. |

---

## Phase 6: Code Review Fixes

| # | Issue | Fix |
|---|-------|-----|
| 1 | Unsafe `static_cast` in `wire()` | Replaced with `dynamic_cast` + null check + throw on type mismatch |
| 3 | Null deref in `input_property::get()` | Added `assert()` before dereference |
| 6 | `if constexpr` hiding reference-type bug | Replaced with `static_assert(!is_reference_v<T>)` |
| 7 | `m_history_properties` collected all properties | Added `is_history()` virtual, filter at finalize |
| 8 | Duplicate edges in Kahn's sort | Extracted `add_edge` helper with dedup for both edge sources |
| 9 | Zero Jacobian in `backward_euler` | Added `abs(j) < 1e-30` guard |
| 10 | Destructor calls throwing `finalize()` | Wrapped in `try-catch(...)` |

---

## Phase 7: Cleanup

- Removed `v2/` directory (all features integrated into v1)
- Removed unused `EdgeKind::Algebraic`

---

## Test Coverage

6 test functions added to `main.cpp`:

| Test | What it exercises |
|------|-------------------|
| `test_context_update` | `material_context::create()`, `finalize()`, Kahn's sort, `ctx.update()` with strain stepper -> linear elasticity chain |
| `test_scalar_weight` | `scalar_identity_weight` with `input_history` wiring via `add_input_history` |
| `test_solver_strategy` | `backward_euler<T>` Newton-Raphson (x^2-4=0 -> x=2), `forward_euler<T>`, solver management in context |
| `test_diagnostics` | `commit()` / `revert()` cycle, `dump()`, execution order for independent materials |
| `test_curing_with_context` | Full 4-material curing simulation via `material_context` (legacy path) |
| `test_curing_solver_strategy` | Curing simulation with `backward_euler<T>` injected via `set_solver()` (strategy path) |

---

## Final File Layout

```
include/numsim-materials/
  type_traits_materials.h    — property_base, property<T>, history_property<T>,
                               input_wire_base, connection_source,
                               input_property<T>, input_history<T>, EdgeKind
  material_base.h            — material_interface<Traits>, material_base<Derived, Traits>
                               (CRTP preserved, solver + typed input support added)
  solver.h                   — nonlinear_system<T>, solver_base<T>,
                               backward_euler<T>, forward_euler<T>
  material_context.h         — material_context<Traits> (create, finalize, update,
                               commit/revert, Kahn's sort, solver management)
  material_property_info.h   — material_property_info (string-owned name)
  check_material_integrity.h — check_material_integrity (unchanged)
  executioner_plan.h         — PropertyGraph, executioner_plan, executioner (unchanged)
  property_graph.h           — property_graph (Boost BGL, unchanged)
  scalar_stepper.h           — scalar_stepper<Traits> (reference fix, override)
  autocatalytic_reaction.h   — autocatalytic_reaction<Traits> (nonlinear_system, dual path)
  scalar_euler_backward.h    — scalar_euler_backward<Traits> (override, legacy solver)
  linear_elasticity.h        — linear_elasticity<Traits> (unchanged)
  tensor_component_stepper.h — tensor_component_stepper<Rank, Traits> (override)
  scalar_weight.h            — scalar_identity_weight, scalar_complement_weight
                               (connection_source::parse, override)
  weighted_sum.h             — weighted_sum<Traits> (override)
  property_console_logger.h  — property_console_logger<Traits> (unchanged)
  registery.h                — type-erased factory (unchanged)
  trace_scope.h              — (unchanged)
  scalar_euler_forward.h     — (unchanged)
```
