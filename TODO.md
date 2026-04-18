# numsim-materials — TODO

## Next Steps

### 1. Real Solid Material (J2 Plasticity or Isotropic Damage)

First real test of the framework beyond the curing reaction.

**J2 plasticity would need:**
- Outputs: `stress` (tensor2), `tangent` (tensor4), `equivalent_plastic_strain` (history scalar)
- Inputs: `strain` (tensor2) from a stepper or FEM kernel
- Internal: yield function evaluation, return mapping algorithm
- Solver: backward Euler on the plastic multiplier (scalar)
- Consistent tangent: algorithmic tangent from implicit differentiation

**Isotropic damage would need:**
- Outputs: `stress`, `tangent`, `damage` (history scalar)
- Inputs: `strain`
- Internal: equivalent strain, damage evolution law, threshold check
- Simpler than plasticity — good first candidate

### 2. Tensor-Valued Solver

Generalize `backward_euler` from scalar to vector/tensor increments.

- Current: `m_delta` is `value_type` (scalar), residual/jacobian are scalar
- Needed: `m_delta` is `vector<T>` or `tensor<T>`, residual is vector, jacobian is matrix
- The Newton iteration becomes: `dz -= J^{-1} * R` (matrix solve)
- `update_source()` mechanism works the same — just the types change
- Consider templating backward_euler on the increment type

### 3. Consistent Tangent

For global FEM Newton convergence, the solver material needs to produce the
consistent (algorithmic) tangent.

```
C_consistent = dsig/deps - dsig/dz * (dR/dz)^{-1} * dR/deps
```

**New properties on the nonlinear material:**
- `dR_deps` — sensitivity of residual to strain (tensor2 for scalar R)
- `dsig_dz` — sensitivity of stress to internal variable (tensor2 for scalar z)
- `dsig_deps` — partial tangent at fixed z (tensor4)

**New output on the solver:**
- `consistent_tangent` (tensor4) — computed from the above after convergence

All wired via `add_output` / `add_input` with `Local` edges.

### 4. JSON Input Parser

Parse configuration files into `parameter_handler` and construct materials
via the factory.

```json
{
  "materials": [
    {"type": "scalar_stepper", "name": "time", "increment": 10.0},
    {"type": "backward_euler", "name": "solver", "function": "curing", "tolerance": 1e-12},
    {"type": "autocatalytic_reaction", "name": "curing", "A": 4e9, "E": 84000, "solver_name": "solver"}
  ]
}
```

- Parse JSON → populate `parameter_handler` per material
- Call `ctx.create(type, params)` for each entry
- `ctx.finalize()` validates and sorts
- nlohmann/json is already a dependency

### 5. Numerical Differentiation Utility

For materials where analytical derivatives are impractical, provide a
finite-difference utility that works through the property system.

```cpp
// Perturb input, re-evaluate output, compute (f(x+h) - f(x)) / h
auto tangent = numerical_diff(ctx, "material", "stress", "stepper", "strain", h);
```

Uses `update_property()` + save/restore pattern internally.

---

## Cleanup / Technical Debt

### Traits Cleanup

The `material_policy_default` in `traits.h` carries types that are no longer used:

| Type | Status |
|------|--------|
| `FunctionHandler` | Dead — never used by any material or engine |
| `MaterialData` | Dead — `m_material_data` was removed from `material_base` |

Remove these from the policy and from any code that references them.

### property_registry Encapsulation

`property_registry` in `property_registry.h` exposes its internal
`unordered_map` via `data()`. Anyone can mutate it directly. Consider
adding proper insert/find/iterate methods and hiding `data()`.

### Destructor Auto-Finalize

`material_context` destructor calls `finalize()` if not already called,
swallowing all exceptions. Consider:
- Removing auto-finalize (require explicit `finalize()`)
- Or at minimum logging a warning when the destructor triggers finalize

### Include Guard Normalization

Normalize all include guards to use `NUMSIM_MATERIALS_` prefix:

| Current | Should be |
|---------|-----------|
| `MATERIAL_BASE_H` | `NUMSIM_MATERIALS_MATERIAL_BASE_H` |
| `MATERIAL_CONTEXT_H` | `NUMSIM_MATERIALS_MATERIAL_CONTEXT_H` |
| `LINEAR_ELASTICITY_H` | `NUMSIM_MATERIALS_LINEAR_ELASTICITY_H` |
| `SCALAR_STEPPER_H` | `NUMSIM_MATERIALS_SCALAR_STEPPER_H` |
| etc. | |

### add_input API Asymmetry

`add_input<T>` takes two strings (`source_material`, `source_property`),
while `add_input_history<T>` takes a `connection_source`. Unify to one
interface — either both take `connection_source` or both take two strings.

### Duplicate Output Detection

`add_output<T>("stress")` called twice silently reuses the existing property
without rebinding the callback. Should either throw or document this behavior.

---

## Future / Wishlist

- **Forward Euler material** — explicit single-step solver for comparison
- **Adaptive timestepping** — adjust dt based on convergence rate
- **Multi-point evaluation** — materials called at multiple Gauss points with different strains
- **Serialization** — save/load material state for restart
- **Thread safety** — if materials are per-domain and domains run in parallel
- **Python bindings** — expose factory + context + update via pybind11
- **Visualization** — dump property graph to DOT format for graphviz
