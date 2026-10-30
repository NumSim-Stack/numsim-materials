# Writing a material

A material is a node of the property graph. It **consumes** properties of other materials (inputs), **produces**
properties (outputs), and fills its outputs in **callbacks**. The engine runs each callback once per update,
in an order derived from what the callbacks declare they read. It runs a callback only once when everything
the callback reads is constant.

This page builds one material, a 1D thermoelastic bar, and introduces each declaration where it is needed:

```
stress = E (strain − α (T − 20)),    energy = stress² / (2E)
```

## 1. Inputs, outputs, parameters

```cpp
template <typename Traits>
class thermo_bar final : public nm::material_base<thermo_bar<Traits>, Traits> {
public:
  using base = nm::material_base<thermo_bar<Traits>, Traits>;
  using value_type = typename base::value_type;
  using input_parameter_controller = typename base::input_parameter_controller;

  template <typename... Args>
  explicit thermo_bar(Args &&...args)
      : base(std::forward<Args>(args)...),
        m_E(base::template get_parameter<value_type>("E")),
        m_alpha(base::template get_parameter<value_type>("alpha")),
        m_strain(base::template add_input<value_type>("strain", "state", nm::EdgeKind::Global)),
        m_temperature(base::template add_input<value_type>("temperature", "value", nm::EdgeKind::Global)),
        m_stiffness(base::template add_output<value_type>("stiffness")),
        m_thermal(base::template add_output<value_type>("thermal_strain")),
        m_stress(base::template add_output<value_type>("stress")),
        m_energy(base::template add_output<value_type>("energy")) {
    // callbacks: section 2
  }

  static input_parameter_controller parameters() {
    input_parameter_controller para{base::parameters()};
    nm::required<value_type>(para, "E");
    nm::required<value_type>(para, "alpha");
    return para;
  }

private:
  value_type const &m_E, &m_alpha; // references: set_parameter() reaches them
  nm::input_property<value_type, nm::property_traits> const &m_strain, &m_temperature;
  value_type &m_stiffness, &m_thermal, &m_stress, &m_energy;
};
```

- **Parameters:**
  - `parameters()` declares them, and the base class checks them at construction.
  - Hold them **by reference**: `set_parameter()` writes in place, and the engine recomputes the constants after it.
- **Inputs:**
  - `add_input<T>(material, property, edge)` creates one.
  - It is wired at `finalize()`, and `get()` reads it.
  - A **global** edge orders this material after the source.
  - A **local** edge reads the value without ordering: the previous pass's value, or one a solver drives (section 6).
- **Outputs:**
  - `add_output<T>(name)` creates one and returns a reference to the value.
  - `add_history_output<T>(name)` creates an output with an old and a new value, which `commit()` and `revert()` manage.

Outputs created with `add_output` and `add_history_output` have no callback yet.

## 2. One callback per computation, with its reads

```cpp
    base::add_callback(&thermo_bar::compute_stiffness, m_stiffness).reads();
    base::add_callback(&thermo_bar::compute_thermal, m_thermal).reads(m_temperature);
    base::add_callback(&thermo_bar::compute_stress, m_stress, m_energy).reads(m_strain, m_stiffness, m_thermal);
```

`add_callback(&callback, outputs...)` binds one member function to the outputs it fills. The outputs can be given as:
- references;
- pointers (a null pointer is skipped);
- ranges of pointers.

The handle then declares what the callback reads besides its parameters:

| declaration | the callback reads | the engine |
|---|---|---|
| `.reads()` | parameters only | runs it once, and again after `set_parameter()` |
| `.reads(inputs...)` | these inputs | orders it after their sources; runs it once while they are constant |
| `.reads(..., own outputs...)` | outputs of other callbacks of this material | orders it after those callbacks |
| `.reads_all_inputs()` | every input the material has at `finalize()` | as `.reads(all inputs)` |
| `.stateful()` | also state of its own, or acts outside its outputs | runs it on every update |

The callback is the node of the graph. Order and constness follow from the reads:
- **Order:** `compute_stress` runs after the stiffness and thermal callbacks, because it reads their outputs. It runs after the strain source, because it reads the strain.
- **Constness:** with a constant temperature, `stiffness` and `thermal_strain` are computed once, while `stress` follows the strain on every update.

One computation, one callback. Splitting a material into callbacks pays off because:
- a constant part is computed once;
- a varying part reruns alone;
- `update_property()` on one output runs only what that output's callback reads.

## 3. History

```cpp
m_kappa(base::template add_history_output<value_type>("kappa")),
...
base::add_callback(&my_yield::compute, m_kappa, m_is_yielding).reads(m_eps_eq);
```

- **Reading history:** a callback reads `m_kappa.old_value()` (the committed value) and writes `m_kappa.new_value()`.
- **Constness:** a history output is never constant.
- **Inputs on history:** an input on another material's history output is an `add_input_history`, and has both values.

## 4. State the reads cannot see: `.stateful()`

A callback has state of its own when it either:
- reads something that is not a parameter, an input or one of its outputs; or
- has an effect besides filling its outputs (it adds to its last value, appends a row, prints).

Such a callback must say so:

```cpp
base::add_callback(&tensor_component_stepper::update, m_tensor).reads().stateful(); // adds to its last value
base::add_callback(&property_recorder::update, tick).reads(m_scalar_inputs).stateful(); // a row per update
```

Without `.stateful()`, a stepper that reads nothing would run once. A recorder of constant sources would also record once.

## 5. Checking the declarations

- **The debug checks** are set by the CMake option `NUMSIM_MATERIALS_DEBUG_CHECKS`:
  - `AUTO` turns them on except in `Release`, `RelWithDebInfo` and `MinSizeRel`; `ON` and `OFF` force them.
  - CMake defines the setting for every translation unit using the target, independently of `NDEBUG`. Without CMake, the macro of the same name follows `NDEBUG` and must be the same in every translation unit of a program.
- **Verification:** with the checks on, every constant callback reruns on each update and throws when an output changes. That catches an incomplete read list and hidden state. `ctx.verify_constants(false)` turns this off, for example in call-counting tests.
- **Undeclared inputs:** with the checks on, a callback that reads an input it did not declare throws on the first run that reads it, constant or not. A read in a branch that has not run yet goes unnoticed until it runs:

  ```
  reads(): 'bar::stress' reads 'strain::state', an input it did not declare
  ```

  `ctx.check_reads(false)` turns this off.
- **Not checked:** reads of the material's own outputs are plain references, which the check cannot see. Declare them carefully (section 7).
- **After a throw:** the context is partly updated, so it is failed. `update()`, `update_property()` and `commit()` throw until the host calls `revert()`, which restores the history and clears the failure.
- **From inside a callback:**
  - `update()` throws.
  - `update_property()` re-evaluates the upstream callbacks, constants included, and leaves the running update untouched (as `numerical_diff_checker` does).
  - A callback must not suspend or move to another thread while it runs: with the debug checks on, the read check's scope is per thread.
- **Inspecting the graph:**
  - `ctx.dump()` prints the execution order and marks the constant outputs;
  - `ctx.is_constant(material, property)` asks for one output.

## 6. Solvers and references

- **A solver that reruns a function material's callbacks:** it reads them over local edges and calls `update_source()`, as `backward_euler` does with the residual. It declares that with `.drives(inputs...)`:

  ```cpp
  base::add_callback(&backward_euler::update, m_delta).reads(m_residual, m_jacobian).drives(m_residual, m_jacobian);
  ```

  The engine then runs the solver after everything the driven callbacks read globally (the time step, the temperature), whatever order the deck creates them in. Reads of the solver's own outputs (a function reading the current unknown) add no edge: the solver fills those itself.
- **What a driven callback must be:** a repeatable trial evaluation. Every call computes from the same committed history and its current inputs, and writes outputs only. It must not commit, count or record, because the solver calls it once per iteration.
- **Driving any output of a group:** an output filled by another output's callback reruns that callback through `update_source()`.
- **A fixed point over a local edge:** a node that reads another over a local edge sees the previous pass. A material that measures how far two passes disagree declares that output with `converges()`, with the tolerance and the pass limit as parameters:

  ```cpp
  base::converges(m_residual, base::template get_parameter<value_type>("tolerance"), base::template get_parameter<int>("max_passes"));
  ```

  At `finalize()` the engine forms an iteration group: the callbacks downstream of the consumer of a local edge and upstream of the residual. `update()` runs:
  - first, the callbacks the group reads, once;
  - then the group, until |residual| ≤ tolerance (a NaN never converges);
  - last, the rest, once. A recorder or a plot therefore writes one row per update.

  After `max_passes` passes, `update()` throws `cutback_error` and names the residual. `update_property()` on a member runs its upstream once, without iterating. These rules throw at `finalize()` and name the node:
  - `converges()` with no local edge behind the residual;
  - a `.stateful()` callback inside a group;
  - a second residual over the callbacks of a group;
  - a tolerance below zero or `max_passes` below one.

  `mean_field_composite_local_nonlinear` declares its `residual` this way.
- **`material_ref`:** a material reached through `material_ref` (a `local_newton` solver object) is not an input. The engine:
  - treats what the target reads as reads of every callback of the referencing material;
  - keeps those callbacks varying unless every output of the target is constant (a target with history, state or a bare output never is);
  - orders the target's bare outputs after the referencing callbacks, which fill them. A bare output has no callback and no writer, like the solver's `iterations`. A referencing callback that reads it is not ordered before it.

## 7. Common mistakes

| mistake | symptom | fix |
|---|---|---|
| a read missing from `.reads(...)` | wrong order, a stale value; with the other reads constant, a value frozen after the first update (release); debug: "reads an input it did not declare" | declare it |
| an own output read missing | wrong order: the value of the previous update | declare the own output |
| a parameter copied into a member | `set_parameter()` has no effect | hold it by reference |
| state in a member (a counter, a last value) | debug: "changed although constant" | `.stateful()`, or make it a history output |
| `.reads_all_inputs()` on a callback that does not use a varying input | never folded | `.reads(...)` with the inputs it uses |

## 8. The deprecated declarations

These still compile, with a deprecation warning, until they are removed:

| deprecated | instead |
|---|---|
| `add_output<T>(name, &callback)` | `add_output<T>(name)` + `add_callback(&callback, output).reads(...)` |
| `add_history_output<T>(name, &callback, initial)` | `add_history_output<T>(name, initial)` + `add_callback(...)` |
| `constant_output(output[, {inputs}])` | `.reads(inputs...)` on the callback that fills it |
| `written_by(output, writer)` | pass the output to the writer's `add_callback(&callback, writer, output)` |
| `runs_after(output, earlier)` | `.reads(earlier)` on the callback of `output` |
| `.constant()` on a callback handle | `.reads_all_inputs()` |

A callback bound the old way has no declared reads. The engine then orders it after every input of its material, never folds it unless `constant_output` says so, and does not check its reads.
