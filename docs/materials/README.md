# Material reports

One folder per material: `README.md` with equations and implementation (written by hand) and a results
section generated from `report.json`. The generated part has the load cases, checks against closed forms,
figures, the wiring of the graph, and the parameters.

| material | kind |
|---|---|
| [linear_elasticity](linear_elasticity/README.md) | small-strain elasticity |
| [j2_plasticity](j2_plasticity/README.md) | small-strain von Mises plasticity |
| [exponential_isotropic_hardening](exponential_isotropic_hardening/README.md) | saturating hardening (with J2) |

## Running

```sh
cmake -B build -DNUMSIM_MATERIALS_BUILD_REPORTS=ON && cmake --build build --target material_report
python3 reports/make_report.py --tool build/reports/material_report docs/materials/<material>          # figures + README
python3 reports/make_report.py --tool build/reports/material_report --check docs/materials/<material>  # checks only
```

With `NUMSIM_MATERIALS_BUILD_TESTS` every `report.json` is also a ctest (`report_<material>`, checks only).

## report.json

| key | content |
|---|---|
| `include` | files merged first: arrays are appended, local keys win ([`standard_small_strain.json`](standard_small_strain.json) holds the common cases, figures and the tangent check) |
| `parameters` | values; any string `"$name"` in the document is replaced by `parameters.name` |
| `derived` | Python expressions of the parameters, for references and checks (`"E": "9 * K * G / (3 * K + G)"`) |
| `model` | the material graph, a `json_model` document (`{"materials": [...]}`) |
| `point` | `strain_source`, `stress_source`, optional `tangent_source`, `time_source` |
| `record` | extra columns wired to graph properties: `"kappa": "j2::equivalent_plastic_strain"`; a rank-2 tensor gives six columns `name_11 ... name_23`, a component is selected with `"j2::plastic_strain[1,1]"` (1-based) |
| `cases` | load cases, see below |
| `figures` | `{"title", "file", "text", "columns", "panels": [...]}` |
| `checks` | `{"name", "case" or "cases", "value", "expected", "tolerance", "relative", "where"}`: max \|value − expected\| over the steps (relative to max \|expected\| if `relative`) must not exceed `tolerance` |

**Cases.** `control`: `strain`, `uniaxial_stress` (11 prescribed, all other stresses zero) or `mixed` with
`stress_components` (e.g. `["22", "33"]`). The load is `scale × path` (piecewise linear, `steps` per segment, `dt`
per step) times `direction` (strain) and `stress` (stress target), both as tensor components
`[11, 22, 33, 12, 13, 23]`.

**Columns** of every case: `step`, `time`, `load`, `iterations`, `converged`, `tangent_error`,
`strain_11 ... strain_23`, `stress_11 ... stress_23` (tensor components), `p`, `q`, and the `record` columns.
Step 0 is the unloaded state.

**Panels.** `kind`: `line` (default: `x`, `y` as expressions or `{"expr", "label"}`, optional `reference`
curves `{"y", "x", "label"}`, `logy`), `residuals` (Newton residual per iteration of every step) or
`iterations` (per step). `case`, `cases` (a list or `"*"`) select the data.

Expressions are Python with `np`, the parameters, the derived values and the columns of the case as arrays.
