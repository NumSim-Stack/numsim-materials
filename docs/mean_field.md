# Mean-field composites

How the composite materials are put together in the property graph: `phase` nodes of produced properties,
tangent producers for the linear schemes (`mean_field_composite_global`, `short_fibre_composite`),
and the localisation `mean_field_strain` with the averages `mean_field_composite_local_linear` / `_nonlinear` for phases that are materials in their own
right. The Eshelby and depolarisation tensors are nodes of the same graph.

The rule behind all of it: **a node consumes properties, never materials.** A composite reaches into no other
object; the engine orders every evaluation, and when a producer changes (a curing matrix, a damaging phase,
an evolving fraction) the consumers follow. A mean-field scheme is a **kernel object** that the nodes hold.

---

## 1. The nodes

| node | consumes | produces |
|---|---|---|
| `poisson_ratio_from_moduli` | parameters `K`, `G` (constant; recomputed after `set_parameter`) | `value` |
| `isotropic_moduli` | `tangent_source` / `tangent_property` | `K`, `G`, `poisson_ratio` |
| `eshelby_sphere`, `eshelby_cylinder`, `eshelby_spheroid`, `eshelby_ellipsoid` | `nu_source` / `nu_property` | `tensor` (rank 4) |
| `depolarisation_sphere`, `_cylinder`, `_spheroid`, `_ellipsoid` | geometry only | `tensor` (rank 2, trace one) |
| `phase` | `tangent_source`, `fraction_source`, optional `eshelby_source`, `depolarisation_source` (`tangent_edge`, `initial_tangent_source`) | `tangent`, `volume_fraction`, `eshelby`, `depolarisation` |
| `mean_field_composite_global` | `phases` (matrix first), `kernel` {object} | `tangent`, `concentration_<phase>` |
| `short_fibre_composite` | `matrix`, `fibre`, `orientation`, `precision`, `averaging` | `tangent`, `concentration_matrix`, `concentration_fibre` |
| `mean_field_strain` | `strain_source` (history), `phases` (matrix first), `kernel` {object} | `strain_<phase>` (history), `concentration_<phase>` |
| `mean_field_composite_local_linear` | `phases` [{`concentration`, `stress`, `tangent`, `fraction`}] | `stress`, `tangent` |
| `mean_field_composite_local_nonlinear` | `phases` [{… , `used_tangent`}] | `stress`, `tangent`, `residual` |
| `scalar_split` | `source` / `property`, `weights` | `value_<i>` = wᵢ · value |

Every `<x>_source` names a material, every `<x>_property` the property on it (defaults: `tangent`, `value`,
`tensor`, `strain`, `stress`). A fixed value is a `constant_scalar`; a value that evolves is whatever node
produces it (`curing_rate` for a fraction, `isotropic_moduli` for a Poisson ratio).

The tangent producers publish no stress: the stress of a linear composite is a `linear_stress` on the
tangent, like for any other tangent (`isotropic_tangent`, a cure-dependent stiffness).

---

## 2. A phase

```json
{"type": "constant_scalar", "name": "Kf", "value": 60}, {"type": "constant_scalar", "name": "Gf", "value": 30},
{"type": "isotropic_tangent", "name": "Cf", "K_source": "Kf", "G_source": "Gf"},
{"type": "constant_scalar", "name": "c", "value": 0.3},
{"type": "poisson_ratio_from_moduli", "name": "nu_m", "K": 3, "G": 1},
{"type": "eshelby_sphere", "name": "S", "nu_source": "nu_m"},
{"type": "phase", "name": "spheres", "tangent_source": "Cf", "fraction_source": "c", "eshelby_source": "S"}
```

A phase bundles a tangent, a volume fraction and, for the mean-field schemes, the Eshelby tensor of its shape
(elasticity) or its depolarisation tensor (conductivity), all as produced properties it republishes. It holds
no material. The fraction must be non-negative; the composite checks that the fractions sum to one. Each
relay is its own callback: a varying fraction (a cure, a damage variable) leaves the tangent and shape relays
constant, and a tangent over a local edge leaves the fraction constant.

The Eshelby node consumes Poisson's ratio of the matrix. Where ν is a constant, `poisson_ratio_from_moduli`
(or a bare `constant_scalar`) produces it once; where it follows a tangent — a curing matrix, an effective
medium — `isotropic_moduli` on that tangent produces it per update and the Eshelby node follows.

---

## 3. Tangent homogenisation (global)

The composite is a tangent producer; the phases are produced tangents. Nothing is evaluated at a wrong
strain because no phase *material* exists in the graph. The homogenisation runs inside one node: `tangent` =
Σ cᵣ ℂᵣ : Aᵣ with the concentrations Aᵣ of its kernel. With constant phases (tangents, fractions and Eshelby
tensors from parameters) the tangent and the concentrations are constant properties, computed once; a phase
whose tangent changes keeps them in the update loop.

```json
{"type": "constant_scalar", "name": "cm", "value": 0.7},
{"type": "phase", "name": "matrix", "tangent_source": "Cm", "fraction_source": "cm"},
{"type": "mean_field_composite_global", "name": "mt", "phases": ["matrix", "spheres"], "kernel": {"type": "mori_tanaka"}},
{"type": "linear_stress", "name": "composite", "tangent_source": "mt", "strain_source": "strain"}
```

### Kernels

A kernel is an object: a deck writes `{"type": ..., ...}`, its parameters are checked like a material's, and
`mean_field::kernel_registry` takes more (`add<K>("name")`). The same kernel classes serve the strain
localisation of §4, so global and linear local agree by construction. With Tᵣ = (I + Sᵣ : ℂ_ref⁻¹ : (ℂᵣ − ℂ_ref))⁻¹
and the matrix as the first phase:

| kernel | reads | concentrations |
|---|---|---|
| `voigt` | fractions | Aᵣ = I |
| `reuss` | tangents | Aᵣ = ℂᵣ⁻¹ : ℂ_R, ℂ_R = (Σ cᵣ ℂᵣ⁻¹)⁻¹ |
| `hill` | tangents | the mean of Voigt and Reuss |
| `dilute` | tangents, `eshelby` of the inclusions | Aᵣ = Tᵣ in the matrix, A_m = (I − Σ cᵣ Tᵣ) / c_m |
| `mori_tanaka` | tangents, `eshelby` of the inclusions | A_m = (c_m I + Σ cᵣ Tᵣ)⁻¹, Aᵣ = Tᵣ : A_m |
| `self_consistent` | tangents; `shapes` of every phase as kernel data | Aᵣ = Tᵣ : (Σ cₛ Tₛ)⁻¹ in the effective medium |

Every kernel satisfies Σ cᵣ Aᵣ = I: the phase strains average to the composite strain.

`short_fibre_composite` is a node of its own: the Mori–Tanaka of one `fibre` phase averaged over its
orientation distribution (`orientation` [a11, a22, a33, a12, a13, a23], Lebedev `precision`, `averaging`
`concentration` | `stiffness`).

Several inclusions: more `phase` entries in `phases`, each with its own Eshelby node. With different shapes
or orientations the Mori–Tanaka tangent loses its major symmetry — a property of the scheme, not of the code.

### Self-consistent

The reference medium is the unknown effective medium, so its Eshelby tensors depend on the result. The kernel
iterates the fixed point inside the node — relaxed steps from the last converged medium (Voigt at first) until
‖ℂ_new − ℂ‖ / ‖ℂ_new‖ ≤ `tolerance` — and evaluates Sᵣ itself from its `shapes`:

```json
{"type": "mean_field_composite_global", "name": "sc", "phases": ["matrix", "spheres"],
 "kernel": {"type": "self_consistent", "relaxation": 0.5, "tolerance": 1e-10, "max_iterations": 100,
            "shapes": {"matrix": {"type": "sphere"}, "spheres": {"type": "spheroid", "aspect_ratio": 3, "direction": [1, 0, 0]}}}}
```

Shapes: `sphere`, `cylinder` (`direction`), `spheroid` (`aspect_ratio`, `direction`), `ellipsoid` (`semi_axes`).
One update gives the converged tangent; it throws with the residual after `max_iterations`. The rule:
**fixed points of the homogenisation converge inside a node, fixed points of the materials go through the
graph** (§5). Sᵣ is the closed form at the Poisson's ratio of the isotropised effective tangent — exact for an
isotropic medium, an approximation for an anisotropic one (aligned fibres).

---

## 4. Strain localisation (local)

When a phase is a material — damage, plasticity, anything with a state — it has to be evaluated at *its*
strain. The composite is split into two nodes around the phase materials:

```
strain ──► mean_field_strain ──strain_matrix──► matrix material ──stress, tangent──► mean_field_composite_local_*
                │             └─strain_spheres─► sphere material ──stress, tangent──┘        ▲
                └──────────────── concentration_matrix, concentration_spheres ───────────────┘
```

- `mean_field_strain` consumes the composite strain (history) and the phases' tangents, fractions and Eshelby
  tensors, and produces `strain_<phase>` = ε_r,ₙ + Aᵣ : Δε (history properties: commit, revert and the UMAT
  state vector carry them) and `concentration_<phase>`. Its `kernel` is the same object as in §3 — any of
  `voigt`, `reuss`, `hill`, `dilute`, `mori_tanaka`, `self_consistent` — so with linear phases one pass equals
  the global composite exactly; the self-consistent kernel converges its own fixed point inside every pass.
  The concentrations and the strains are separate callbacks: with constant phases the concentrations are
  computed once and only the strains every update; a phase read over a local edge recomputes them every pass.
- The phase materials consume those strains like any other strain: `"strain_source": "mt_strain",
  "strain_property": "strain_matrix"`. Every strain consumer takes `strain_property` (default `strain`).
- `mean_field_composite_local_linear` / `_nonlinear` produce `stress` = Σ cᵣ σᵣ and `tangent` = Σ cᵣ ℂᵣ : Aᵣ.
  Per phase they consume four produced properties by role — `concentration`, `stress`, `tangent`, `fraction`,
  each a `"material::property"` — and do not care which scheme produced the concentrations; they check
  Σ cᵣ Aᵣ = I every update, so a fraction wired to the wrong phase shows at once. The nonlinear one also takes
  `used_tangent` for the phases whose tangent changes and reports the `residual` of §5.

**Linear phases.** The tangents are producers, every edge is global, one pass is exact and the result equals
§3 to round-off — with the phase strains and stresses available as properties:

```json
{"type": "mean_field_strain", "name": "mt_strain", "strain_source": "strain", "phases": ["matrix", "spheres"], "kernel": {"type": "mori_tanaka"}},
{"type": "linear_stress", "name": "m", "tangent_source": "Cm", "strain_source": "mt_strain", "strain_property": "strain_matrix"},
{"type": "linear_stress", "name": "s", "tangent_source": "Cf", "strain_source": "mt_strain", "strain_property": "strain_spheres"},
{"type": "mean_field_composite_local_linear", "name": "mt", "phases": [
  {"concentration": "mt_strain::concentration_matrix", "stress": "m::stress", "tangent": "Cm::tangent", "fraction": "matrix::volume_fraction"},
  {"concentration": "mt_strain::concentration_spheres", "stress": "s::stress", "tangent": "Cf::tangent", "fraction": "spheres::volume_fraction"}]}
```

**Nonlinear phases.** A phase whose tangent depends on its own strain closes a cycle: the concentrations
need the algorithmic tangent, the tangent needs the local strain. The phase node reads the tangent over a
local edge, starting from the elastic one:

```json
{"type": "phase", "name": "matrix", "tangent_source": "damage", "tangent_property": "damaged_tangent",
 "tangent_edge": "local", "initial_tangent_source": "Cm", "fraction_source": "cm"},
{"type": "mean_field_strain", "name": "mt_strain", "strain_source": "strain", "phases": ["matrix", "spheres"], "kernel": {"type": "mori_tanaka"}},
{"type": "linear_elasticity", "name": "elastic", "strain_producer_name": "mt_strain", "strain_property": "strain_matrix", "K": 3, "G": 1},
{"type": "von_mises_state_function", "name": "state", "strain_source": "mt_strain", "strain_property": "strain_matrix"},
{"type": "strain_threshold_yield", "name": "yield", "state_source": "state", "kappa_critical": 0.002},
{"type": "exponential_damage_law", "name": "law", "yield_source": "yield", "kappa_0": 0.002, "beta": 30},
{"type": "isotropic_damage", "name": "damage", "elastic_source": "elastic", "damage_source": "law", "state_source": "state", "yield_source": "yield"},
{"type": "linear_stress", "name": "s", "tangent_source": "Cf", "strain_source": "mt_strain", "strain_property": "strain_spheres"},
{"type": "mean_field_composite_local_nonlinear", "name": "mt", "phases": [
  {"concentration": "mt_strain::concentration_matrix", "stress": "damage::damaged_stress", "tangent": "damage::damaged_tangent",
   "fraction": "matrix::volume_fraction", "used_tangent": "matrix::tangent"},
  {"concentration": "mt_strain::concentration_spheres", "stress": "s::stress", "tangent": "Cf::tangent", "fraction": "spheres::volume_fraction"}]}
```

One `update()` is one pass with the tangent of the previous pass; `residual` is the largest relative
difference between the `used_tangent` a phase node carried into the localisation and its material's current
`tangent`. The whole
damage chain is evaluated at the matrix strain, with its own history. Isotropic damage keeps Poisson's ratio,
so the Eshelby node reads ν of the undamaged tangent.

The published tangent Σ cᵣ ℂᵣ : Aᵣ is first order in Δε: exact as the step vanishes, not a central
difference across a damage kink. The same holds for the localisation itself — a Reuss composite equalises the
stress *increments* of the algorithmic tangents, so its phase stresses agree to first order in the step while
the phases harden. A softening phase in series (damage under Reuss or Hill) is a snap-back problem: there is no
stable equal-stress path and the result depends on the step — a property of the problem, which the passes do
not hide.

### Short fibres at their local strains

`short_fibre_composite` (§3) averages one fibre over its orientations and only knows tangents. When the
fibres are materials (damage), every orientation is a phase of its own with its own history. The deck entry
`orientation_phases` writes these phases: it is expanded into plain entries before the graph is built, one
fibre phase per node of the half Lebedev rule fitted to the orientation tensor (an antipodal pair is one
direction carrying both weights):

```json
{"type": "orientation_phases", "name": "sf", "strain_source": "strain", "matrix": "matrix",
 "matrix_stress": "elastic_m::stress", "fraction_source": "c", "orientation": [0.7, 0.25, 0.05, 0, 0, 0], "precision": 11,
 "kernel": {"type": "mori_tanaka", "eshelby": {"type": "eshelby_spheroid", "aspect_ratio": 15, "nu_source": "moduli", "nu_property": "poisson_ratio"}},
 "phase": {"tangent_source": "Cf"},
 "per_direction": [
   {"type": "linear_elasticity", "name": "fe_{q}", "strain_producer_name": "sf_strain", "strain_property": "strain_sf_{q}", "K": 166666.67, "G": 76923.08},
   ...,
   {"type": "isotropic_damage", "name": "fibre_{q}", "elastic_source": "fe_{q}", ...}],
 "stress": "fibre_{q}::damaged_stress", "tangent": "fibre_{q}::damaged_tangent", "matrix_tangent": "elastic_m::tangent"}
```

It becomes `sf_fractions` (a `scalar_split` of `c` by the node weights), per direction q the phase `sf_<q>`
(from the `phase` template, fraction `sf_fractions::value_<q>`) and a copy of `per_direction` with `{q}` → q,
then `sf_strain` (`mean_field_strain`, the matrix first) and `sf` (`mean_field_composite_local_linear`; the
stress and tangent of phase q from the `stress` / `tangent` templates, which default to the phase node's
tangent). The phase material of direction q reads `strain_sf_<q>` of `sf_strain`.

The kernel is any of §3. For `dilute` and `mori_tanaka`, `eshelby` is the template of an Eshelby node that is
emitted per direction along it. For `self_consistent`, `shape` is the kernel shape of a fibre, placed along every
direction, and `matrix_shape` (default `sphere`) is the shape of the matrix. With elastic fibres one pass is the
global composite over the same phases; Mori–Tanaka equals `short_fibre_composite` with `averaging`
`concentration`.

The phases localise with their initial tangents (the `phase` template's `tangent_source`), so the localisation
is linear and one pass is exact, while the averaged stress and tangent are the damaged ones.

With `"localisation": "nonlinear"` the fibres localise with their current tangents: the `phase` template reads
the fibre material's tangent over a local edge, and the composite becomes
`mean_field_composite_local_nonlinear` with `used_tangent` = `sf_<q>::tangent` per fibre (`matrix_used_tangent`
for a nonlinear matrix). The passes run to the fixed point of §5:

```json
"localisation": "nonlinear",
"phase": {"tangent_source": "fibre_{q}", "tangent_property": "damaged_tangent", "tangent_edge": "local", "initial_tangent_source": "Cf"}
```

Under Mori–Tanaka a softening fibre then attracts strain, so it damages more than in the linear localisation.

---

## 5. Who runs the fixed point

Local edges make every pass acyclic. The engine runs the fixed point inside `update()`:

- `mean_field_composite_local_nonlinear` declares its `residual` with `converges()`. Its parameters are
  `tolerance` (default 1e-10) and `max_passes` (default 200).
- The engine reruns the iteration group until the residual is at most the tolerance. The group is the
  callbacks from the phase that reads its tangent over the local edge to the composite.
- The callbacks before the group (the strain source, the constant tangents) and after it (recorders, plots)
  run once per update.
- After `max_passes` passes, `update()` throws `cutback_error`, and the host cuts the step back.
- Every host converges without further setup: the UMAT, the drivers, the report and a test calling
  `ctx.update()`. A solver's Newton loop re-evaluates the material each iteration, so the returned stress is
  consistent for the given increment.
- Deprecated: the material point's `residual_source` (`residual_property` of `material_point_evaluator`,
  `"point": {"residual_source": ...}` in a report) with `residual_tolerance` and `max_passes` repeats the whole
  update. With a converging composite it finds the residual converged after the first update. In a report, set
  `"tolerance"` and `"max_passes"` on the composite instead (on the `orientation_phases` entry for a
  generated one).

It is a Picard iteration on the tangents — linear convergence, fine for damage with moderate softening; near
snap-back it can stall, which the residual then shows. The self-consistent kernel needs none of this: its own
fixed point converges inside the node (§3).

---

## 6. Conductivity

The same wiring with rank-2 quantities: `depolarisation_*` nodes are the Eshelby tensors of conductivity,
republished by `phase` as `depolarisation` (#110 for the composite nodes).

---

## 7. Reports and tests

`docs/materials/mean_field_composite_global` (dilute, Mori–Tanaka, self-consistent, unidirectional fibres),
`mean_field_composite_global_rule_of_mixture`, `short_fibre_composite`, `short_fibre_damage` (`orientation_phases` with
damaging fibres, nonlinear against linear localisation), `mean_field_damage` (the local pair with a
damaging matrix); tests `test_mean_field_kernels`, `test_mean_field_composite_global`, `test_rule_of_mixture`,
`test_short_fibre_composite`, `test_mean_field_local`, `test_mean_field_damage`, `test_orientation_phases`,
`test_json_expansions`, `test_scalar_split`,
`test_eshelby_materials`, `test_depolarisation_materials`, `test_scalar_producers`. The uvwmaterial cards
map onto these nodes through `tools/convert_uvwmaterial_card.py` (see `uvwmaterial-mapping.md`).
