# Mean-field composites

How the composite materials are put together in the property graph: `phase` nodes of produced properties,
tangent producers for the linear schemes (`rule_of_mixture`, `mean_field_tangent`, `short_fibre_composite`),
and the strain/stress pair `mean_field_strain` / `mean_field_stress` for phases that are materials in their own
right. The Eshelby and depolarisation tensors are nodes of the same graph.

The rule behind all of it: **a node consumes properties, never materials.** A composite reaches into no other
object; the engine orders every evaluation, and when a producer changes (a curing matrix, a damaging phase,
a self-consistent effective medium) the consumers follow.

---

## 1. The nodes

| node | consumes | produces |
|---|---|---|
| `poisson_ratio_from_moduli` | parameters `K`, `G` (once, at construction) | `value` |
| `isotropic_moduli` | `tangent_source` / `tangent_property` | `K`, `G`, `poisson_ratio` |
| `eshelby_sphere`, `eshelby_cylinder`, `eshelby_spheroid`, `eshelby_ellipsoid` | `nu_source` / `nu_property` (`nu_edge`, `nu_initial`) | `tensor` (rank 4) |
| `depolarisation_sphere`, `_cylinder`, `_spheroid`, `_ellipsoid` | geometry only | `tensor` (rank 2, trace one) |
| `phase` | `tangent_source`, `fraction_source`, optional `eshelby_source`, `depolarisation_source` (`tangent_edge`, `initial_tangent_source`) | `tangent`, `volume_fraction`, `eshelby`, `depolarisation` |
| `rule_of_mixture` | `phases` | `tangent`, `concentration_<phase>` |
| `mean_field_tangent` | `matrix`, `phases`, `kernel`, `relaxation` | `tangent`, `concentration_<phase>`, `residual` |
| `short_fibre_composite` | `matrix`, `fibre`, `orientation`, `precision`, `averaging` | `tangent`, `concentration_matrix`, `concentration_fibre` |
| `mean_field_strain` | `strain_source` (history), `matrix`, `phases`, `kernel` (voigt, reuss, hill, dilute, mori_tanaka) | `strain_<phase>` (history), `concentration_<phase>` |
| `mean_field_stress` | `concentration_source`, `matrix`, `phases`, `matrix_stress_source`, `stress_sources`, `tangent_sources` | `stress`, `tangent`, `residual` |

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
no material. The fraction must be non-negative; the composite checks that the fractions sum to one.

The Eshelby node consumes Poisson's ratio of the matrix. Where ν is a constant, `poisson_ratio_from_moduli`
(or a bare `constant_scalar`) produces it once; where it follows a tangent — a curing matrix, an effective
medium — `isotropic_moduli` on that tangent produces it per update and the Eshelby node follows.

---

## 3. Tangent homogenisation (global)

The composite is a tangent producer; the phases are produced tangents. Nothing is evaluated at a wrong
strain because no phase *material* exists in the graph.

```json
{"type": "constant_scalar", "name": "cm", "value": 0.7},
{"type": "phase", "name": "matrix", "tangent_source": "Cm", "fraction_source": "cm"},
{"type": "mean_field_tangent", "name": "mt", "matrix": "matrix", "phases": ["spheres"], "kernel": "mori_tanaka"},
{"type": "linear_stress", "name": "composite", "tangent_source": "mt", "strain_source": "strain"}
```

Kernels, with Tᵣ = (I + Sᵣ : ℂ_ref⁻¹ : (ℂᵣ − ℂ_ref))⁻¹:

| kernel | reference medium | concentrations | tangent |
|---|---|---|---|
| `dilute` | the matrix | A_m = I, Aᵣ = Tᵣ | ℂ_m + Σ cᵣ (ℂᵣ − ℂ_m) : Aᵣ |
| `mori_tanaka` | the matrix | A_m = (c_m I + Σ cᵣ Tᵣ)⁻¹, Aᵣ = Tᵣ : A_m | ℂ_m + Σ cᵣ (ℂᵣ − ℂ_m) : Aᵣ |
| `self_consistent` | the effective medium of the previous pass | Aᵣ = Tᵣ : (Σ cₛ Tₛ)⁻¹, matrix included | Σ cᵣ ℂᵣ : Aᵣ, relaxed |

`rule_of_mixture` (`voigt`, `reuss`, `hill`) is the same shape without Eshelby tensors;
`short_fibre_composite` averages the Mori–Tanaka of one `fibre` phase over its orientation distribution
(`orientation` [a11, a22, a33, a12, a13, a23], Lebedev `precision`, `averaging` `concentration` | `stiffness`).

Several inclusions: more `phase` entries in `phases`, each with its own Eshelby node. With different shapes
or orientations the Mori–Tanaka tangent loses its major symmetry — a property of the scheme, not of the code.

### The self-consistent cycle

The effective medium needs its own Poisson's ratio, which is an output of the composite — a cycle. It is
closed by one **local edge**:

```json
{"type": "isotropic_moduli", "name": "moduli_eff", "tangent_source": "mt", "tolerance": 0},
{"type": "eshelby_sphere", "name": "S", "nu_source": "moduli_eff", "nu_property": "poisson_ratio", "nu_edge": "local", "nu_initial": 0.35},
{"type": "phase", "name": "matrix", "tangent_source": "Cm", "fraction_source": "cm", "eshelby_source": "S"},
{"type": "phase", "name": "spheres", "tangent_source": "Ci", "fraction_source": "c", "eshelby_source": "S"},
{"type": "mean_field_tangent", "name": "mt", "matrix": "matrix", "phases": ["spheres"], "kernel": "self_consistent", "relaxation": 0.5}
```

`EdgeKind::Local` imposes no ordering: the Eshelby nodes read the ν of the previous pass (`nu_initial` on the
first), the matrix is a phase with its own shape, and one `update()` is one relaxed Picard step with
`residual` = ‖ℂ_new − ℂ‖ / ‖ℂ_new‖. Whoever calls `update()` runs the fixed point — see §5. `tolerance: 0` on
`isotropic_moduli` projects the effective tangent without the isotropy check; for an anisotropic effective
medium (aligned fibres) the closed-form Eshelby tensors at that ν are an approximation.

---

## 4. Strain localisation (local)

When a phase is a material — damage, plasticity, anything with a state — it has to be evaluated at *its*
strain. The composite is split into two nodes around the phase materials:

```
strain ──► mean_field_strain ──strain_matrix──► matrix material ──stress, tangent──► mean_field_stress
                │             └─strain_spheres─► sphere material ──stress, tangent──┘        ▲
                └──────────────── concentration_matrix, concentration_spheres ───────────────┘
```

- `mean_field_strain` consumes the composite strain (history) and the phases' tangents, fractions and Eshelby
  tensors, and produces `strain_<phase>` = ε_r,ₙ + Aᵣ : Δε (history properties: commit, revert and the UMAT
  state vector carry them) and `concentration_<phase>`. Its `kernel` is any of §3 — `voigt` (A = I, reads no
  tangents), `reuss`, `hill` (Aᵣ = ℂᵣ⁻¹ : ℂ_R and the mean with I; no Eshelby tensors), `dilute`, `mori_tanaka`
  — so the rule of mixtures localises through the same pair.
- The phase materials consume those strains like any other strain: `"strain_source": "mt_strain",
  "strain_property": "strain_matrix"`. Every strain consumer takes `strain_property` (default `strain`).
- `mean_field_stress` produces `stress` = Σ cᵣ σᵣ and `tangent` = Σ cᵣ ℂᵣ : Aᵣ from the materials' current
  stresses and tangents (`matrix_stress_source`, `stress_sources`, optional `tangent_sources`;
  `matrix_stress_property` / `matrix_tangent_property` when the matrix material names them differently) and
  the concentrations of the strain node.

**Linear phases.** The tangents are producers, every edge is global, one pass is exact and the result equals
§3 to round-off — with the phase strains and stresses available as properties:

```json
{"type": "mean_field_strain", "name": "mt_strain", "strain_source": "strain", "matrix": "matrix", "phases": ["spheres"]},
{"type": "linear_stress", "name": "m", "tangent_source": "Cm", "strain_source": "mt_strain", "strain_property": "strain_matrix"},
{"type": "linear_stress", "name": "s", "tangent_source": "Cf", "strain_source": "mt_strain", "strain_property": "strain_spheres"},
{"type": "mean_field_stress", "name": "mt", "concentration_source": "mt_strain", "matrix": "matrix", "phases": ["spheres"],
 "matrix_stress_source": "m", "stress_sources": ["s"], "matrix_tangent_source": "Cm", "tangent_sources": ["Cf"]}
```

**Nonlinear phases.** A phase whose tangent depends on its own strain closes a cycle: the concentrations
need the algorithmic tangent, the tangent needs the local strain. The phase node reads the tangent over a
local edge, starting from the elastic one:

```json
{"type": "phase", "name": "matrix", "tangent_source": "damage", "tangent_property": "damaged_tangent",
 "tangent_edge": "local", "initial_tangent_source": "Cm", "fraction_source": "cm"},
{"type": "mean_field_strain", "name": "mt_strain", "strain_source": "strain", "matrix": "matrix", "phases": ["spheres"]},
{"type": "linear_elasticity", "name": "elastic", "strain_producer_name": "mt_strain", "strain_property": "strain_matrix", "K": 3, "G": 1},
{"type": "von_mises_state_function", "name": "state", "strain_source": "mt_strain", "strain_property": "strain_matrix"},
{"type": "strain_threshold_yield", "name": "yield", "state_source": "state", "kappa_critical": 0.002},
{"type": "exponential_damage_law", "name": "law", "yield_source": "yield", "kappa_0": 0.002, "beta": 30},
{"type": "isotropic_damage", "name": "damage", "elastic_source": "elastic", "damage_source": "law", "state_source": "state", "yield_source": "yield"},
{"type": "linear_stress", "name": "s", "tangent_source": "Cf", "strain_source": "mt_strain", "strain_property": "strain_spheres"},
{"type": "mean_field_stress", "name": "mt", "concentration_source": "mt_strain", "matrix": "matrix", "phases": ["spheres"],
 "matrix_stress_source": "damage", "matrix_stress_property": "damaged_stress", "matrix_tangent_property": "damaged_tangent",
 "stress_sources": ["s"], "tangent_sources": ["Cf"]}
```

One `update()` is one pass with the tangent of the previous pass; `mean_field_stress::residual` is the largest
relative difference between the tangent a phase node carried and its material's current one. The whole
damage chain is evaluated at the matrix strain, with its own history. Isotropic damage keeps Poisson's ratio,
so the Eshelby node reads ν of the undamaged tangent; an anisotropic algorithmic tangent would use
`isotropic_moduli` with a local `nu_edge` the same way as §3.

The published tangent Σ cᵣ ℂᵣ : Aᵣ is first order in Δε: exact as the step vanishes, not a central
difference across a damage kink. The same holds for the localisation itself — a Reuss composite equalises the
stress *increments* of the algorithmic tangents, so its phase stresses agree to first order in the step while
the phases harden. A softening phase in series (damage under Reuss or Hill) is a snap-back problem: there is no
stable equal-stress path and the result depends on the step — a property of the problem, which the passes do
not hide.

---

## 5. Who runs the fixed point

Local edges make every pass acyclic; the fixed point is reached by repeating `update()`:

- `material_point_evaluator`: `residual_property` (`"material::property"`), `residual_tolerance`, `max_passes`
  — the graph is updated again per host call until the residual is at most the tolerance; the UMAT and the
  drivers go through it. In a report document: `"point": {..., "residual": "mt::residual", "tolerance": 1e-10,
  "max_passes": 200}`.
- A solver's Newton loop re-evaluates the material each iteration; the material's own loop above still runs
  so that the returned stress is consistent for the given increment.
- In a test: `while (ctx.get<T>("mt", "residual") > tol) ctx.update();`.

It is a Picard iteration on the tangents — linear convergence, fine for damage with moderate softening and
for self-consistent contrasts of order 10–100; near snap-back it can stall, which the residual then shows.

---

## 6. Conductivity

The same wiring with rank-2 quantities: `depolarisation_*` nodes are the Eshelby tensors of conductivity,
republished by `phase` as `depolarisation` (#110 for the composite nodes).

---

## 7. Reports and tests

`docs/materials/rule_of_mixture`, `mean_field_tangent` (dilute, Mori–Tanaka, self-consistent, unidirectional
fibres), `short_fibre_composite`, `mean_field_damage` (the local pair with a damaging matrix); tests
`test_rule_of_mixture`, `test_mean_field_tangent`, `test_short_fibre_composite`, `test_mean_field_local`, `test_mean_field_damage`,
`test_eshelby_materials`, `test_depolarisation_materials`, `test_scalar_producers`. The uvwmaterial cards
map onto these nodes through `tools/convert_uvwmaterial_card.py` (see `uvwmaterial-mapping.md`).
