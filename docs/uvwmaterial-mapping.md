# uvwmaterial → numsim-materials

How the classes of [petlenz/uvwmaterial](https://github.com/petlenz/uvwmaterial) map onto numsim-materials
(epic #94). uvwmaterial built one object tree per material with positional `std::vector` parameters and a nested
JSON card; numsim-materials builds a flat graph of named materials with named parameters, where every connection
is a `*_source` parameter. `tools/convert_uvwmaterial_card.py` converts the old cards for the supported types.

## Types

| uvwmaterial | numsim-materials | parameters | status |
|---|---|---|---|
| `linear_elasticity` (E, ν) | `linear_elasticity` | `K`, `G` (= E/(3(1−2ν)), E/(2(1+ν))), `strain_producer_name`, optional `eigenstrain_source` | existing; eigenstrain #135 |
| `linear_elastic_incremental`, `incremental_solid_material` | `incremental_linear_elasticity` | `strain_source`, `tangent_source` | #134 |
| `saint_venant_kirchhoff` (K, G) | `saint_venant_kirchhoff` | `K`, `G`, `deformation_gradient_source` | existing |
| `neo_hookean_incompressible` | — | | #112 (formulation open) |
| `hencky_strains_linear_elastic` (K, G), `_thermo_elastic`, `_thermo_chemo_elastic` | `hencky_elasticity` (+ `thermal_eigenstrain`) | `tangent_source` (an `isotropic_tangent` of produced K, G), `deformation_gradient_source`, optional `eigenstrain_source`; outputs P and dP/dF instead of S(ln C) and dS/dC | #141 |
| `incremental_saint_venant_kirchhoff` | — | | #113 |
| `small_strain_rule_of_mixture_composite` (`kernal` voigt / reuss / vrh) | `mean_field_composite_global` + `linear_stress` | `phases` (phase nodes), `kernel` {`type`: voigt / reuss / hill}; a tangent producer, the stress is a `linear_stress` on it; nonlinear phases go through `mean_field_strain` (same kernels) and `mean_field_stress` | #130 |
| `eshelby_tensor_solid_sphere`, `_cylinder_x1/x2/x3`, `_cylinder`, `_cylindrical_ellipsoid`, `_general_geometry` | `eshelby_sphere`, `eshelby_cylinder`, `eshelby_spheroid`, `eshelby_ellipsoid` | `nu_source` / `nu_property` (Poisson's ratio as a produced property: `poisson_ratio_from_moduli` of K and G, or `isotropic_moduli` of a tangent), `direction`, `aspect_ratio`, `semi_axes`; output `tensor` | #128 |
| `inclusion` (a phase of any composite) | `phase` | produced properties only: `tangent_source`, `fraction_source` (a `constant_scalar` or a cure), optional `eshelby_source` / `depolarisation_source` (the nodes above); `tangent_edge` local for a nonlinear phase inside `mean_field_strain` | #130, #131, #137, #144 |
| `small_strain_mean_field_composite` (`kernal` dilute / mori_tanaka / scs) | `mean_field_composite_global` + `linear_stress` | `phases` (phase nodes, the matrix first), `kernel` {`type`: dilute / mori_tanaka / self_consistent}; self-consistent carries the `shapes` of every phase in its kernel and converges inside the node | #131 |
| `small_strain_short_fibre_composite` | `short_fibre_composite` + `linear_stress` | `matrix`, `fibre` (phases; the fibre's Eshelby node lies along x1), `orientation` (6 components), `precision`, `averaging` | #133 |
| `small_strain_isotropic_damage` | `isotropic_damage` + `*_state_function` + `strain_threshold_yield` + a damage law | `elastic_source`, `damage_source`, `state_source`, `yield_source` | existing; report #138 |
| propagation law `linear` | `linear_damage_law` | `K` | existing |
| `power_law` | `power_damage_law` | `K`, `m`, `kappa_0` | #138 |
| `exponential_law` | `saturation_damage_law` | `K`, `m`, `kappa_0` | #138 |
| `damage_exponential2` (eps0, epsf) | `exponential_damage_law` | `kappa_0` = eps0, `beta` = 1/(epsf − eps0) | existing |
| `damage_exponential3` | `peerlings_damage_law` | `kappa_0`, `alpha`, `beta` | #138 |
| `damage_exponential1` | — | | dropped (#119) |
| state functions `von_mises_strain`, `strain_energy_state_function`, `vector_strain_state_function`, `von_mises` | the materials of the same names | | existing |
| `strain_based_damage` (crit_val) | `strain_threshold_yield` | `kappa_critical` | existing |
| `autocatalytic_reaction`, `nth_order_reaction` | `autocatalytic_reaction` / `curing_rate` (m = 0 for nth order) | `A`, `E`, `n`, `m` | existing |
| `nth_order_autocatalytic_reaction` | `kamal_sourour_rate` | `A1`, `E1`, `A2`, `E2`, `m`, `n` | #136 |
| `eigenstrain_temperature_material_base`, `thermo_mechanical_elastic` | `thermal_eigenstrain` | `alpha_source` (a produced coefficient), `reference_temperature`, `temperature_source` | #135 |
| `eigenstrain_curing_material_base`, `degree_of_cure_dependent_material` | `thermal_eigenstrain` with the cure as source; `inclusion.fraction_source`; `weighted_sum` | | #137 |
| `small_strain_rule_of_mixture_history` (Voigt) | `weighted_sum` with `constant_scalar` fractions: every phase reads the composite strain, the sum is Σ cᵢσᵢ, Σ cᵢCᵢ (`tests/test_voigt_composite.cpp`) | `terms`, weights | #117; not from JSON yet (no reader for `terms`) |
| `small_strain_rule_of_mixture_history` (Reuss) | — | | #117, after the vector Newton lands |
| `small_strain_mean_field_composite_history`, `_plastic`, `_damage` | `mean_field_strain` + the phase materials + `mean_field_stress` | the strain node produces `strain_<phase>` (history) and the concentrations; the materials consume them (`strain_source`, `strain_property`); the stress node averages what they produce; a nonlinear phase reads its tangent over a local edge (`tangent_edge`) and the point closes the fixed point with `residual` | #116 |
| `small_strain_short_fibre_history_composite` | — | | #118 |
| `*_nonlocal*`, `element_deletion` | — | | #121 |
| `finite_strain_mean_field_composite*` | — | | #122 |
| `linear_isotropic_conductivity` | — | | #109 |
| `eshelby_tensor_conductivity_*` | `depolarisation_sphere`, `depolarisation_cylinder`, `depolarisation_spheroid`, `depolarisation_ellipsoid` (producers, output `tensor`; a phase names one as `depolarisation`) | `direction`, `aspect_ratio`, `semi_axes` | #129 |
| `conductivity_mean_field_composite` | — | | #110 |
| `polymer_matrix_conductivity_material`, `gamm2022` | — | | #123 |
| `plastic_material_*`, `propagation_law_viscous_regularization` | `j2_plasticity`, `drucker_prager_plasticity` exist; Hill and the regularisation | | #120 |
| `tensor_isotropization_general` | `isotropize` (core) | | #129 |
| `sphere_lebedev_rule` | `lebedev` (core, precisions 3–23) | | #129 |
| `uvwmat_functions` (K/E/ν conversions, rotations) | `core/elastic_constants.h` | | #127 |
| Boost `ellint_*` | `core/elliptic_integrals.h` | | #127 |

Base classes (`material_base`, `solid_material_base`, `composite_material_base`, the kernel bases, …) have no
counterpart: the graph replaces the inheritance tree.

## Cards

uvwmaterial nests the phases inside one card and repeats the matrix Poisson ratio in every Eshelby tensor
(`"eshelby_tensor": {"type": "sphere", "parameter": {"nu": 0.45}}`). numsim-materials names every material once,
makes every Eshelby tensor a node consuming the Poisson ratio that `isotropic_moduli` produces from the matrix tangent,
every phase a `phase` node of produced properties, the composite a tangent producer with a `linear_stress`, and connects the parts by name:

```json
{"name": "small_strain_mean_field_composite", "kernal": "mori_tanaka",
 "matrix": {"material": {"name": "linear_elasticity", "parameter": {"E": 1800, "nu": 0.45}}, "volume_fraction": 0.5},
 "inclusions": {"inclusion1": {"material": {"name": "linear_elasticity", "parameter": {"E": 2100, "nu": 0.3}},
                               "volume_fraction": 0.5, "eshelby_tensor": {"type": "sphere", "parameter": {"nu": 0.45}}}}}
```

becomes (`convert_uvwmaterial_card.py card.json --prefix mt`; `model` is the json_model document, `point` the
evaluation point of a report)

```json
{"model": {"materials": [
   {"type": "external_strain_source", "name": "strain"},
   {"type": "constant_scalar", "name": "mt_matrix_material_K", "value": 6000.0},
   {"type": "constant_scalar", "name": "mt_matrix_material_G", "value": 620.69},
   {"type": "isotropic_tangent", "name": "mt_matrix_material_C", "K_source": "mt_matrix_material_K", "G_source": "mt_matrix_material_G"},
   {"type": "isotropic_moduli", "name": "mt_matrix_material_C_moduli", "tangent_source": "mt_matrix_material_C"},
   {"type": "constant_scalar", "name": "mt_inclusion1_material_K", "value": 1750.0},
   {"type": "constant_scalar", "name": "mt_inclusion1_material_G", "value": 807.69},
   {"type": "isotropic_tangent", "name": "mt_inclusion1_material_C", "K_source": "mt_inclusion1_material_K", "G_source": "mt_inclusion1_material_G"},
   {"type": "eshelby_sphere", "name": "mt_inclusion1_shape", "nu_source": "mt_matrix_material_C_moduli", "nu_property": "poisson_ratio"},
   {"type": "constant_scalar", "name": "mt_inclusion1_c", "value": 0.5},
   {"type": "phase", "name": "mt_inclusion1", "tangent_source": "mt_inclusion1_material_C", "eshelby_source": "mt_inclusion1_shape", "fraction_source": "mt_inclusion1_c"},
   {"type": "constant_scalar", "name": "mt_matrix_c", "value": 0.5},
   {"type": "phase", "name": "mt_matrix", "tangent_source": "mt_matrix_material_C", "fraction_source": "mt_matrix_c"},
   {"type": "mean_field_composite_global", "name": "mt_tangent", "phases": ["mt_matrix", "mt_inclusion1"], "kernel": {"type": "mori_tanaka"}},
   {"type": "linear_stress", "name": "mt", "tangent_source": "mt_tangent", "strain_source": "strain"}]},
 "point": {"strain_source": "strain", "stress_source": "mt", "tangent_source": "mt_tangent"}}
```

Differences to know: shear strains are tensor components everywhere; every `eshelby_tensor` becomes its own node consuming the matrix Poisson ratio (for `scs` a shape of the kernel), with `aspec_ratio` and
`direction_vector` as its `aspect_ratio` and `direction`; the 9-component `fibre_orientation_tensor` becomes the
6 symmetric components `orientation`; `vrh` is `hill`, `scs` is `self_consistent`; every volume fraction is a produced property (a `constant_scalar` for a fixed one); the matrix
phase gets the fraction the inclusions leave; the point names the tangent producer (`tangent_source`) because the stress node has no tangent.
