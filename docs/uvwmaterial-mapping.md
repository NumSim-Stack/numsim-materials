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
| `incremental_saint_venant_kirchhoff`, `hencky_strains_linear_*` | — | | #113 |
| `small_strain_rule_of_mixture_composite` (`kernal` voigt / reuss / vrh) | `rule_of_mixture` | `phases`, `fractions`, `kernel` voigt / reuss / hill, `strain_source` | #130 |
| `inclusion` + `eshelby_tensor_solid_*` | `inclusion` | `material`, `volume_fraction` or `fraction_source`, `shape` sphere / cylinder / spheroid / ellipsoid, `aspect_ratio`, `direction`, `semi_axes` | #131, #137 |
| `small_strain_mean_field_composite` (`kernal` dilute / mori_tanaka / scs) | `mean_field_composite` | `matrix`, `inclusions`, `kernel` dilute / mori_tanaka / self_consistent, `strain_source` | #131 |
| `small_strain_short_fibre_composite` | `short_fibre_composite` | `matrix`, `fibre`, `volume_fraction`, `aspect_ratio`, `orientation` (6 components), `precision`, `averaging` | #133 |
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
| `eigenstrain_temperature_material_base`, `thermo_mechanical_elastic` | `thermal_eigenstrain` | `alpha`, `reference_temperature`, `temperature_source` | #135 |
| `eigenstrain_curing_material_base`, `degree_of_cure_dependent_material` | `thermal_eigenstrain` with the cure as source; `inclusion.fraction_source`; `weighted_sum` | | #137 |
| `small_strain_rule_of_mixture_history` | — (Voigt is `weighted_sum` of the phase stresses) | | #117 |
| `small_strain_mean_field_composite_history`, `_plastic` | — | | #116 |
| `small_strain_short_fibre_history_composite` | — | | #118 |
| `*_nonlocal*`, `element_deletion` | — | | #121 |
| `finite_strain_mean_field_composite*` | — | | #122 |
| `linear_isotropic_conductivity` | — | | #109 |
| `conductivity_mean_field_composite`, `eshelby_tensor_conductivity_*` | `eshelby::depolarisation` (utility) | | #129; material #110 |
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
reads the matrix Poisson ratio from the matrix tangent, and connects the parts by name:

```json
{"name": "small_strain_mean_field_composite", "kernal": "mori_tanaka",
 "matrix": {"material": {"name": "linear_elasticity", "parameter": {"E": 1800, "nu": 0.45}}, "volume_fraction": 0.5},
 "inclusions": {"inclusion1": {"material": {"name": "linear_elasticity", "parameter": {"E": 2100, "nu": 0.3}},
                               "volume_fraction": 0.5, "eshelby_tensor": {"type": "sphere", "parameter": {"nu": 0.45}}}}}
```

becomes (`convert_uvwmaterial_card.py card.json --prefix mt`)

```json
{"materials": [
  {"type": "external_strain_source", "name": "strain"},
  {"type": "linear_elasticity", "name": "mt_matrix", "strain_producer_name": "strain", "K": 6000.0, "G": 620.69},
  {"type": "linear_elasticity", "name": "mt_inclusion1_material", "strain_producer_name": "strain", "K": 1750.0, "G": 807.69},
  {"type": "inclusion", "name": "mt_inclusion1", "material": "mt_inclusion1_material", "volume_fraction": 0.5, "shape": "sphere"},
  {"type": "mean_field_composite", "name": "mt", "matrix": "mt_matrix", "inclusions": ["mt_inclusion1"], "kernel": "mori_tanaka", "strain_source": "strain"}]}
```

Differences to know: shear strains are tensor components everywhere; the Eshelby geometry keys `aspec_ratio` and
`direction_vector` become `aspect_ratio` and `direction`; the 9-component `fibre_orientation_tensor` becomes the
6 symmetric components `orientation`; `vrh` is `hill`, `scs` is `self_consistent`; the matrix volume fraction is
implied by the inclusions.
