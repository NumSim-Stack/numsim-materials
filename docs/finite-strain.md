# Finite strain: the Hencky chain

Hencky elasticity is not a material of its own. It is linear elasticity in the
logarithmic strain, composed from producers:

```
F source ─► log_strain ─► elastic_strain ─► linear_stress / linear_elasticity ─► kirchhoff_to_first_piola
            ln V,           ε_e = ln V − Σ ε*    τ = C : ε_e                        P = τ F⁻ᵀ, dP/dF
            d lnV/dF
```

| Node | Reads | Outputs |
|---|---|---|
| `log_strain` | `deformation_gradient_source` | `strain` = ln V = ½ ln(F Fᵀ), `d_strain_d_deformation_gradient` |
| `elastic_strain` | `strain_source`, `eigenstrain_sources` | `strain` = ε − Σ ε*ᵢ |
| `linear_stress` | `strain_source`, `tangent_source` | `stress` = τ = C : ε_e |
| `kirchhoff_to_first_piola` | `stress_source`, `tangent_source`, `strain_source` (the `log_strain`), `deformation_gradient_source` | `stress` = P, `tangent` = dP/dF |

For fixed K and G, `linear_elasticity` replaces `linear_stress` and the
`isotropic_tangent`, and the push-forward's `tangent_source` is that
`linear_elasticity`. A complete JSON deck is in `tests/test_hencky_elasticity.cpp`
(`deck()`).

## What the user is responsible for

**The stress law must be isotropic.** τ = C : ln V is objective only if
C : (Q ln V Qᵀ) = Q (C : ln V) Qᵀ for every rotation Q, that is for an isotropic C.
An inelastic law in log strain, such as J2 between `elastic_strain` and the
push-forward, must be isotropic too. The chain does not check this: a transversely isotropic or mean-field tangent is
accepted and gives results that depend on the rigid rotation. An anisotropic
material needs the Lagrangian chain below.

**Stress and tangent come from the same law.** The push-forward reads τ and
dτ/dε_e through two parameters. Both must name the same stress law: the
`isotropic_tangent` that `linear_stress` reads, or the `linear_elasticity` node
itself. A tangent from a different node leaves P correct and makes dP/dF wrong,
which only shows as slow Newton convergence. The driver's `tangent_error` check
(finite differences of P) catches it in a test of the deck.

**Eigenstrains do not depend on F.** dε_e / d ln V is taken as the identity, so
the eigenstrains (thermal, curing) must not depend on the deformation.

## Anisotropic laws: the Lagrangian chain

With the strain in the reference configuration the chain is objective for any
stress law, isotropic or not:

```
F source ─► seth_hill_strain ─► elastic_strain ─► linear_stress ─► lagrangian_stress_to_first_piola
            E(C), dE/dC,          E_e = E − Σ E*     T = ℂ : E_e      P = F S, S = 2 T : dE/dC, dP/dF
            d²E/dC²
```

| Node | Reads | Outputs |
|---|---|---|
| `seth_hill_strain` | `m`, `deformation_gradient_source` | `strain` = E = (C^{m/2} − I)/m, `d_strain_d_right_cauchy_green`, `d2_strain_d_right_cauchy_green2` (on demand) |
| `lagrangian_stress_to_first_piola` | `stress_source`, `tangent_source`, `strain_source` (the `seth_hill_strain`), `deformation_gradient_source` | `stress` = P, `tangent` = dP/dF |

`m` selects the measure: 0 Hencky (½ ln C), 1 Biot (U − I), 2 Green–Lagrange
(½ (C − I)), −2 the Lagrangian Almansi-type strain ½ (I − C⁻¹); any real `m` is
accepted. The strain node owns the eigendecomposition of C and gives dE/dC and
d²E/dC² with it; the pull-back is the same chain rule for every measure:

- S = 2 T : dE/dC and P = F S;
- dP/dF = dEᵀ : ℂ : dE + dCᵀ : (T : d²E/dC²) : dC + I ⊗ S, where dE = dE/dC : dC and dC = dC/dF.

T is the stress work-conjugate to E. ℂ is constant in the material frame: a fibre
direction is a direction of the reference configuration. For an isotropic law with
m = 0 both chains give the same P and dP/dF. With m = 2 and `linear_stress` the
chain is `saint_venant_kirchhoff`.

d²E/dC² is a rank-6 output computed only while a node reads it (`on_demand()` in
the engine). A deck that reads `strain` without the pull-back skips it, and a
direct `get` of it then throws.

The rule for stress and tangent from the same law holds here as well. Eigenstrains
are reference-configuration strains and must not depend on F.

## Cost

One update with stress and tangent costs the same as the former single
`hencky_elasticity` material (about 1.9 µs, g++-14 -O2): the tensor logarithm and
its derivative dominate, and the extra nodes are negligible.

The Lagrangian chain with m = 0 or 1 costs about 4.4 times the Eulerian one (10.8 µs
against 2.5 µs on the same machine). Most of the difference is the rank-6
d²E/dC² from tmech's `second_derivative()`; the pull-back only contracts it with T.
Green–Lagrange (m = 2) needs no eigendecomposition and costs 2.8 µs.
