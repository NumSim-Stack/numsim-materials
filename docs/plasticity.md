# Plasticity materials

How the plasticity models are put together after the split of
`small_strain_plasticity`, what each one requires, and which assumptions are
load-bearing.

Current as of `refactor/scalar-newton-split` (PRs #39 and #40).

---

## 1. The layout

```
                    ┌──────────────────────┐
                    │   newton_scalar<T>   │   plain algorithm
                    │  no graph, no props  │   returns {x, converged, iterations}
                    └──────────┬───────────┘
                     used by   │
              ┌────────────────┴────────────────┐
              │                                 │
   ┌──────────▼──────────┐          ┌───────────▼───────────┐
   │   backward_euler    │          │     local_newton      │
   │  MATERIAL, graph-   │          │  MATERIAL, driven by  │
   │  driven             │          │  another material     │
   │  "function" REQUIRED│          │  exposes solve(eval)  │
   └──────────┬──────────┘          └───────────┬───────────┘
              │                                 │  material_ref
   ┌──────────▼──────────┐          ┌───────────▼───────────┐
   │ autocatalytic_      │          │  j2_plasticity        │
   │ reaction, curing    │          │  drucker_prager_      │
   │                     │          │  plasticity           │
   └─────────────────────┘          └───────────────────────┘

   rk_plasticity iterates its own Butcher tableau and uses neither.
```

Three plasticity materials, none of them templated on a yield function any
more:

| material | integrator | yield surface | flow |
|---|---|---|---|
| `j2_plasticity` | backward Euler (radial return) | von Mises cylinder | associative |
| `drucker_prager_plasticity` | backward Euler + apex branch | DP cone | **non**-associative (β ≠ η) |
| `rk_plasticity` | Runge–Kutta, any Butcher tableau | J2 | associative |

---

## 2. Interfaces

### `j2_plasticity`

| parameter | meaning |
|---|---|
| `hardening_source` | material publishing `hardening_stress` and `hardening_modulus` |
| `strain_source` | material publishing `strain` |
| `solver_source` | a `local_newton` |
| `K`, `G` | bulk and shear moduli — the elastic stiffness is built from these |
| `sigma_0` | initial yield stress |

Outputs: `stress`, `tangent`, and the history pair `plastic_strain`,
`equivalent_plastic_strain`.

### `drucker_prager_plasticity`

Same, minus `K`, plus the cone:

| parameter | meaning |
|---|---|
| `eta` | friction (pressure coefficient) |
| `beta` | dilatancy — `beta != eta` is what makes the flow non-associative |
| `K_bulk` | bulk modulus, used both for the cone apex and for the elastic stiffness |

These used to arrive inside a `yield_function` **C++ object**, which the JSON
reader cannot convert — so the material could not be configured from a document
at all. As plain scalars it can (issue #33).

### `rk_plasticity`

Takes `K`, `G`, `sigma_0`, the two sources, plus `tolerance`, `max_iter` and a
`tableau` pointer. Available tableaus: `forward_euler`, `explicit_midpoint`,
`rk4`, `implicit_euler`, `implicit_midpoint`, `crank_nicolson`, `sdirk3`,
`gauss_legendre_4`.

### Hardening

Both publish `hardening_stress` (H) and `hardening_modulus` (dH/dκ), and read
`equivalent_plastic_strain` from the plasticity material — a `Local` edge,
because H depends on κ, which is the unknown being solved for.

| material | law | parameters |
|---|---|---|
| `linear_isotropic_hardening` | `H = K κ` | `source`, `K` |
| `exponential_isotropic_hardening` | `H = K_inf (1 − e^{−δκ})` | `source`, `K_inf`, `delta` |

---

## 3. What is assumed, and where it bites

### Isotropic elasticity is required, not preferred

Every closed form below rests on `C_e` being isotropic:

```
C_e : N        = 2G dev(N) + K tr(N) I
N : C_e : N    = 3G                        (J2, N deviatoric)
C_e : X : C_e  = 4G² X                     (X deviatoric in both index pairs)
```

This is why the plasticity materials **build their own** `C_e` from `K` and `G`
rather than reading a rank-4 tangent from an elastic material. Accepting an
arbitrary `C_e` advertised a generality none of them can honour — a caller
wiring an anisotropic tangent would have got silently wrong answers.

It also keeps `linear_elasticity` out of plasticity graphs, which matters for a
second reason: its `stress` output is `C : ε` with `ε_p` ignored, so inside a
plasticity graph it is not the stress of anything and diverges as plastic strain
accumulates.

| step | κ | `stress` (real) | `elastic::stress` | error |
|---|---|---|---|---|
| 40 | 0.0094 | 106.25 | 107.69 | +1.4% |
| 120 | 0.1094 | 306.25 | 323.08 | +5.5% |
| 200 | 0.2094 | 506.25 | 538.46 | **+6.4%** |

`linear_elasticity` remains a valid material — for a genuinely elastic model
`C : ε` *is* the answer, and `isotropic_damage` consumes both its stress and its
tangent legitimately. It is simply the wrong dependency for plasticity.

### Clamps belong to the caller

`newton_scalar` does not clamp its result. `max(x, 0)` is a statement about a
plastic multiplier and `abs(x)` about a curing degree; neither is about Newton's
method. Each lives in the material that owns the assumption (issue #13).

### The return map cannot be graph-driven

`local_newton` exists rather than everything using `backward_euler` because a
return map solves **twice per update** — smooth cone, then apex — with different
residuals, and chooses the branch on the first solve's convergence. A property
edge carries one number; it cannot carry "and it failed, so take the other
branch".

Convergence therefore travels *with* the result (`{x, converged, iterations}`)
rather than being queried from the solver afterwards, where it went stale
between the two solves.

---

## 4. The consistent tangent

For the smooth return, with `M` the yield normal and `N` the flow normal
(`M == N` when associative):

```
A            = C_e − Δλ (C_e : dN/dσ : C_e)
dλ/dε        = (M : C_e) / (M : C_e : N + H′)
C_consistent = A − (C_e : N) ⊗ dλ/dε
```

`M ≠ N` under non-associative flow makes the tangent **major-asymmetric**,
measured at 0.6–0.8 % of peak magnitude on multiaxial DP paths.

J2 collapses this to the standard closed form:

```
C = C_e − (6G²Δλ/σ_eq) IIdev + (4G²Δλ/σ_eq − 4G²/(3G+H′)) N ⊗ N
```

which is algebraically identical, not an approximation — confirmed by
bit-identical stress against the general path.

### Apex return (Drucker-Prager only)

When the deviatoric correction would overshoot the cone tip
(`G Δλ ≥ √J₂`), the return goes to the apex instead: deviatoric stress to zero,
pressure pinned at `(k + H)/η`, volumetric plastic flow from `β`. Its tangent is
a **branch** tangent — valid only for perturbations staying on the apex — because
the return map is non-smooth there.

---

## 5. Verification

`tangent_checker` compares the analytical tangent against a central difference
through the graph. Coverage:

| path | purpose |
|---|---|
| uniaxial | the historical case |
| pure shear, biaxial, triaxial, mixed dev+shear | exercise genuine non-associativity |
| hydrostatic | the **only** path that reaches the apex branch |

All agree to ~1e-10, the apex to 8e-9.

Two coverage lessons worth keeping:

- **A single load path proves one path.** The apex return was executed by *no*
  test for its entire existence — instrumenting the predicate gave
  `APEX_HITS=0` across every binary — because every path was uniaxial and the
  apex sits on the hydrostatic axis. It was unreachable by construction, not by
  oversight.
- **A tolerance set by the worst step licenses errors in all the others.**
  `J2TangentTest` bounded a whole run at `0.1` because the elastic→plastic
  transition step is genuinely inexact. A 0.5 % error injected into the tangent
  landed at 2.4e-4 and passed, while real plastic steps sit at 4.4e-10. The two
  regimes are now bounded separately.

---

## 6. Performance

Two binaries built from the actual commits, run interleaved so each pair sees
the same machine state; 15 pairs; speedup computed per pair.

| | old (median) | new (median) | paired speedup | range |
|---|---|---|---|---|
| J2 | 1466.2 ns/step | **211.5** | **7.00×** | 5.54–8.84× |
| Drucker-Prager | 1176.3 ns/step | **652.4** | **1.79×** | 1.58–1.90× |

The range is the honest figure: the *same* binary measured 1171–2253 ns across
runs on this machine, so any single-run comparison is worth about one
significant digit.

J2 gains more because it also sheds machinery it never used. DP keeps the
non-associative structure and the apex branch and gains only the arithmetic —
which is the correct outcome. **The generality DP pays for is generality DP
uses.**

---

## 7. Building a model

```cpp
// solver — one instance can serve several materials
p.insert<std::string>("name", "solver");
ctx.create<local_newton<policy>>(p);

// hardening — reads back from the plasticity material
p.clear();
p.insert<std::string>("name", "hardening");
p.insert<std::string>("source", "j2");
p.insert<T>("K", 1000.0);
ctx.create<linear_isotropic_hardening<policy>>(p);

// plasticity — no elastic material anywhere
p.clear();
p.insert<std::string>("name", "j2");
p.insert<std::string>("hardening_source", "hardening");
p.insert<std::string>("strain_source", "stepper");
p.insert<std::string>("solver_source", "solver");
p.insert<T>("K", 166.67);
p.insert<T>("G", 76.92);
p.insert<T>("sigma_0", 50.0);
ctx.create<j2_plasticity<policy>>(p);
```

Note `hardening.source = "j2"` while `j2.hardening_source = "hardening"`. The
cycle is deliberate and is why those edges are `Local`: H depends on κ, which is
what the return map solves for, so the hardening material is re-evaluated inside
the Newton loop through `update_source()`.

The same model in JSON, which is possible for Drucker-Prager only since its cone
parameters became plain scalars:

```json
{"type": "drucker_prager_plasticity", "name": "dp",
 "hardening_source": "hardening", "strain_source": "stepper",
 "solver_source": "solver",
 "G": 76.92, "sigma_0": 20.0,
 "eta": 0.1, "beta": 0.05, "K_bulk": 166.67}
```

---

## 8. Known gaps

- **`rk_plasticity` is still templated** on a yield function with one
  instantiation (`j2_rk_plasticity`) — the same shape that was removed from
  `small_strain_plasticity`.
- **`material_ref`** exists for exactly two call sites (the two backward-Euler
  return maps) and costs ~99 lines of core machinery. It bypasses the
  topological sort, so the plasticity↔solver ordering is not an edge the engine
  knows about. Safe today because `local_newton` holds no per-solve state.
- **The apex tangent is a branch tangent.** Verified consistent *on* the branch;
  a perturbation that leaves the apex back onto the smooth cone is not covered,
  and cannot be by a central difference.
