# Plasticity materials

How the three plasticity models — `j2_plasticity`, `drucker_prager_plasticity`
and `j2_rk_plasticity` — are put together, what each one requires, and which
assumptions are load-bearing.

This is the design. The mathematics behind it, from additive kinematics through
the return map to the consistent tangent, is in
[`plasticity-theory.md`](plasticity-theory.md).

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

   j2_rk_plasticity iterates its own Butcher tableau and uses neither.
```

Three plasticity materials, none of them templated on a yield function any
more:

| material | integrator | yield surface | flow |
|---|---|---|---|
| `j2_plasticity` | backward Euler (radial return) | von Mises cylinder | associative |
| `drucker_prager_plasticity` | backward Euler + apex branch | DP cone | **non**-associative (β ≠ η) |
| `j2_rk_plasticity` | Runge–Kutta, any Butcher tableau | von Mises cylinder | associative |

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

### `j2_rk_plasticity`

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
C_e : X : C_e  = 4G² X                     (X traceless AND minor-symmetric
                                            in both index pairs)
```

The third line's precondition is four conditions, not two. `C_e : X = 2G X`
needs `X` traceless in the first index pair (so the `3K IIvol` term drops) *and*
minor-symmetric in it (so `IIdev : X` returns `X`), and the mirror pair for
`X : C_e`. A traceless `X` that is skew in the first pair gives **100 % error**.
Both `dN/dσ` here are `IIdev` minus `v ⊗ v` with `v` deviatoric and symmetric,
so all four hold structurally.

Stated as a restriction on a *future* yield function it is narrower than
"deviatoric": **the volumetric part of `N` must be constant with respect to `σ`.**
A pressure-dependent dilatancy `β(p) = β₀ + c·p` adds `(c/9) I ⊗ I` to `dN/dσ`,
which is traceless in neither pair; `c = 1e-3` measures a **35 %** tangent error,
and the error scales with `K/G` rather than with `c` relative to `β`. Cap
models, Matsuoka–Nakai and Lade surfaces, and any smoothed cone tip share the
property. Nothing in the code guards it.

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

### The return map solves conditionally; a graph property is evaluated unconditionally

This is why `local_newton` exists rather than everything using
`backward_euler`.

A property's update callback runs whenever the graph updates. **Plasticity only
solves when the trial state exceeds yield** — and in a real analysis most
integration points are elastic most of the time. Both return maps short-circuit
before touching the solver:

```cpp
if (sig_eq - sigma_0 - H <= 0) { /* elastic: stress = trial, tangent = C_e */ return; }
```

Making the solve a property callback would run a Newton at every elastic point:

| | measured |
|---|---|
| elastic step | ~92 ns |
| plastic step | ~168 ns |

Roughly 80 % more work, paid exactly where a real analysis spends most of its
time.

It is worse than a cost. At `Δλ = 0` on an elastic step the residual is
negative, so Newton drives `Δλ` negative; holding it at zero needs a clamp —
which is precisely the `max(x, 0)` that issue #13 objects to. **The clamp and
the graph-driving are the same problem**: the graph mode has no way to express
*do not solve*.

Drucker-Prager cannot be graph-driven for a second, independent reason: it
solves up to **twice** per update — an apex pre-check, a smooth-cone solve, then
an apex fallback if that fails to converge — and chooses the branch on the first
solve's convergence. A property edge carries one number, not "and it failed, so
take the other branch".

J2 solves once and has no such branch, so *only* the conditional argument
applies to it. It could be graph-driven; it would simply cost more than it
saves, and would leave two patterns where there is now one.

Convergence therefore travels *with* the result (`{x, converged, iterations}`)
rather than being queried from the solver afterwards, where it went stale
between Drucker-Prager's two solves.

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

The criterion is applied to the **converged** `Δλ`, after the smooth Newton has
run. It used to be applied to the zero-hardening bound `Δλ_max = F/G_eff`, which
is an *upper* bound on `Δλ`: "the bound triggers apex" does not imply "the true
`Δλ` triggers apex", and the branch fired on a strict superset. The two regions
coincide only at `H' = 0`; with `H' = 500` the pre-check fired for `q_trial` up
to 923 where the true threshold was 1.5, and with `β = 0` for every
pressure-overshooting state. There is no correct cheap pre-check — a sound one
needs a *lower* bound on `Δλ`, which needs an upper bound on `H'` — so the
smooth Newton now always runs.

The apex tangent is rank 1 in every case, not only for `H' = 0`: it is a
multiple of `I ⊗ I`, and the apex update sets `dev(ε_p) = dev(ε)` so every
deviatoric mode has zero stiffness. An element with all Gauss points at the apex
has a singular stiffness whatever the hardening.

---

## 5. Verification

`tangent_checker` compares the analytical tangent against a central difference
through the graph. Coverage:

| path | purpose |
|---|---|
| uniaxial | the historical case |
| pure shear, biaxial, triaxial, mixed dev+shear | exercise genuine non-associativity |
| hydrostatic | the only path in this suite that reaches the apex branch |

All agree to ~1e-10, the apex to 8e-9.

Two coverage lessons worth keeping:

- **A single load path proves one path.** The apex return was executed by *no*
  test for its entire existence — instrumenting the predicate gave
  `APEX_HITS=0` across every binary — because every path was uniaxial and the
  apex sits on the hydrostatic axis. It was unreachable by construction, not by
  oversight.
- **A tangent check cannot see a wrong branch.** On a state misclassified into
  the apex branch, the analytical tangent and the central difference both stay
  inside that branch, so they agree perfectly while the stress is wrong — and
  the wrong stress still satisfies `F = 0` to 4e-14, because it is *on* the
  yield surface, just the wrong projection onto it. `tangent_checker` validates
  the derivative of whatever the code does. The branch choice needs its own
  test, which is now `DPBranch.*`.
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

## 8. What is still generic, and why

`plasticity_utils`' free functions — `compute_trial`, `evaluate_at_state`,
`compute_tangent` — remain templated on a yield function, and that generality is
real: `drucker_prager_plasticity` instantiates them with the DP cone and
`j2_rk_plasticity` with the von Mises cylinder. Two callers, two yield
functions, shared return-mapping algebra.

That is the distinction worth holding onto. A template parameter with **one**
argument is indirection — it was removed from the two classes that carried it,
which are now `drucker_prager_plasticity` and `j2_rk_plasticity`. A template
parameter with two genuinely different arguments is
what templates are for.

## 9. Settled: the solver is reached by `material_ref`, not by the graph

The return maps hold a `material_ref<local_newton>` and call `solve()`. The
alternative — publishing a `yielding` flag so a graph-driven `backward_euler`
knows when to iterate — was considered and **rejected**.

The pattern would work: `strain_threshold_yield` already publishes
`is_yielding` and `isotropic_damage` consumes it, so a gating flag is native to
this codebase. It would retire `local_newton` and, with it, `material_ref` —
about 99 lines of core machinery whose only two call sites are these.

It was rejected on what it would cost to express:

- **The material splits into phases.** `compute()` currently does trial →
  solve → update in one callback. Graph-driven it becomes publish
  `yielding`/`residual`/`jacobian`, solver runs, read `delta`, update. Two or
  three properties where there is one, and graph dispatch measures ~30 ns even
  for a trivial graph.
- **A flag cannot carry Drucker-Prager's branch.** The apex branch is chosen on
  the *converged* multiplier — either the smooth solve failed, or it succeeded
  and its `Δλ` overshoots the tip — neither of which is known until it has run.
  Expressible only as trial → smooth solver → `needs_apex` → apex solver gated
  on that flag → update: two solver instances, three flags, four phases,
  replacing one `if`. This got stronger, not weaker, when the branch stopped
  being decided by a pre-check (§4).
- **The elastic saving is partial** anyway — the solver's callback still fires
  and returns early on the flag.

The cost of the machinery is real but bounded; the cost of removing it is a
graph harder to read than the code it replaces.

**One constraint this decision carries.** `material_ref` bypasses the
topological sort, so the plasticity↔solver ordering is not an edge the engine
knows about. That is safe only because `local_newton` holds **no per-solve
state** — its `solve()` is `const` and returns everything it computes. Give it
mutable state and the ordering becomes real and unenforced.

## 10. Known gaps

- **The apex tangent is a branch tangent.** Verified consistent *on* the branch;
  a perturbation that leaves the apex back onto the smooth cone is not covered,
  and cannot be by a central difference. It is also rank 1 by construction, so
  an element fully at the apex is singular regardless of hardening.
- **Unstable softening is rejected, not resolved.** `Δλ = F/(G_eff + H')` turns
  negative once `H' ≤ −G_eff`, where the yield residual has positive slope and
  the local problem has no admissible solution. Both return maps now throw
  there; they used to clamp `Δλ` to zero, which returned the elastic stress —
  measured on a uniaxial path with `H' = −300` against `3G = 230.8`, the
  equivalent stress climbed past `σ₀ = 50` to 76.9 with `α` identically zero.
  Resolving that branch needs viscous or gradient regularisation, which is out
  of scope for a local return map. Moderate softening, `−G_eff < H' < 0`, works.
  The apex criterion also inverts under softening, giving a *missed* apex with
  `q_new < 0`; that is still unguarded.
- **The yield-function template is unconstrained.** `compute_tangent`'s 4G²
  collapse holds for the two yield functions in the tree and would be wrong for
  a stress-dependent volumetric flow (§3). A `static_assert` cannot express it;
  nothing checks it.
