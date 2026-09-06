# Small-strain J2 and Drucker-Prager plasticity: from kinematics to the linearized stress

This is the theory behind `j2_plasticity`, `drucker_prager_plasticity` and
`j2_rk_plasticity`: the continuum statement, the discrete return map, and the
linearized stress each one hands back to the host solver. Every equation is
carried through to the expression that appears in the code, and the code line is
named where it does.

Its companion, [`plasticity.md`](plasticity.md), covers the *design* — why these
materials build their own elastic tangent, why the solver is reached by
`material_ref`, what is verified and what is not. This document is the
mathematics.

**Conventions.** Small strain throughout. Tensors are `Dim`-dimensional; `D`
below is `Traits::Dim`, which is 3 for every registered material. `⊗` is the
dyadic product (`tmech::otimes`), `:` the double contraction
(`tmech::dcontract`). `I` is the second-order identity, `IIsym` the symmetric
fourth-order identity, and

```
IIdev = IIsym - I⊗I/D                             plasticity_utils.h: make_IIdev
```

the deviatoric projector, so `dev(a) = IIdev : a = a - tr(a)/D · I`.

---

## Contents

1. [Kinematics](#1-kinematics)
2. [Elastic law](#2-elastic-law)
3. [Yield functions](#3-yield-functions)
4. [Flow rule and plastic potential](#4-flow-rule-and-plastic-potential)
5. [Hardening and the internal variable](#5-hardening-and-the-internal-variable)
6. [Loading, unloading, consistency](#6-loading-unloading-consistency)
7. [The return map](#7-the-return-map)
8. [J2 in closed form](#8-j2-in-closed-form)
9. [Drucker-Prager in closed form](#9-drucker-prager-in-closed-form)
10. [The apex branch](#10-the-apex-branch)
11. [The linearized stress](#11-the-linearized-stress)
12. [The multi-stage variant](#12-the-multi-stage-variant)
13. [Equation-to-code map](#13-equation-to-code-map)
14. [What the derivations assume](#14-what-the-derivations-assume)

---

## 1. Kinematics

The strain is small and the decomposition additive:

```
ε = ε_e + ε_p                                                             (1.1)
```

with `ε` the symmetric small-strain tensor supplied by the host (a `strain_source`
in the graph), `ε_e` elastic and `ε_p` plastic. Both `ε_p` and the hardening
variable `κ` of §5 are **history variables**: they carry an old and a new value
across the step, and the driver — never the material — advances them by calling
`commit()`.

There is no multiplicative split, no rotation of the plastic frame, no objective
rate. `ε_p` is a plain tensor added to `ε_e`, which is what makes every
expression below algebraic rather than a differential equation in a rotating
frame. It is valid while strains and rotations are small; beyond that the model
is out of scope and nothing in the code detects it.

In rate form, with `λ̇ ≥ 0` the plastic multiplier and `N` the flow direction
of §4:

```
ε̇_p = λ̇ N                                                                (1.2)
```

`ε̇_p` is the only source of dissipation in the model; the elastic response is
reversible by construction.

---

## 2. Elastic law

The elastic response is linear and **isotropic**:

```
σ = C_e : ε_e = C_e : (ε - ε_p)                                           (2.1)

C_e = K I⊗I + 2G IIdev                    plasticity_utils.h: make_isotropic_tangent
```

with `K` the bulk and `G` the shear modulus. Contracting once gives the identity
every fast path in the code relies on:

```
C_e : a = K tr(a) I + 2G dev(a)                                           (2.2)
```

For `a` deviatoric this is `2G a`; for `a = I` it is `D·K·I`. Splitting (2.1),

```
p = tr(σ)/3 = K tr(ε_e)          (D = 3)
s = dev(σ)  = 2G dev(ε_e)                                                 (2.3)
```

The bulk term is written `K I⊗I`, not `3K IIvol`. Those coincide only at `D = 3`,
because `IIvol = I⊗I/D`, and the shortcut (2.2) assumes the first. Writing the
second made a material disagree with itself by 30 % at `D = 2` — see §14.

Every plasticity material builds `C_e` itself from its own `K` and `G` rather
than reading a rank-four tangent from an elastic material. That is not a
convenience: §11 and §14 show the closed forms are *false* for an anisotropic
`C_e`, so accepting one would advertise a generality the model cannot honour.

---

## 3. Yield functions

Both models are pressure-and-deviator functions of the stress, but they use
**different normalizations of the deviatoric invariant**, and that difference
propagates into every effective modulus below. It is a genuine difference in the
two files, not a presentational one.

### 3.1 J2 (von Mises)

```
J₂    = ½ s:s
σ_eq  = √(3J₂) = √(3/2 · s:s)             yield_functions.h: equivalent_stress
F     = σ_eq - σ₀ - H(κ)                  yield_functions.h: trial_yield         (3.1)
```

`σ_eq` is the uniaxial-equivalent stress: in uniaxial tension `σ_eq = |σ₁₁|`, so
`σ₀` is the uniaxial yield stress and can be read off a tensile test. The yield
surface is a cylinder about the hydrostatic axis — pressure-insensitive, no
apex, `needs_apex_return` returns `false` unconditionally.

### 3.2 Drucker-Prager

```
q  = √J₂ = √(½ s:s)              drucker_prager_yield_function.h: equivalent_stress
p  = tr(σ)/3
F  = q + η p - k - H(κ)          drucker_prager_yield_function.h: trial_yield     (3.2)
```

Note `q = √J₂`, **without** the `√3`. The surface is a cone about the hydrostatic
axis, opening towards compression or tension according to the sign convention of
`η`:

- `η` — the friction (pressure) coefficient. `η > 0` makes tensile pressure
  destabilising, so the material yields earlier in tension than in compression.
- `k` — the cohesion, the `sigma_0` parameter. At `p = 0` it is the yield value
  of `q`.
- `η = 0` reduces the cone to a cylinder, which is von Mises up to the
  normalization difference above: `σ_eq = √3 q`, so a DP material with `η = 0`
  and cohesion `k` behaves as J2 with `σ₀ = √3 k`.

The cone closes at the **apex** on the hydrostatic axis, where `q = 0` and
`η p = k + H`. That point is not smooth, and §10 is about it.

Because `η > 0` penalises tensile pressure, uniaxial *strain* yields at
different magnitudes in the two directions. With `q = 2G|e|/√3` and `p = K e`
for a uniaxial strain `e`,

```
e_tension        = k / (2G/√3 + ηK)
|e_compression|  = k / (2G/√3 - ηK)                                       (3.3)
```

which coincide only at `η = 0`. `DruckerPragerPressure.YieldsEarlierInTensionThanInCompression`
pins exactly this ratio, which makes it a test of the friction term rather than
of yielding in general.

---

## 4. Flow rule and plastic potential

The flow direction comes from a plastic potential `Q`:

```
N = ∂Q/∂σ                                                                 (4.1)
```

and the *yield* normal, which appears in the consistency condition and hence in
the tangent, from `F`:

```
M = ∂F/∂σ                                                                 (4.2)
```

`M = N` when the flow is **associative**, `M ≠ N` when it is not. Keeping the two
apart is what lets one implementation carry both models.

### 4.1 J2 — associative

```
∂σ_eq/∂σ:   σ_eq² = 3/2 s:s  ⟹  2σ_eq ∂σ_eq/∂σ = 3 s : IIdev = 3 s

N = M = 3/2 · s/σ_eq                      yield_functions.h: flow_normal          (4.3)
```

Two properties are used repeatedly:

```
tr(N) = 0                    plastic flow is isochoric
N:N   = 9/4 · (s:s)/σ_eq² = 3/2                                           (4.4)
```

`tr(N) = 0` is why the volumetric term of (2.2) drops out of every J2 expression
and why `C_e : N = 2G N` exactly.

### 4.2 Drucker-Prager — non-associative

The potential replaces the friction coefficient `η` by a **dilatancy**
coefficient `β`:

```
Q = q + β p

N = s/(2q) + β/3 · I           drucker_prager_yield_function.h: flow_normal
M = s/(2q) + η/3 · I           drucker_prager_yield_function.h: yield_normal     (4.5)
```

using `∂q/∂σ = s/(2q)` and `∂p/∂σ = I/3`. Now

```
tr(N) = β                                                                 (4.6)
```

so plastic flow is **dilatant**: `β > 0` produces plastic volume increase. This
is the physical reason the model is non-associative — taking `β = η` would tie
dilatancy to friction and, for realistic friction angles, over-predict volume
change badly. The price is that `C_ep` loses its major symmetry (§11.6).

The `/3` in (4.5) is a hard 3, while `dev` uses `/D`. At `D = 3` they agree; see
§14.

---

## 5. Hardening and the internal variable

One scalar internal variable `κ` with

```
κ̇ = λ̇                                                                    (5.1)
```

so `κ` is the accumulated plastic multiplier. For J2 that is exactly the
conventional equivalent plastic strain, and the check is worth doing because it
fixes what `σ₀` and `H` mean:

```
‖ε̇_p‖ = λ̇ √(N:N) = λ̇ √(3/2)
ε̇_p^eq ≡ √(2/3) ‖ε̇_p‖ = λ̇                                                (5.2)
```

and the plastic work is

```
σ : ε̇_p = λ̇ σ:N = λ̇ · 3/2 · (s:s)/σ_eq = λ̇ σ_eq                        (5.3)
```

so `(σ_eq, κ)` are work-conjugate. `H(κ)` is therefore the uniaxial hardening
curve, directly comparable to a tensile test.

For Drucker-Prager `κ` is the multiplier itself and is *not* a strain measure of
the same kind — the normalization `q = √J₂` and the volumetric part of `N` both
enter — so `k` and `H` there are cone parameters, not uniaxial ones.

The material never evaluates `H` itself. It reads two properties from a
hardening material through Local edges:

```
H  = "hardening_stress"       H'  = "hardening_modulus"
```

`linear_isotropic_hardening` supplies `H = K κ`, `H' = K`. Any other law that
publishes the same two properties works unchanged; the return maps only ever use
`H(κ)` and `H' = dH/dκ`.

`H' < 0` is softening. It is admissible while `H' > -G_eff`; at or below that the
local problem has no admissible solution and both materials throw — §14.

---

## 6. Loading, unloading, consistency

The Karush-Kuhn-Tucker conditions:

```
F ≤ 0        λ̇ ≥ 0        λ̇ F = 0                                        (6.1)
```

and, during plastic flow, the consistency condition `Ḟ = 0`:

```
M : σ̇ - H' κ̇ = 0                                                         (6.2)
```

Substituting `σ̇ = C_e : (ε̇ - λ̇ N)` gives the multiplier in rate form,

```
λ̇ = (M : C_e : ε̇) / (M : C_e : N + H')                                   (6.3)
```

The denominator is the quantity the code calls the **effective modulus** plus the
hardening modulus. It is worth naming, because it is the same scalar in the rate
form, in the discrete residual, and in the tangent:

```
G_eff ≡ M : C_e : N                       *_yield_function.h: effective_modulus  (6.4)
```

### 6.1 `G_eff` for the two models

**J2.** `C_e : N = 2G N` since `tr(N) = 0`, so

```
G_eff = N : C_e : N = 2G (N:N) = 2G · 3/2 = 3G                            (6.5)
```

**Drucker-Prager.** From (2.2) and (4.5),

```
C_e : N = 2G · s/(2q) + K β I = G s/q + K β I                             (6.6)

G_eff = M : C_e : N
      = [s/(2q) + η/3 I] : [G s/q + Kβ I]
      = G (s:s)/(2q²) + (η/3) Kβ tr(I)
      = G + K η β                                                          (6.7)
```

using `s:s = 2q²`, `tr(s) = 0`, `s:I = 0` and `tr(I) = 3`. The cross terms
vanish; only the shear term and the friction-dilatancy product survive. Both
match the two `effective_modulus` implementations exactly.

`G_eff` is the first place isotropy is load-bearing: for an anisotropic `C_e`,
`M : C_e : N` is state-dependent and no longer a material constant, so the
*residual itself* would be wrong — not merely the tangent.

---

## 7. The return map

### 7.1 Discretisation

Backward Euler on (1.2) over a step, with the flow direction evaluated at the
**trial** state:

```
ε_p^{n+1} = ε_p^n + Δλ N_tr
κ^{n+1}   = κ^n + Δλ                                                      (7.1)
```

`Δλ ≥ 0` is the one unknown of the step.

### 7.2 Trial state

Freeze the plastic strain and load elastically:

```
σ_tr  = C_e : (ε^{n+1} - ε_p^n)           plasticity_utils.h: compute_trial
F_tr  = F(σ_tr, κ^n)                                                      (7.2)
```

`F_tr ≤ 0` ⟹ the step is elastic: `σ = σ_tr`, `C_ep = C_e`, history unchanged.
That branch is not an optimisation, it is the KKT case `λ̇ = 0`.

`F_tr > 0` ⟹ the return map runs. **This is the only case in which a solve
happens at all**, which is why the scalar solver is reached by `material_ref`
rather than by a graph edge — see `plasticity.md` §9.

### 7.3 Reduction to one scalar

Substituting (7.1) into (2.1),

```
σ = C_e : (ε - ε_p^n - Δλ N_tr) = σ_tr - Δλ (C_e : N_tr)                  (7.3)
```

The whole tensor update is therefore known once the single scalar `Δλ` is. What
makes the reduction exact rather than an approximation is that `N` does **not**
rotate during the return: §8.1 and §9.1 show, for each model separately, that the
returned deviator stays parallel to the trial deviator. That is the property that
turns a tensor-valued nonlinear system into a scalar one.

Contracting (7.3) with `M` and using (6.4) gives the residual:

```
r(Δλ) = φ_tr - G_eff Δλ - Y₀ - H(κ^n + Δλ) = 0            *_yield_function.h: residual
r'(Δλ) = -(G_eff + H')                                    *_yield_function.h: jacobian
```

where `φ_tr` is the **modified equivalent stress** at the trial state —
`σ_eq_tr` for J2, `q_tr + η p_tr` for DP — and `Y₀` is `σ₀` or `k`. This is one
scalar equation in one unknown, solved by `newton_scalar`. For linear hardening
it is linear, so

```
Δλ = F_tr / (G_eff + H')                                                  (7.4)
```

and Newton lands on the root in a single step. That is also the reason forward
Euler, backward Euler and the monolithic map must give **identical** answers for
linear hardening, which `J2RKStage.ExplicitAndImplicitStagesSolveTheSameEquation`
asserts to 1e-12 relative rather than to a tolerance.

---

## 8. J2 in closed form

### 8.1 Radial return

From (7.3) with `C_e : N = 2G N` and `N_tr = (3/2) s_tr/σ_eq_tr`:

```
s = s_tr - 2G Δλ N_tr = s_tr (1 - 3GΔλ/σ_eq_tr)                           (8.1)
```

The returned deviator is a **scalar multiple** of the trial deviator: the return
is radial in deviatoric space and `N = N_tr` exactly, not approximately. Taking
the norm,

```
σ_eq = σ_eq_tr - 3G Δλ                                                    (8.2)
```

which is (7.3) contracted, and recovers `G_eff = 3G`. The volumetric stress is
untouched, since `tr(N) = 0`.

### 8.2 The update

```
σ = σ_tr - 2G Δλ N                        j2_plasticity.h: m_stress
ε_p^{n+1} = ε_p^n + Δλ N
κ^{n+1}   = κ^n + Δλ                                                      (8.3)
```

The stress line uses `C_e : N = 2G N` rather than a rank-four contraction —
algebraically identical, and it measures 33 ns against ~245 ns for the whole
step.

### 8.3 Residual

```
r(Δλ)  = σ_eq_tr - 3G Δλ - σ₀ - H(κ^n + Δλ)
r'(Δλ) = -3G - H'                                                         (8.4)
```

---

## 9. Drucker-Prager in closed form

### 9.1 Smooth cone return

From (6.6),

```
s   = s_tr - Δλ G s_tr/q_tr = s_tr (1 - GΔλ/q_tr)                         (9.1)
p   = p_tr - K β Δλ
q   = q_tr - G Δλ
```

Deviatorically the return is again radial, so `N` does not rotate. The
volumetric part *is* affected, because `tr(N) = β ≠ 0` — this is the essential
difference from J2 and the reason the volumetric term of (2.2) may not be
dropped.

Substituting into (3.2):

```
F = (q_tr - GΔλ) + η(p_tr - KβΔλ) - k - H
  = (q_tr + η p_tr) - (G + Kηβ) Δλ - k - H                                (9.2)
```

recovering `φ_tr = q_tr + η p_tr` and `G_eff = G + Kηβ` — (6.7) again, now from
the discrete side.

### 9.2 The update

```
C_e:N = 2G dev(N) + K tr(N) I                drucker_prager_plasticity.h: Ce_N
σ     = σ_tr - Δλ (C_e:N)
ε_p^{n+1} = ε_p^n + Δλ N
κ^{n+1}   = κ^n + Δλ                                                      (9.3)
```

The volumetric term does not drop out as it does for J2, so both terms of (2.2)
are kept.

### 9.3 Branch condition

(9.1) is only meaningful while the deviatoric correction does not overshoot the
tip. It stops being so at

```
G Δλ ≥ q_tr                    drucker_prager_yield_function.h: needs_apex_return  (9.4)
```

exactly where (9.1) would drive `q` to zero and then negative — the returned
deviator would flip sign. Beyond it the smooth branch is not the answer and §10
takes over.

**The criterion is applied to the converged `Δλ`, after the Newton has run.** It
is monotone increasing in `Δλ`, so feeding it an *upper* bound on `Δλ` — for
instance the zero-hardening `F_tr/G_eff` — and concluding "apex" is invalid: an
upper bound crossing a threshold says nothing about the true value. Only the
contrapositive holds. A sound cheap pre-check would need a *lower* bound on `Δλ`,
which needs an upper bound on `H'`, which is not available; so the smooth Newton
always runs. `plasticity.md` §4 records what the invalid version cost.

---

## 10. The apex branch

At the apex the deviatoric stress vanishes and the cone condition degenerates to
a purely volumetric statement:

```
s = 0            η p = k + H(κ^n + Δκ)                                   (10.1)
```

The algorithm is a **projection**, not a single-multiplier flow update: the
deviatoric plastic strain is set directly to enforce `q = 0`,

```
dev(ε_p) = dev(ε)                    drucker_prager_yield_function.h:
tr(ε_p) += β Δκ                          apex_plastic_strain                     (10.2)
```

while `Δκ` follows from (10.1). With `p = p_tr - KβΔκ`,

```
η(p_tr - K β Δκ) - k - H(κ^n + Δκ) = 0
⟹  φ_apex - G_eff^apex Δκ - k - H = 0                                    (10.3)

φ_apex       = η p_tr                    apex_modified_sig_eq
G_eff^apex   = K η β                     apex_effective_modulus
```

the same scalar residual as (7.3) with two substituted arguments, which is why
one `solve_scalar_return` serves both branches.

### 10.1 Apex tangent

Differentiate (10.1) and `p = p_tr - KβΔκ`:

```
η dp = H' dΔκ
dp   = dp_tr - K β dΔκ
⟹  η dp_tr = (H' + Kηβ) dΔκ
⟹  dp       = dp_tr · H'/(H' + Kηβ)                                      (10.4)
```

With `dp_tr = K I:dε` and `σ = p I` on the branch,

```
C_ep^apex = K H' / (Kηβ + H') · I⊗I       drucker_prager_yield_function.h:
                                              apex_tangent                       (10.5)
```

Two properties of (10.5) matter to the host:

- It is **rank one in every case**, not only the degenerate one. It is a multiple
  of `I⊗I`, and (10.2) pins `dev(ε_p) = dev(ε)` so `dev(σ) ≡ 0` for *any*
  perturbation — every deviatoric mode has zero stiffness. An element with all
  its Gauss points at the apex is singular whatever the hardening.
- `H' = 0` gives a **zero** tangent regardless of `Kηβ`, since the numerator is
  `K H'`.

It is a **branch** tangent: valid for perturbations that stay on the apex. The
return map is genuinely non-smooth there, so no single tangent describes both
sides, and a central difference across the boundary measures neither.

---

## 11. The linearized stress

This is the object the host Newton actually consumes:

```
dσ = C_ep : dε         C_ep = ∂σ^{n+1}/∂ε^{n+1}                          (11.1)
```

the derivative of the **algorithm** — the consistent or algorithmic tangent —
not of the continuum rate law. The distinction is not academic: §11.5 shows the
two differ by a term proportional to `Δλ`, and using the continuum one degrades
the host Newton from quadratic to linear convergence.

### 11.1 The two dependencies

From (7.3), `σ` depends on `ε` twice: explicitly, and through `Δλ(ε)`.

```
σ(ε) = C_e : (ε - ε_p^n - Δλ(ε) N_tr(ε))                                 (11.2)
```

so by the chain rule

```
C_ep = A + (∂σ/∂Δλ) ⊗ (dΔλ/dε)                                           (11.3)
```

with `A` the derivative at frozen `Δλ`.

### 11.2 The frozen-multiplier part `A`

`N_tr` depends on `ε` through `σ_tr = C_e : (ε - ε_p^n)`, so

```
dN_tr/dε = (dN/dσ) : C_e

A = C_e : (IIsym - Δλ dN_tr/dε)
  = C_e - Δλ · C_e : (dN/dσ) : C_e                                       (11.4)
```

The two flow-normal derivatives:

```
J2:  dN/dσ = (3/2 · IIdev - N⊗N) / σ_eq        yield_functions.h
DP:  dN/dσ = (IIdev - s⊗s/(2J₂)) / (2q)        drucker_prager_yield_function.h
                                               flow_normal_stress_derivative     (11.5)
```

Both are `IIdev` minus a dyad of a deviatoric symmetric tensor. Neither depends
on the volumetric part of `N` — for DP the `β/3 I` term is constant in `σ` and
differentiates away, which is exactly the property §14 warns a future yield
function not to break.

### 11.3 The `4G²` collapse

`A` as written needs two rank-four × rank-four contractions. For isotropic `C_e`
they collapse:

```
C_e : X : C_e = 4G² X                                                    (11.6)
```

**with four preconditions**, not two: `X` must be *traceless* **and**
*minor-symmetric* in the first index pair, and likewise in the second. From
`C_e = 3K IIvol + 2G IIdev`, the volumetric term drops only if `X` is traceless,
and `IIdev : X` returns `X` only if `IIsym : X` does, which is symmetry. A
traceless `X` that is skew in the first pair gives **100 % error**.

Both derivatives in (11.5) satisfy all four structurally, so

```
A = C_e - 4G² Δλ (dN/dσ)                     plasticity_utils.h: compute_tangent (11.7)
```

Verified to 2.5e-16 against the explicit double contraction for both models.

### 11.4 The rank-one part

The multiplier is defined implicitly by `r(Δλ, ε) = 0`, so by the implicit
function theorem

```
dΔλ/dε = -(∂r/∂Δλ)⁻¹ (∂r/∂ε)                                            (11.8)
```

From §7.3, `r = φ_tr(ε) - G_eff Δλ - Y₀ - H(κ^n + Δλ)`, and `φ_tr` depends on `ε`
only through `σ_tr`:

```
∂r/∂ε   = (∂φ/∂σ) : C_e = M : C_e
∂r/∂Δλ  = -(G_eff + H') = -(M : C_e : N + H')                           (11.9)
```

and from (7.3), `∂σ/∂Δλ = -C_e : N`. Assembling (11.3):

```
             ┌                     ┐   ┌            ┐
C_ep = C_e - │ 4G² Δλ (dN/dσ)      │ - │ (C_e:N) ⊗ (M:C_e) │
             └                     ┘   └ ───────────────── ┘
                                          M:C_e:N + H'                  (11.10)
```

which is `compute_tangent` line for line.

### 11.5 The continuum tangent, and why it is not enough

Setting `Δλ = 0` in (11.10) leaves

```
C_ep^cont = C_e - (C_e:N) ⊗ (M:C_e) / (G_eff + H')                      (11.11)
```

which is the classical elastoplastic tangent obtained from the rate form (6.3)
alone. The difference is the `4G² Δλ (dN/dσ)` term, which accounts for the
*rotation of the flow direction over the finite step*. It vanishes only in the
limit of an infinitesimal step. Keeping it is what makes the tangent consistent
with the discrete update and the host Newton quadratic.

### 11.6 J2 in closed form

For J2, `M = N`, `C_e : N = 2G N`, `M : C_e : N = 3G`, and `dN/dσ` is (11.5).
Substituting into (11.10):

```
A = C_e - 4G²Δλ (3/2 · IIdev - N⊗N)/σ_eq
  = C_e - (6G²Δλ/σ_eq) IIdev + (4G²Δλ/σ_eq) N⊗N

rank-one term = -(2G N) ⊗ (2G N)/(3G + H') = -(4G²/(3G+H')) N⊗N
```

Collecting the two `N⊗N` contributions:

```
C_ep = C_e - a IIdev + b N⊗N

a = 6G² Δλ / σ_eq_tr
b = 4G² Δλ / σ_eq_tr - 4G²/(3G + H')       j2_plasticity.h: m_tangent           (11.12)
```

term for term what the code computes. This is the Simo-Hughes radial-return
tangent under the `σ_eq = √(3J₂)`, `Δλ = Δε_p^eq` normalization; converting with
`Δγ = √(3/2) Δλ` reproduces their `2Gθ` / `-2Gθ̄` coefficients identically.

`C_ep` here is **symmetric**: `M = N`, so the dyad in (11.10) is `N⊗N`.

### 11.7 Drucker-Prager in closed form

DP keeps the general form (11.10). Two differences from J2:

- `M ≠ N` when `β ≠ η`, so the rank-one term is `(C_e:N) ⊗ (M:C_e)` with two
  *different* tensors. **`C_ep` is not symmetric.** That is intrinsic to
  non-associative flow, not an implementation artefact, and a host assembling
  only the upper triangle will get the wrong stiffness.
- `C_e : N` retains its volumetric part (6.6), and `M : C_e = 2G dev(M) + K tr(M) I`
  likewise with `tr(M) = η`.

### 11.8 Where the tangent is evaluated

At the **trial** state, not the converged one. The `∂ε_p/∂ε|_Δλ` term
differentiates `N_tr(σ_tr)`, so `dN/dσ` must be taken at `σ_eq_tr`. Substituting
the converged `σ_eq = σ_eq_tr - 3GΔλ` degrades the tangent by 2.6e-3 relative
against a central difference — a real loss of Newton rate, measured, not a wash.

`N` itself is a separate question: for radial return `s ∥ s_tr` exactly, so
`N_tr ≡ N`, and that substitution genuinely does not matter. The code uses trial
quantities for both.

---

## 12. The multi-stage variant

`j2_rk_plasticity` replaces the single backward-Euler step with a Runge-Kutta
tableau `(a, b, c)`. Per stage `i`, with `s` stages:

```
ε_p^{(i)} = ε_p^n + Σ_{j<i} a_ij Δλ_j N_j  ( + a_ii Δλ_i N_tr  if implicit )
κ^{(i)}   = κ^n   + Σ_{j<i} a_ij Δλ_j      ( + a_ii Δλ_i       if implicit )
```

with `Δλ_i` from the same scalar residual as (7.3) evaluated at the stage state —
directly for an explicit stage (`a_ii = 0`), by Newton for an implicit one. The
step closes with the `b`-weighted sum:

```
ε_p^{n+1} = ε_p^n + Σ_i b_i Δλ_i N_i
κ^{n+1}   = κ^n   + Σ_i b_i Δλ_i
Δλ_total  = Σ_i b_i Δλ_i                                                 (12.1)
```

and the tangent is (11.10) evaluated at the **converged** state with
`Δλ = Δλ_total`.

Three things follow, all of them limitations worth stating plainly:

- **The stage sum runs over `j < i` only.** A tableau with coupling *above* the
  diagonal — a fully implicit scheme such as Gauss-Legendre — is not integrated
  by this loop; its `a_0,1` is silently dropped. Before it was guarded,
  `gauss_legendre_4` produced an equivalent plastic strain of **-8.81** with a
  yield residual of +10880 and no error. The constructor now rejects any tableau
  that is neither DIRK nor explicit.
- **The implicit stage iterates on `N_tr`, not on its own stage normal.** It
  stores the converged stage normal afterwards. This is an approximation, and it
  is one reason the RK tangent is not exact.
- **The tangent is assembled from `Δλ_total` rather than differentiated through
  the stages.** Measured against a central difference on fully plastic steps it
  sits at ~3e-4, against ~1e-10 for the monolithic J2 tangent. That is a genuine
  approximation, not roundoff, and the test bounds reflect it.

For linear hardening the stage residual is linear, so every consistent DIRK or
explicit scheme lands on the same root as the monolithic map — which is the exact
identity §7.3 uses as a test.

---

## 13. Equation-to-code map

| equation | quantity | code |
|---|---|---|
| (2.1) | `C_e = K I⊗I + 2G IIdev` | `plasticity_utils.h: make_isotropic_tangent` |
| (3.1) | `σ_eq = √(3J₂)`, `F` | `yield_functions.h: equivalent_stress`, `trial_yield` |
| (3.2) | `q = √J₂`, `F = q + ηp - k - H` | `drucker_prager_yield_function.h: trial_yield` |
| (4.3) | `N = 3/2 s/σ_eq` | `yield_functions.h: flow_normal` |
| (4.5) | `N`, `M` with `β`, `η` | `drucker_prager_yield_function.h: flow_normal`, `yield_normal` |
| (6.5) | `G_eff = 3G` | `yield_functions.h: effective_modulus` |
| (6.7) | `G_eff = G + Kηβ` | `drucker_prager_yield_function.h: effective_modulus` |
| (7.2) | trial state | `plasticity_utils.h: compute_trial`, `evaluate_at_state` |
| (7.3) | scalar residual, jacobian | `*_yield_function.h: residual`, `jacobian` |
| (8.3) | `σ = σ_tr - 2GΔλ N` | `j2_plasticity.h: compute` |
| (9.3) | `σ = σ_tr - Δλ C_e:N` | `drucker_prager_plasticity.h: do_smooth_return` |
| (9.4) | `GΔλ ≥ q_tr` | `drucker_prager_yield_function.h: needs_apex_return` |
| (10.2) | apex projection | `drucker_prager_yield_function.h: apex_plastic_strain` |
| (10.5) | `C_ep^apex` | `drucker_prager_yield_function.h: apex_tangent` |
| (11.5) | `dN/dσ` | `*_yield_function.h: flow_normal_stress_derivative` |
| (11.10) | general `C_ep` | `plasticity_utils.h: compute_tangent` |
| (11.12) | J2 closed-form `C_ep` | `j2_plasticity.h: compute` |
| (12.1) | RK stage assembly | `j2_rk_plasticity.h: compute` |

### 13.1 The identities, checked

Every algebraic identity this document rests on, evaluated on a generic
multiaxial stress with `K = 166.67`, `G = 76.92`, `η = 0.1`, `β = 0.05`, against
the actual yield-function and tangent code:

| identity | equation | result |
|---|---|---|
| `C:a = K tr(a) I + 2G dev(a)` | (2.2) | 4.1e-12 on a scale of 3.6e+4 |
| `tr(N) = 0` (J2) | (4.4) | -2.8e-17 |
| `N:N = 3/2` (J2) | (4.4) | -2.2e-16 |
| `C:N = 2G N` (J2) | (4.4) | 1.5e-14 |
| `N:C:N = 3G` | (6.5) | 230.760000000 vs 230.760000000 |
| `tr(N) = β` (DP) | (4.6) | 1.4e-17 |
| `C:N = G s/q + Kβ I` (DP) | (6.6) | 1.7e-14 |
| `M:C:N = G + Kηβ` | (6.7) | 77.753350000 vs 77.753350000 |
| `C:X:C = 4G² X`, J2 `dN/dσ` | (11.6) | rel 2.2e-16 |
| `C:X:C = 4G² X`, DP `dN/dσ` | (11.6) | rel 6.6e-16 |
| same, `X` traceless but **skew** in the first pair | (11.6) | **rel 1.000** |
| J2 closed form == general `compute_tangent` | (11.12) | rel 1.2e-16 |
| `e_c/e_t` uniaxial-strain asymmetry | (3.3) | 0.189598 / 0.277190 = 1.462 |

The skew row is the point of the four preconditions in §11.3: `X` traceless in
both index pairs is not sufficient, and the failure is total rather than
marginal.

The closed forms are also checked continuously against central differences
through the graph by `tangent_checker` — see `plasticity.md` §5 for the load
paths and what that check can and cannot see.

---

## 14. What the derivations assume

Collected, because each one is a place where a plausible extension silently
produces wrong numbers.

**Isotropic elasticity, in three independent places.** `G_eff = M:C_e:N` is a
material constant only for isotropic `C_e` — otherwise the *residual* is wrong,
not just the tangent. The stress shortcut (2.2) is the second. The `4G²` collapse
(11.6) is the third. Measured with `C_e` made mildly orthotropic (1.6× on one
shear component): the closed-form tangent misses a central difference by 2.3 %,
the general `compute_tangent` by 1.9 %, and the stress shortcut by 5.19 absolute.
Percent-level errors with no diagnostic. This is why the materials build `C_e`
themselves.

**The volumetric part of `N` must be constant in `σ`.** This is the operative
form of the `4G²` precondition for a *future* yield function, and it is narrower
than "deviatoric". A pressure-dependent dilatancy `β(p) = β₀ + c·p` adds
`(c/9) I⊗I` to `dN/dσ`, traceless in neither index pair; `c = 1e-3` measures a
**35 %** tangent error, and the error scales with `K/G` rather than with `c`
relative to `β`. Cap models, Matsuoka-Nakai and Lade surfaces, and any smoothed
cone tip share the property. Nothing checks it.

**Small strain.** No objective rate, no frame rotation. Nothing detects the
violation.

**`H' > -G_eff`.** At or below that the yield residual has positive slope and
the return map has no admissible solution. Both materials throw rather than
clamp: clamping `Δλ` to zero returns the elastic stress, and on a uniaxial path
with `H' = -300` against `3G = 230.8` the equivalent stress climbed past
`σ₀ = 50` to 76.9 with `α` identically zero — 54 % outside the yield surface,
silently. Moderate softening works. The *apex* criterion also inverts under
softening, giving a missed apex with `q < 0`; that is still unguarded.

**`D = 3` for the pressure terms.** `dev` divides the trace by `D`, but
`p = tr(σ)/3` and the `η/3 I`, `β/3 I` terms in (4.5) use a hard 3. At `D = 3`
they agree. At `D = 2` they do not, so the DP normals and the deviator would use
different conventions. `material_policy_2d` exists but is registered by nothing
and instantiated by no test, so this is latent — as was the `3K IIvol` spelling
of §2 until it was measured at 30 % and fixed. If 2D is ever wanted, these are
the lines to revisit.

**The apex tangent is a branch tangent.** Verified consistent *on* the branch to
3.8e-8; a perturbation leaving the apex back onto the smooth cone is not covered
and cannot be by a central difference. At the true branch boundary `C₀₁₀₁` jumps
from 0 to 71.4 — inherent to a non-smooth return map.
