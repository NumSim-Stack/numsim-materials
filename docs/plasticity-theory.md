# Small-strain J2 and Drucker–Prager plasticity: from kinematics to the linearized stress

This is the theory behind `j2_plasticity`, `drucker_prager_plasticity` and
`j2_rk_plasticity`: the continuum statement, the discrete return map, and the
linearized stress each one hands back to the host solver. Every equation is
carried through to the expression that appears in the code, and the code line is
named where it does.

Its companion, [`plasticity.md`](plasticity.md), covers the *design* — why these
materials build their own elastic tangent, why the solver is reached by
`material_ref`, what is verified and what is not. This document is the
mathematics.

> Equations are LaTeX and render on GitHub. In a plain-text viewer they are
> readable as source.

**Conventions.** Small strain throughout. Tensors are $D$-dimensional, where
$D$ is `Traits::Dim`, which is $3$ for every registered material. Second-order
tensors are bold lower case, fourth-order blackboard bold. $\otimes$ is the
dyadic product (`tmech::otimes`), $:$ the double contraction
(`tmech::dcontract`), $\boldsymbol{I}$ the second-order identity and
$\mathbb{I}^{\mathrm{sym}}$ the symmetric fourth-order identity. The deviatoric
projector is

$$\mathbb{I}^{\mathrm{dev}} = \mathbb{I}^{\mathrm{sym}} - \tfrac{1}{D}\,\boldsymbol{I}\otimes\boldsymbol{I}$$

*(`plasticity_utils.h: make_IIdev`)*, so
$\operatorname{dev}(\boldsymbol{a}) = \mathbb{I}^{\mathrm{dev}} : \boldsymbol{a} = \boldsymbol{a} - \tfrac{1}{D}\operatorname{tr}(\boldsymbol{a})\,\boldsymbol{I}$.

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
9. [Drucker–Prager in closed form](#9-druckerprager-in-closed-form)
10. [The apex branch](#10-the-apex-branch)
11. [The linearized stress](#11-the-linearized-stress)
12. [The multi-stage variant](#12-the-multi-stage-variant)
13. [Equation-to-code map](#13-equation-to-code-map)
14. [What the derivations assume](#14-what-the-derivations-assume)

---

## 1. Kinematics

The strain is small and the decomposition additive:

$$\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}^e + \boldsymbol{\varepsilon}^p \tag{1.1}$$

with $\boldsymbol{\varepsilon}$ the symmetric small-strain tensor supplied by the
host (a `strain_source` in the graph), $\boldsymbol{\varepsilon}^e$ elastic and
$\boldsymbol{\varepsilon}^p$ plastic. Both $\boldsymbol{\varepsilon}^p$ and the
hardening variable $\kappa$ of §5 are **history variables**: they carry an old
and a new value across the step, and the driver — never the material — advances
them by calling `commit()`.

There is no multiplicative split, no rotation of the plastic frame, no objective
rate. $\boldsymbol{\varepsilon}^p$ is a plain tensor added to
$\boldsymbol{\varepsilon}^e$, which is what makes every expression below
algebraic rather than a differential equation in a rotating frame. It is valid
while strains and rotations are small; beyond that the model is out of scope and
nothing in the code detects it.

In rate form, with $\dot\lambda \ge 0$ the plastic multiplier and
$\boldsymbol{N}$ the flow direction of §4:

$$\dot{\boldsymbol{\varepsilon}}^p = \dot\lambda\,\boldsymbol{N} \tag{1.2}$$

$\dot{\boldsymbol{\varepsilon}}^p$ is the only source of dissipation in the
model; the elastic response is reversible by construction.

---

## 2. Elastic law

The elastic response is linear and **isotropic**:

$$\boldsymbol{\sigma} = \mathbb{C}_e : \boldsymbol{\varepsilon}^e = \mathbb{C}_e : (\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p) \tag{2.1}$$

$$\mathbb{C}_e = K\,\boldsymbol{I}\otimes\boldsymbol{I} + 2G\,\mathbb{I}^{\mathrm{dev}}$$

*(`plasticity_utils.h: make_isotropic_tangent`)*, with $K$ the bulk and $G$ the
shear modulus. Contracting once gives the identity every fast path in the code
relies on:

$$\mathbb{C}_e : \boldsymbol{a} = K\operatorname{tr}(\boldsymbol{a})\,\boldsymbol{I} + 2G\operatorname{dev}(\boldsymbol{a}) \tag{2.2}$$

For $\boldsymbol{a}$ deviatoric this is $2G\boldsymbol{a}$; for
$\boldsymbol{a} = \boldsymbol{I}$ it is $D K \boldsymbol{I}$. Splitting (2.1),

$$p = \tfrac{1}{3}\operatorname{tr}(\boldsymbol{\sigma}) = K\operatorname{tr}(\boldsymbol{\varepsilon}^e), \qquad
\boldsymbol{s} = \operatorname{dev}(\boldsymbol{\sigma}) = 2G\operatorname{dev}(\boldsymbol{\varepsilon}^e) \tag{2.3}$$

for $D = 3$.

The bulk term is written $K\,\boldsymbol{I}\otimes\boldsymbol{I}$, not
$3K\,\mathbb{I}^{\mathrm{vol}}$. Those coincide only at $D = 3$, because
$\mathbb{I}^{\mathrm{vol}} = \boldsymbol{I}\otimes\boldsymbol{I}/D$, and the
shortcut (2.2) assumes the first. Writing the second made a material disagree
with itself by 30 % at $D = 2$ — see §14.

Every plasticity material builds $\mathbb{C}_e$ itself from its own $K$ and $G$
rather than reading a rank-four tangent from an elastic material. That is not a
convenience: §11 and §14 show the closed forms are *false* for an anisotropic
$\mathbb{C}_e$, so accepting one would advertise a generality the model cannot
honour.

---

## 3. Yield functions

Both models are pressure-and-deviator functions of the stress, but they use
**different normalizations of the deviatoric invariant**, and that difference
propagates into every effective modulus below. It is a genuine difference in the
two files, not a presentational one.

### 3.1 J2 (von Mises)

$$J_2 = \tfrac{1}{2}\,\boldsymbol{s}:\boldsymbol{s}, \qquad
\sigma_{\mathrm{eq}} = \sqrt{3J_2} = \sqrt{\tfrac{3}{2}\,\boldsymbol{s}:\boldsymbol{s}}$$

$$F = \sigma_{\mathrm{eq}} - \sigma_0 - H(\kappa) \tag{3.1}$$

*(`yield_functions.h: equivalent_stress`, `trial_yield`)*

$\sigma_{\mathrm{eq}}$ is the uniaxial-equivalent stress: in uniaxial tension
$\sigma_{\mathrm{eq}} = |\sigma_{11}|$, so $\sigma_0$ is the uniaxial yield
stress and can be read off a tensile test. The yield surface is a cylinder about
the hydrostatic axis — pressure-insensitive, no apex, `needs_apex_return`
returns `false` unconditionally.

### 3.2 Drucker–Prager

$$q = \sqrt{J_2} = \sqrt{\tfrac{1}{2}\,\boldsymbol{s}:\boldsymbol{s}}, \qquad
p = \tfrac{1}{3}\operatorname{tr}(\boldsymbol{\sigma})$$

$$F = q + \eta\,p - k - H(\kappa) \tag{3.2}$$

*(`drucker_prager_yield_function.h: equivalent_stress`, `trial_yield`)*

Note $q = \sqrt{J_2}$, **without** the $\sqrt{3}$. The surface is a cone about
the hydrostatic axis:

- $\eta$ — the friction (pressure) coefficient. $\eta > 0$ makes tensile
  pressure destabilising, so the material yields earlier in tension than in
  compression.
- $k$ — the cohesion, the `sigma_0` parameter. At $p = 0$ it is the yield value
  of $q$.
- $\eta = 0$ reduces the cone to a cylinder, which is von Mises up to the
  normalization difference above: $\sigma_{\mathrm{eq}} = \sqrt{3}\,q$, so a DP
  material with $\eta = 0$ and cohesion $k$ behaves as J2 with
  $\sigma_0 = \sqrt{3}\,k$.

The cone closes at the **apex** on the hydrostatic axis, where $q = 0$ and
$\eta p = k + H$. That point is not smooth, and §10 is about it.

Because $\eta > 0$ penalises tensile pressure, uniaxial *strain* yields at
different magnitudes in the two directions. With $q = 2G|e|/\sqrt{3}$ and
$p = Ke$ for a uniaxial strain $e$,

$$e_{\mathrm{tension}} = \frac{k}{2G/\sqrt{3} + \eta K}, \qquad
|e_{\mathrm{compression}}| = \frac{k}{2G/\sqrt{3} - \eta K} \tag{3.3}$$

which coincide only at $\eta = 0$.
`DruckerPragerPressure.YieldsEarlierInTensionThanInCompression` pins exactly this
ratio, which makes it a test of the friction term rather than of yielding in
general.

---

## 4. Flow rule and plastic potential

The flow direction comes from a plastic potential $Q$, and the *yield* normal,
which appears in the consistency condition and hence in the tangent, from $F$:

$$\boldsymbol{N} = \frac{\partial Q}{\partial\boldsymbol{\sigma}}, \qquad
\boldsymbol{M} = \frac{\partial F}{\partial\boldsymbol{\sigma}} \tag{4.1}$$

$\boldsymbol{M} = \boldsymbol{N}$ when the flow is **associative**,
$\boldsymbol{M} \neq \boldsymbol{N}$ when it is not. Keeping the two apart is
what lets one implementation carry both models.

### 4.1 J2 — associative

From $\sigma_{\mathrm{eq}}^2 = \tfrac{3}{2}\boldsymbol{s}:\boldsymbol{s}$,

$$2\sigma_{\mathrm{eq}}\frac{\partial\sigma_{\mathrm{eq}}}{\partial\boldsymbol{\sigma}}
= 3\,\boldsymbol{s} : \mathbb{I}^{\mathrm{dev}} = 3\,\boldsymbol{s}
\qquad\Longrightarrow\qquad
\boldsymbol{N} = \boldsymbol{M} = \frac{3}{2}\frac{\boldsymbol{s}}{\sigma_{\mathrm{eq}}} \tag{4.2}$$

*(`yield_functions.h: flow_normal`)*. Two properties are used repeatedly:

$$\operatorname{tr}(\boldsymbol{N}) = 0, \qquad
\boldsymbol{N}:\boldsymbol{N} = \frac{9}{4}\frac{\boldsymbol{s}:\boldsymbol{s}}{\sigma_{\mathrm{eq}}^2} = \frac{3}{2} \tag{4.3}$$

$\operatorname{tr}(\boldsymbol{N}) = 0$ is why the volumetric term of (2.2) drops
out of every J2 expression and why
$\mathbb{C}_e : \boldsymbol{N} = 2G\boldsymbol{N}$ exactly.

### 4.2 Drucker–Prager — non-associative

The potential replaces the friction coefficient $\eta$ by a **dilatancy**
coefficient $\beta$, so $Q = q + \beta p$ and, using
$\partial q/\partial\boldsymbol{\sigma} = \boldsymbol{s}/(2q)$ and
$\partial p/\partial\boldsymbol{\sigma} = \boldsymbol{I}/3$,

$$\boldsymbol{N} = \frac{\boldsymbol{s}}{2q} + \frac{\beta}{3}\boldsymbol{I}, \qquad
\boldsymbol{M} = \frac{\boldsymbol{s}}{2q} + \frac{\eta}{3}\boldsymbol{I} \tag{4.4}$$

*(`drucker_prager_yield_function.h: flow_normal`, `yield_normal`)*. Now

$$\operatorname{tr}(\boldsymbol{N}) = \beta \tag{4.5}$$

so plastic flow is **dilatant**: $\beta > 0$ produces plastic volume increase.
This is the physical reason the model is non-associative — taking
$\beta = \eta$ would tie dilatancy to friction and, for realistic friction
angles, over-predict volume change badly. The price is that $\mathbb{C}_{ep}$
loses its major symmetry (§11.7).

The $1/3$ in (4.4) is a hard $3$, while $\operatorname{dev}$ uses $1/D$. At
$D = 3$ they agree; see §14.

---

## 5. Hardening and the internal variable

One scalar internal variable $\kappa$ with

$$\dot\kappa = \dot\lambda \tag{5.1}$$

so $\kappa$ is the accumulated plastic multiplier. For J2 that is exactly the
conventional equivalent plastic strain, and the check is worth doing because it
fixes what $\sigma_0$ and $H$ mean:

$$\lVert\dot{\boldsymbol{\varepsilon}}^p\rVert = \dot\lambda\sqrt{\boldsymbol{N}:\boldsymbol{N}} = \dot\lambda\sqrt{\tfrac{3}{2}}
\qquad\Longrightarrow\qquad
\dot\varepsilon^p_{\mathrm{eq}} \equiv \sqrt{\tfrac{2}{3}}\,\lVert\dot{\boldsymbol{\varepsilon}}^p\rVert = \dot\lambda \tag{5.2}$$

and the plastic work is

$$\boldsymbol{\sigma} : \dot{\boldsymbol{\varepsilon}}^p
= \dot\lambda\,\boldsymbol{\sigma}:\boldsymbol{N}
= \dot\lambda\,\frac{3}{2}\frac{\boldsymbol{s}:\boldsymbol{s}}{\sigma_{\mathrm{eq}}}
= \dot\lambda\,\sigma_{\mathrm{eq}} \tag{5.3}$$

so $(\sigma_{\mathrm{eq}}, \kappa)$ are work-conjugate. $H(\kappa)$ is therefore
the uniaxial hardening curve, directly comparable to a tensile test.

For Drucker–Prager $\kappa$ is the multiplier itself and is *not* a strain
measure of the same kind — the normalization $q = \sqrt{J_2}$ and the volumetric
part of $\boldsymbol{N}$ both enter — so $k$ and $H$ there are cone parameters,
not uniaxial ones.

The material never evaluates $H$ itself. It reads two properties from a
hardening material through Local edges: $H$ as `"hardening_stress"` and
$H' = \mathrm{d}H/\mathrm{d}\kappa$ as `"hardening_modulus"`.
`linear_isotropic_hardening` supplies $H = K\kappa$, $H' = K$. Any other law
that publishes the same two properties works unchanged.

$H' < 0$ is softening. It is admissible while $H' > -G_{\mathrm{eff}}$; at or
below that the local problem has no admissible solution and both materials throw
— §14.

---

## 6. Loading, unloading, consistency

The Karush–Kuhn–Tucker conditions:

$$F \le 0, \qquad \dot\lambda \ge 0, \qquad \dot\lambda F = 0 \tag{6.1}$$

and, during plastic flow, the consistency condition $\dot F = 0$:

$$\boldsymbol{M} : \dot{\boldsymbol{\sigma}} - H'\dot\kappa = 0 \tag{6.2}$$

Substituting
$\dot{\boldsymbol{\sigma}} = \mathbb{C}_e : (\dot{\boldsymbol{\varepsilon}} - \dot\lambda\boldsymbol{N})$
gives the multiplier in rate form,

$$\dot\lambda = \frac{\boldsymbol{M} : \mathbb{C}_e : \dot{\boldsymbol{\varepsilon}}}
{\boldsymbol{M} : \mathbb{C}_e : \boldsymbol{N} + H'} \tag{6.3}$$

The denominator is the quantity the code calls the **effective modulus** plus
the hardening modulus. It is worth naming, because it is the same scalar in the
rate form, in the discrete residual, and in the tangent:

$$G_{\mathrm{eff}} \equiv \boldsymbol{M} : \mathbb{C}_e : \boldsymbol{N} \tag{6.4}$$

*(`*_yield_function.h: effective_modulus`)*

### 6.1 $G_{\mathrm{eff}}$ for the two models

**J2.** $\mathbb{C}_e : \boldsymbol{N} = 2G\boldsymbol{N}$ since
$\operatorname{tr}(\boldsymbol{N}) = 0$, so

$$G_{\mathrm{eff}} = \boldsymbol{N} : \mathbb{C}_e : \boldsymbol{N}
= 2G\,(\boldsymbol{N}:\boldsymbol{N}) = 2G\cdot\tfrac{3}{2} = 3G \tag{6.5}$$

**Drucker–Prager.** From (2.2) and (4.4),

$$\mathbb{C}_e : \boldsymbol{N} = 2G\frac{\boldsymbol{s}}{2q} + K\beta\boldsymbol{I}
= G\frac{\boldsymbol{s}}{q} + K\beta\boldsymbol{I} \tag{6.6}$$

$$\begin{aligned}
G_{\mathrm{eff}} &= \boldsymbol{M} : \mathbb{C}_e : \boldsymbol{N}
= \left[\frac{\boldsymbol{s}}{2q} + \frac{\eta}{3}\boldsymbol{I}\right] :
  \left[G\frac{\boldsymbol{s}}{q} + K\beta\boldsymbol{I}\right] \\[2pt]
&= G\frac{\boldsymbol{s}:\boldsymbol{s}}{2q^2} + \frac{\eta}{3}K\beta\operatorname{tr}(\boldsymbol{I})
= G + K\eta\beta
\end{aligned} \tag{6.7}$$

using $\boldsymbol{s}:\boldsymbol{s} = 2q^2$,
$\operatorname{tr}(\boldsymbol{s}) = 0$, $\boldsymbol{s}:\boldsymbol{I} = 0$ and
$\operatorname{tr}(\boldsymbol{I}) = 3$. The cross terms vanish; only the shear
term and the friction–dilatancy product survive. Both match the two
`effective_modulus` implementations exactly.

$G_{\mathrm{eff}}$ is the first place isotropy is load-bearing: for an
anisotropic $\mathbb{C}_e$, $\boldsymbol{M}:\mathbb{C}_e:\boldsymbol{N}$ is
state-dependent and no longer a material constant, so the *residual itself*
would be wrong — not merely the tangent.

---

## 7. The return map

### 7.1 Discretisation

Backward Euler on (1.2) over a step, with the flow direction evaluated at the
**trial** state:

$$\boldsymbol{\varepsilon}^p_{n+1} = \boldsymbol{\varepsilon}^p_n + \Delta\lambda\,\boldsymbol{N}^{\mathrm{tr}},
\qquad \kappa_{n+1} = \kappa_n + \Delta\lambda \tag{7.1}$$

$\Delta\lambda \ge 0$ is the one unknown of the step.

### 7.2 Trial state

Freeze the plastic strain and load elastically:

$$\boldsymbol{\sigma}^{\mathrm{tr}} = \mathbb{C}_e : (\boldsymbol{\varepsilon}_{n+1} - \boldsymbol{\varepsilon}^p_n),
\qquad F^{\mathrm{tr}} = F(\boldsymbol{\sigma}^{\mathrm{tr}}, \kappa_n) \tag{7.2}$$

*(`plasticity_utils.h: compute_trial`)*

$F^{\mathrm{tr}} \le 0$ ⟹ the step is elastic:
$\boldsymbol{\sigma} = \boldsymbol{\sigma}^{\mathrm{tr}}$,
$\mathbb{C}_{ep} = \mathbb{C}_e$, history unchanged. That branch is not an
optimisation, it is the KKT case $\dot\lambda = 0$.

$F^{\mathrm{tr}} > 0$ ⟹ the return map runs. **This is the only case in which a
solve happens at all**, which is why the scalar solver is reached by
`material_ref` rather than by a graph edge — see `plasticity.md` §9.

### 7.3 Reduction to one scalar

Substituting (7.1) into (2.1),

$$\boldsymbol{\sigma} = \mathbb{C}_e : (\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p_n - \Delta\lambda\boldsymbol{N}^{\mathrm{tr}})
= \boldsymbol{\sigma}^{\mathrm{tr}} - \Delta\lambda\,(\mathbb{C}_e : \boldsymbol{N}^{\mathrm{tr}}) \tag{7.3}$$

The whole tensor update is therefore known once the single scalar
$\Delta\lambda$ is. What makes the reduction exact rather than an approximation
is that $\boldsymbol{N}$ does **not** rotate during the return: §8.1 and §9.1
show, for each model separately, that the returned deviator stays parallel to the
trial deviator. That is the property that turns a tensor-valued nonlinear system
into a scalar one.

Contracting (7.3) with $\boldsymbol{M}$ and using (6.4) gives the residual:

$$r(\Delta\lambda) = \varphi^{\mathrm{tr}} - G_{\mathrm{eff}}\Delta\lambda - Y_0 - H(\kappa_n + \Delta\lambda) = 0$$

$$r'(\Delta\lambda) = -(G_{\mathrm{eff}} + H') \tag{7.4}$$

*(`*_yield_function.h: residual`, `jacobian`)*

where $\varphi^{\mathrm{tr}}$ is the **modified equivalent stress** at the trial
state — $\sigma_{\mathrm{eq}}^{\mathrm{tr}}$ for J2,
$q^{\mathrm{tr}} + \eta p^{\mathrm{tr}}$ for DP — and $Y_0$ is $\sigma_0$ or $k$.
This is one scalar equation in one unknown, solved by `newton_scalar`. For linear
hardening it is linear, so

$$\Delta\lambda = \frac{F^{\mathrm{tr}}}{G_{\mathrm{eff}} + H'} \tag{7.5}$$

and Newton lands on the root in a single step. That is also the reason forward
Euler, backward Euler and the monolithic map must give **identical** answers for
linear hardening, which `J2RKStage.ExplicitAndImplicitStagesSolveTheSameEquation`
asserts to $10^{-12}$ relative rather than to a tolerance.

---

## 8. J2 in closed form

### 8.1 Radial return

From (7.3) with $\mathbb{C}_e : \boldsymbol{N} = 2G\boldsymbol{N}$ and
$\boldsymbol{N}^{\mathrm{tr}} = \tfrac{3}{2}\boldsymbol{s}^{\mathrm{tr}}/\sigma_{\mathrm{eq}}^{\mathrm{tr}}$:

$$\boldsymbol{s} = \boldsymbol{s}^{\mathrm{tr}} - 2G\Delta\lambda\boldsymbol{N}^{\mathrm{tr}}
= \boldsymbol{s}^{\mathrm{tr}}\left(1 - \frac{3G\Delta\lambda}{\sigma_{\mathrm{eq}}^{\mathrm{tr}}}\right) \tag{8.1}$$

The returned deviator is a **scalar multiple** of the trial deviator: the return
is radial in deviatoric space and $\boldsymbol{N} = \boldsymbol{N}^{\mathrm{tr}}$
exactly, not approximately. Taking the norm,

$$\sigma_{\mathrm{eq}} = \sigma_{\mathrm{eq}}^{\mathrm{tr}} - 3G\Delta\lambda \tag{8.2}$$

which is (7.3) contracted, and recovers $G_{\mathrm{eff}} = 3G$. The volumetric
stress is untouched, since $\operatorname{tr}(\boldsymbol{N}) = 0$.

### 8.2 The update

$$\boldsymbol{\sigma} = \boldsymbol{\sigma}^{\mathrm{tr}} - 2G\Delta\lambda\,\boldsymbol{N},
\qquad \boldsymbol{\varepsilon}^p_{n+1} = \boldsymbol{\varepsilon}^p_n + \Delta\lambda\boldsymbol{N},
\qquad \kappa_{n+1} = \kappa_n + \Delta\lambda \tag{8.3}$$

*(`j2_plasticity.h: compute`)*. The stress line uses
$\mathbb{C}_e : \boldsymbol{N} = 2G\boldsymbol{N}$ rather than a rank-four
contraction — algebraically identical, and it measures 33 ns against ~245 ns for
the whole step.

### 8.3 Residual

$$r(\Delta\lambda) = \sigma_{\mathrm{eq}}^{\mathrm{tr}} - 3G\Delta\lambda - \sigma_0 - H(\kappa_n + \Delta\lambda),
\qquad r'(\Delta\lambda) = -3G - H' \tag{8.4}$$

---

## 9. Drucker–Prager in closed form

### 9.1 Smooth cone return

From (6.6),

$$\boldsymbol{s} = \boldsymbol{s}^{\mathrm{tr}}\left(1 - \frac{G\Delta\lambda}{q^{\mathrm{tr}}}\right),
\qquad p = p^{\mathrm{tr}} - K\beta\Delta\lambda,
\qquad q = q^{\mathrm{tr}} - G\Delta\lambda \tag{9.1}$$

Deviatorically the return is again radial, so $\boldsymbol{N}$ does not rotate.
The volumetric part *is* affected, because
$\operatorname{tr}(\boldsymbol{N}) = \beta \neq 0$ — this is the essential
difference from J2 and the reason the volumetric term of (2.2) may not be
dropped.

Substituting into (3.2):

$$\begin{aligned}
F &= (q^{\mathrm{tr}} - G\Delta\lambda) + \eta(p^{\mathrm{tr}} - K\beta\Delta\lambda) - k - H \\[2pt]
  &= (q^{\mathrm{tr}} + \eta p^{\mathrm{tr}}) - (G + K\eta\beta)\Delta\lambda - k - H
\end{aligned} \tag{9.2}$$

recovering $\varphi^{\mathrm{tr}} = q^{\mathrm{tr}} + \eta p^{\mathrm{tr}}$ and
$G_{\mathrm{eff}} = G + K\eta\beta$ — (6.7) again, now from the discrete side.

### 9.2 The update

$$\boldsymbol{\sigma} = \boldsymbol{\sigma}^{\mathrm{tr}} - \Delta\lambda\left[2G\operatorname{dev}(\boldsymbol{N}) + K\operatorname{tr}(\boldsymbol{N})\boldsymbol{I}\right],
\qquad \boldsymbol{\varepsilon}^p_{n+1} = \boldsymbol{\varepsilon}^p_n + \Delta\lambda\boldsymbol{N} \tag{9.3}$$

*(`drucker_prager_plasticity.h: do_smooth_return`)*. The volumetric term does not
drop out as it does for J2, so both terms of (2.2) are kept.

### 9.3 Branch condition

(9.1) is only meaningful while the deviatoric correction does not overshoot the
tip. It stops being so at

$$G\Delta\lambda \ge q^{\mathrm{tr}} \tag{9.4}$$

*(`drucker_prager_yield_function.h: needs_apex_return`)* — exactly where (9.1)
would drive $q$ to zero and then negative, so the returned deviator would flip
sign. Beyond it the smooth branch is not the answer and §10 takes over.

**The criterion is applied to the converged $\Delta\lambda$, after the Newton has
run.** It is monotone increasing in $\Delta\lambda$, so feeding it an *upper*
bound on $\Delta\lambda$ — for instance the zero-hardening
$F^{\mathrm{tr}}/G_{\mathrm{eff}}$ — and concluding "apex" is invalid: an upper
bound crossing a threshold says nothing about the true value. Only the
contrapositive holds. A sound cheap pre-check would need a *lower* bound on
$\Delta\lambda$, which needs an upper bound on $H'$, which is not available; so
the smooth Newton always runs. `plasticity.md` §4 records what the invalid
version cost.

---

## 10. The apex branch

At the apex the deviatoric stress vanishes and the cone condition degenerates to
a purely volumetric statement:

$$\boldsymbol{s} = \boldsymbol{0}, \qquad \eta p = k + H(\kappa_n + \Delta\kappa) \tag{10.1}$$

The algorithm is a **projection**, not a single-multiplier flow update: the
deviatoric plastic strain is set directly to enforce $q = 0$,

$$\operatorname{dev}(\boldsymbol{\varepsilon}^p_{n+1}) = \operatorname{dev}(\boldsymbol{\varepsilon}),
\qquad \operatorname{tr}(\boldsymbol{\varepsilon}^p_{n+1}) = \operatorname{tr}(\boldsymbol{\varepsilon}^p_n) + \beta\Delta\kappa \tag{10.2}$$

*(`drucker_prager_yield_function.h: apex_plastic_strain`)*, while $\Delta\kappa$
follows from (10.1). With $p = p^{\mathrm{tr}} - K\beta\Delta\kappa$,

$$\eta(p^{\mathrm{tr}} - K\beta\Delta\kappa) - k - H(\kappa_n + \Delta\kappa) = 0
\;\Longleftrightarrow\;
\varphi^{\mathrm{apex}} - G_{\mathrm{eff}}^{\mathrm{apex}}\Delta\kappa - k - H = 0 \tag{10.3}$$

$$\varphi^{\mathrm{apex}} = \eta p^{\mathrm{tr}}, \qquad
G_{\mathrm{eff}}^{\mathrm{apex}} = K\eta\beta$$

the same scalar residual as (7.4) with two substituted arguments, which is why
one `solve_scalar_return` serves both branches.

### 10.1 Apex tangent

Differentiate (10.1) and $p = p^{\mathrm{tr}} - K\beta\Delta\kappa$:

$$\eta\,\mathrm{d}p = H'\mathrm{d}\Delta\kappa, \qquad
\mathrm{d}p = \mathrm{d}p^{\mathrm{tr}} - K\beta\,\mathrm{d}\Delta\kappa$$

$$\Longrightarrow\quad
\eta\,\mathrm{d}p^{\mathrm{tr}} = (H' + K\eta\beta)\,\mathrm{d}\Delta\kappa
\quad\Longrightarrow\quad
\mathrm{d}p = \mathrm{d}p^{\mathrm{tr}}\,\frac{H'}{H' + K\eta\beta} \tag{10.4}$$

With $\mathrm{d}p^{\mathrm{tr}} = K\,\boldsymbol{I}:\mathrm{d}\boldsymbol{\varepsilon}$
and $\boldsymbol{\sigma} = p\boldsymbol{I}$ on the branch,

$$\mathbb{C}_{ep}^{\mathrm{apex}} = \frac{K H'}{K\eta\beta + H'}\;\boldsymbol{I}\otimes\boldsymbol{I} \tag{10.5}$$

*(`drucker_prager_yield_function.h: apex_tangent`)*

Two properties of (10.5) matter to the host:

- It is **rank one in every case**, not only the degenerate one. It is a multiple
  of $\boldsymbol{I}\otimes\boldsymbol{I}$, and (10.2) pins
  $\operatorname{dev}(\boldsymbol{\varepsilon}^p) = \operatorname{dev}(\boldsymbol{\varepsilon})$
  so $\operatorname{dev}(\boldsymbol{\sigma}) \equiv \boldsymbol{0}$ for *any*
  perturbation — every deviatoric mode has zero stiffness. An element with all
  its Gauss points at the apex is singular whatever the hardening.
- $H' = 0$ gives a **zero** tangent regardless of $K\eta\beta$, since the
  numerator is $KH'$.

It is a **branch** tangent: valid for perturbations that stay on the apex. The
return map is genuinely non-smooth there, so no single tangent describes both
sides, and a central difference across the boundary measures neither.

---

## 11. The linearized stress

This is the object the host Newton actually consumes:

$$\mathrm{d}\boldsymbol{\sigma} = \mathbb{C}_{ep} : \mathrm{d}\boldsymbol{\varepsilon},
\qquad \mathbb{C}_{ep} = \frac{\partial\boldsymbol{\sigma}_{n+1}}{\partial\boldsymbol{\varepsilon}_{n+1}} \tag{11.1}$$

the derivative of the **algorithm** — the consistent or algorithmic tangent —
not of the continuum rate law. The distinction is not academic: §11.5 shows the
two differ by a term proportional to $\Delta\lambda$, and using the continuum one
degrades the host Newton from quadratic to linear convergence.

### 11.1 The two dependencies

From (7.3), $\boldsymbol{\sigma}$ depends on $\boldsymbol{\varepsilon}$ twice:
explicitly, and through $\Delta\lambda(\boldsymbol{\varepsilon})$.

$$\boldsymbol{\sigma}(\boldsymbol{\varepsilon}) = \mathbb{C}_e : \left(\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p_n - \Delta\lambda(\boldsymbol{\varepsilon})\,\boldsymbol{N}^{\mathrm{tr}}(\boldsymbol{\varepsilon})\right) \tag{11.2}$$

so by the chain rule

$$\mathbb{C}_{ep} = \mathbb{A} + \frac{\partial\boldsymbol{\sigma}}{\partial\Delta\lambda} \otimes \frac{\mathrm{d}\Delta\lambda}{\mathrm{d}\boldsymbol{\varepsilon}} \tag{11.3}$$

with $\mathbb{A}$ the derivative at frozen $\Delta\lambda$.

### 11.2 The frozen-multiplier part

$\boldsymbol{N}^{\mathrm{tr}}$ depends on $\boldsymbol{\varepsilon}$ through
$\boldsymbol{\sigma}^{\mathrm{tr}} = \mathbb{C}_e : (\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p_n)$, so

$$\frac{\mathrm{d}\boldsymbol{N}^{\mathrm{tr}}}{\mathrm{d}\boldsymbol{\varepsilon}}
= \frac{\partial\boldsymbol{N}}{\partial\boldsymbol{\sigma}} : \mathbb{C}_e$$

$$\mathbb{A} = \mathbb{C}_e : \left(\mathbb{I}^{\mathrm{sym}} - \Delta\lambda\frac{\mathrm{d}\boldsymbol{N}^{\mathrm{tr}}}{\mathrm{d}\boldsymbol{\varepsilon}}\right)
= \mathbb{C}_e - \Delta\lambda\,\mathbb{C}_e : \frac{\partial\boldsymbol{N}}{\partial\boldsymbol{\sigma}} : \mathbb{C}_e \tag{11.4}$$

The two flow-normal derivatives:

$$\text{J2:}\quad \frac{\partial\boldsymbol{N}}{\partial\boldsymbol{\sigma}}
= \frac{\tfrac{3}{2}\mathbb{I}^{\mathrm{dev}} - \boldsymbol{N}\otimes\boldsymbol{N}}{\sigma_{\mathrm{eq}}}$$

$$\text{DP:}\quad \frac{\partial\boldsymbol{N}}{\partial\boldsymbol{\sigma}}
= \frac{\mathbb{I}^{\mathrm{dev}} - \boldsymbol{s}\otimes\boldsymbol{s}/(2J_2)}{2q} \tag{11.5}$$

*(`*_yield_function.h: flow_normal_stress_derivative`)*

Both are $\mathbb{I}^{\mathrm{dev}}$ minus a dyad of a deviatoric symmetric
tensor. Neither depends on the volumetric part of $\boldsymbol{N}$ — for DP the
$\tfrac{\beta}{3}\boldsymbol{I}$ term is constant in $\boldsymbol{\sigma}$ and
differentiates away, which is exactly the property §14 warns a future yield
function not to break.

### 11.3 The $4G^2$ collapse

$\mathbb{A}$ as written needs two rank-four × rank-four contractions. For
isotropic $\mathbb{C}_e$ they collapse:

$$\mathbb{C}_e : \mathbb{X} : \mathbb{C}_e = 4G^2\,\mathbb{X} \tag{11.6}$$

**with four preconditions**, not two: $\mathbb{X}$ must be *traceless* **and**
*minor-symmetric* in the first index pair, and likewise in the second. From
$\mathbb{C}_e = 3K\,\mathbb{I}^{\mathrm{vol}} + 2G\,\mathbb{I}^{\mathrm{dev}}$,
the volumetric term drops only if $\mathbb{X}$ is traceless, and
$\mathbb{I}^{\mathrm{dev}} : \mathbb{X}$ returns $\mathbb{X}$ only if
$\mathbb{I}^{\mathrm{sym}} : \mathbb{X}$ does, which is symmetry. A traceless
$\mathbb{X}$ that is skew in the first pair gives **100 % error**.

Both derivatives in (11.5) satisfy all four structurally, so

$$\mathbb{A} = \mathbb{C}_e - 4G^2\Delta\lambda\,\frac{\partial\boldsymbol{N}}{\partial\boldsymbol{\sigma}} \tag{11.7}$$

*(`plasticity_utils.h: compute_tangent`)*. Verified to $2.5\times10^{-16}$
against the explicit double contraction for both models.

### 11.4 The rank-one part

The multiplier is defined implicitly by
$r(\Delta\lambda, \boldsymbol{\varepsilon}) = 0$, so by the implicit function
theorem

$$\frac{\mathrm{d}\Delta\lambda}{\mathrm{d}\boldsymbol{\varepsilon}}
= -\left(\frac{\partial r}{\partial\Delta\lambda}\right)^{-1}\frac{\partial r}{\partial\boldsymbol{\varepsilon}} \tag{11.8}$$

From §7.3,
$r = \varphi^{\mathrm{tr}}(\boldsymbol{\varepsilon}) - G_{\mathrm{eff}}\Delta\lambda - Y_0 - H(\kappa_n + \Delta\lambda)$,
and $\varphi^{\mathrm{tr}}$ depends on $\boldsymbol{\varepsilon}$ only through
$\boldsymbol{\sigma}^{\mathrm{tr}}$:

$$\frac{\partial r}{\partial\boldsymbol{\varepsilon}} = \frac{\partial\varphi}{\partial\boldsymbol{\sigma}} : \mathbb{C}_e = \boldsymbol{M} : \mathbb{C}_e,
\qquad
\frac{\partial r}{\partial\Delta\lambda} = -(G_{\mathrm{eff}} + H') = -(\boldsymbol{M} : \mathbb{C}_e : \boldsymbol{N} + H') \tag{11.9}$$

and from (7.3),
$\partial\boldsymbol{\sigma}/\partial\Delta\lambda = -\mathbb{C}_e : \boldsymbol{N}$.
Assembling (11.3):

$$\boxed{\;
\mathbb{C}_{ep} = \mathbb{C}_e - 4G^2\Delta\lambda\,\frac{\partial\boldsymbol{N}}{\partial\boldsymbol{\sigma}}
- \frac{(\mathbb{C}_e : \boldsymbol{N}) \otimes (\boldsymbol{M} : \mathbb{C}_e)}
{\boldsymbol{M} : \mathbb{C}_e : \boldsymbol{N} + H'}
\;} \tag{11.10}$$

which is `compute_tangent` line for line.

### 11.5 The continuum tangent, and why it is not enough

Setting $\Delta\lambda = 0$ in (11.10) leaves

$$\mathbb{C}_{ep}^{\mathrm{cont}} = \mathbb{C}_e - \frac{(\mathbb{C}_e : \boldsymbol{N}) \otimes (\boldsymbol{M} : \mathbb{C}_e)}{G_{\mathrm{eff}} + H'} \tag{11.11}$$

which is the classical elastoplastic tangent obtained from the rate form (6.3)
alone. The difference is the
$4G^2\Delta\lambda\,\partial\boldsymbol{N}/\partial\boldsymbol{\sigma}$ term,
which accounts for the *rotation of the flow direction over the finite step*. It
vanishes only in the limit of an infinitesimal step. Keeping it is what makes
the tangent consistent with the discrete update and the host Newton quadratic.

### 11.6 J2 in closed form

For J2, $\boldsymbol{M} = \boldsymbol{N}$,
$\mathbb{C}_e : \boldsymbol{N} = 2G\boldsymbol{N}$,
$\boldsymbol{M} : \mathbb{C}_e : \boldsymbol{N} = 3G$, and
$\partial\boldsymbol{N}/\partial\boldsymbol{\sigma}$ is (11.5). Substituting into
(11.10):

$$\begin{aligned}
\mathbb{A} &= \mathbb{C}_e - 4G^2\Delta\lambda\,\frac{\tfrac{3}{2}\mathbb{I}^{\mathrm{dev}} - \boldsymbol{N}\otimes\boldsymbol{N}}{\sigma_{\mathrm{eq}}} \\[2pt]
&= \mathbb{C}_e - \frac{6G^2\Delta\lambda}{\sigma_{\mathrm{eq}}}\mathbb{I}^{\mathrm{dev}}
+ \frac{4G^2\Delta\lambda}{\sigma_{\mathrm{eq}}}\boldsymbol{N}\otimes\boldsymbol{N}
\end{aligned}$$

$$\text{rank-one term} = -\frac{(2G\boldsymbol{N})\otimes(2G\boldsymbol{N})}{3G + H'}
= -\frac{4G^2}{3G + H'}\boldsymbol{N}\otimes\boldsymbol{N}$$

Collecting the two $\boldsymbol{N}\otimes\boldsymbol{N}$ contributions:

$$\mathbb{C}_{ep} = \mathbb{C}_e - a\,\mathbb{I}^{\mathrm{dev}} + b\,\boldsymbol{N}\otimes\boldsymbol{N}$$

$$a = \frac{6G^2\Delta\lambda}{\sigma_{\mathrm{eq}}^{\mathrm{tr}}},
\qquad
b = \frac{4G^2\Delta\lambda}{\sigma_{\mathrm{eq}}^{\mathrm{tr}}} - \frac{4G^2}{3G + H'} \tag{11.12}$$

*(`j2_plasticity.h: m_tangent`)* — term for term what the code computes. This is
the Simo–Hughes radial-return tangent under the
$\sigma_{\mathrm{eq}} = \sqrt{3J_2}$,
$\Delta\lambda = \Delta\varepsilon^p_{\mathrm{eq}}$ normalization; converting
with $\Delta\gamma = \sqrt{3/2}\,\Delta\lambda$ reproduces their $2G\theta$ /
$-2G\bar\theta$ coefficients identically.

$\mathbb{C}_{ep}$ here is **symmetric**: $\boldsymbol{M} = \boldsymbol{N}$, so
the dyad in (11.10) is $\boldsymbol{N}\otimes\boldsymbol{N}$.

### 11.7 Drucker–Prager in closed form

DP keeps the general form (11.10). Two differences from J2:

- $\boldsymbol{M} \neq \boldsymbol{N}$ when $\beta \neq \eta$, so the rank-one
  term is
  $(\mathbb{C}_e : \boldsymbol{N}) \otimes (\boldsymbol{M} : \mathbb{C}_e)$ with
  two *different* tensors. **$\mathbb{C}_{ep}$ is not symmetric.** That is
  intrinsic to non-associative flow, not an implementation artefact, and a host
  assembling only the upper triangle will get the wrong stiffness.
- $\mathbb{C}_e : \boldsymbol{N}$ retains its volumetric part (6.6), and
  $\boldsymbol{M} : \mathbb{C}_e = 2G\operatorname{dev}(\boldsymbol{M}) + K\operatorname{tr}(\boldsymbol{M})\boldsymbol{I}$
  likewise with $\operatorname{tr}(\boldsymbol{M}) = \eta$.

### 11.8 Where the tangent is evaluated

At the **trial** state, not the converged one. The
$\partial\boldsymbol{\varepsilon}^p/\partial\boldsymbol{\varepsilon}|_{\Delta\lambda}$
term differentiates
$\boldsymbol{N}^{\mathrm{tr}}(\boldsymbol{\sigma}^{\mathrm{tr}})$, so
$\partial\boldsymbol{N}/\partial\boldsymbol{\sigma}$ must be taken at
$\sigma_{\mathrm{eq}}^{\mathrm{tr}}$. Substituting the converged
$\sigma_{\mathrm{eq}} = \sigma_{\mathrm{eq}}^{\mathrm{tr}} - 3G\Delta\lambda$
degrades the tangent by $2.6\times10^{-3}$ relative against a central difference
— a real loss of Newton rate, measured, not a wash.

$\boldsymbol{N}$ itself is a separate question: for radial return
$\boldsymbol{s} \parallel \boldsymbol{s}^{\mathrm{tr}}$ exactly, so
$\boldsymbol{N}^{\mathrm{tr}} \equiv \boldsymbol{N}$, and that substitution
genuinely does not matter. The code uses trial quantities for both.

---

## 12. The multi-stage variant

`j2_rk_plasticity` replaces the single backward-Euler step with a Runge–Kutta
tableau $(a_{ij}, b_i, c_i)$. Per stage $i$, of $s$ stages:

$$\boldsymbol{\varepsilon}^{p,(i)} = \boldsymbol{\varepsilon}^p_n
+ \sum_{j<i} a_{ij}\Delta\lambda_j\boldsymbol{N}_j
\;\left(+\; a_{ii}\Delta\lambda_i\boldsymbol{N}^{\mathrm{tr}}\ \text{if implicit}\right)$$

$$\kappa^{(i)} = \kappa_n + \sum_{j<i} a_{ij}\Delta\lambda_j
\;\left(+\; a_{ii}\Delta\lambda_i\ \text{if implicit}\right) \tag{12.1}$$

with $\Delta\lambda_i$ from the same scalar residual as (7.4) evaluated at the
stage state — directly for an explicit stage ($a_{ii} = 0$), by Newton for an
implicit one. The step closes with the $b$-weighted sum:

$$\boldsymbol{\varepsilon}^p_{n+1} = \boldsymbol{\varepsilon}^p_n + \sum_i b_i\Delta\lambda_i\boldsymbol{N}_i,
\qquad \kappa_{n+1} = \kappa_n + \sum_i b_i\Delta\lambda_i,
\qquad \Delta\lambda_{\mathrm{tot}} = \sum_i b_i\Delta\lambda_i \tag{12.2}$$

and the tangent is (11.10) evaluated at the **converged** state with
$\Delta\lambda = \Delta\lambda_{\mathrm{tot}}$.

Three things follow, all of them limitations worth stating plainly:

- **The stage sum runs over $j < i$ only.** A tableau with coupling *above* the
  diagonal — a fully implicit scheme such as Gauss–Legendre — is not integrated
  by this loop; its $a_{01}$ is silently dropped. Before it was guarded,
  `gauss_legendre_4` produced an equivalent plastic strain of $-8.81$ with a
  yield residual of $+10880$ and no error. The constructor now rejects any
  tableau that is neither DIRK nor explicit.
- **The implicit stage iterates on $\boldsymbol{N}^{\mathrm{tr}}$, not on its own
  stage normal.** It stores the converged stage normal afterwards. This is an
  approximation, and it is one reason the RK tangent is not exact.
- **The tangent is assembled from $\Delta\lambda_{\mathrm{tot}}$ rather than
  differentiated through the stages.** Measured against a central difference on
  fully plastic steps it sits at $\sim3\times10^{-4}$, against
  $\sim10^{-10}$ for the monolithic J2 tangent. That is a genuine approximation,
  not roundoff, and the test bounds reflect it.

For linear hardening the stage residual is linear, so every consistent DIRK or
explicit scheme lands on the same root as the monolithic map — which is the exact
identity §7.3 uses as a test.

---

## 13. Equation-to-code map

| equation | quantity | code |
|---|---|---|
| (2.1) | $\mathbb{C}_e = K\boldsymbol{I}\otimes\boldsymbol{I} + 2G\mathbb{I}^{\mathrm{dev}}$ | `plasticity_utils.h: make_isotropic_tangent` |
| (3.1) | $\sigma_{\mathrm{eq}} = \sqrt{3J_2}$, $F$ | `yield_functions.h: equivalent_stress`, `trial_yield` |
| (3.2) | $q = \sqrt{J_2}$, $F = q + \eta p - k - H$ | `drucker_prager_yield_function.h: trial_yield` |
| (4.2) | $\boldsymbol{N} = \tfrac{3}{2}\boldsymbol{s}/\sigma_{\mathrm{eq}}$ | `yield_functions.h: flow_normal` |
| (4.4) | $\boldsymbol{N}$, $\boldsymbol{M}$ with $\beta$, $\eta$ | `drucker_prager_yield_function.h: flow_normal`, `yield_normal` |
| (6.5) | $G_{\mathrm{eff}} = 3G$ | `yield_functions.h: effective_modulus` |
| (6.7) | $G_{\mathrm{eff}} = G + K\eta\beta$ | `drucker_prager_yield_function.h: effective_modulus` |
| (7.2) | trial state | `plasticity_utils.h: compute_trial`, `evaluate_at_state` |
| (7.4) | scalar residual, jacobian | `*_yield_function.h: residual`, `jacobian` |
| (8.3) | $\boldsymbol{\sigma} = \boldsymbol{\sigma}^{\mathrm{tr}} - 2G\Delta\lambda\boldsymbol{N}$ | `j2_plasticity.h: compute` |
| (9.3) | $\boldsymbol{\sigma} = \boldsymbol{\sigma}^{\mathrm{tr}} - \Delta\lambda\,\mathbb{C}_e:\boldsymbol{N}$ | `drucker_prager_plasticity.h: do_smooth_return` |
| (9.4) | $G\Delta\lambda \ge q^{\mathrm{tr}}$ | `drucker_prager_yield_function.h: needs_apex_return` |
| (10.2) | apex projection | `drucker_prager_yield_function.h: apex_plastic_strain` |
| (10.5) | $\mathbb{C}_{ep}^{\mathrm{apex}}$ | `drucker_prager_yield_function.h: apex_tangent` |
| (11.5) | $\partial\boldsymbol{N}/\partial\boldsymbol{\sigma}$ | `*_yield_function.h: flow_normal_stress_derivative` |
| (11.10) | general $\mathbb{C}_{ep}$ | `plasticity_utils.h: compute_tangent` |
| (11.12) | J2 closed-form $\mathbb{C}_{ep}$ | `j2_plasticity.h: compute` |
| (12.1) | RK stage assembly | `j2_rk_plasticity.h: compute` |

### 13.1 The identities, checked

Every algebraic identity this document rests on, evaluated on a generic
multiaxial stress with $K = 166.67$, $G = 76.92$, $\eta = 0.1$, $\beta = 0.05$,
against the actual yield-function and tangent code:

| identity | equation | result |
|---|---|---|
| $\mathbb{C}_e:\boldsymbol{a} = K\operatorname{tr}(\boldsymbol{a})\boldsymbol{I} + 2G\operatorname{dev}(\boldsymbol{a})$ | (2.2) | $4.1\times10^{-12}$ on a scale of $3.6\times10^{4}$ |
| $\operatorname{tr}(\boldsymbol{N}) = 0$ (J2) | (4.3) | $-2.8\times10^{-17}$ |
| $\boldsymbol{N}:\boldsymbol{N} = 3/2$ (J2) | (4.3) | $-2.2\times10^{-16}$ |
| $\mathbb{C}_e:\boldsymbol{N} = 2G\boldsymbol{N}$ (J2) | (4.3) | $1.5\times10^{-14}$ |
| $\boldsymbol{N}:\mathbb{C}_e:\boldsymbol{N} = 3G$ | (6.5) | $230.760000000$ vs $230.760000000$ |
| $\operatorname{tr}(\boldsymbol{N}) = \beta$ (DP) | (4.5) | $1.4\times10^{-17}$ |
| $\mathbb{C}_e:\boldsymbol{N} = G\boldsymbol{s}/q + K\beta\boldsymbol{I}$ (DP) | (6.6) | $1.7\times10^{-14}$ |
| $\boldsymbol{M}:\mathbb{C}_e:\boldsymbol{N} = G + K\eta\beta$ | (6.7) | $77.753350000$ vs $77.753350000$ |
| $\mathbb{C}_e:\mathbb{X}:\mathbb{C}_e = 4G^2\mathbb{X}$, J2 | (11.6) | rel $2.2\times10^{-16}$ |
| $\mathbb{C}_e:\mathbb{X}:\mathbb{C}_e = 4G^2\mathbb{X}$, DP | (11.6) | rel $6.6\times10^{-16}$ |
| same, $\mathbb{X}$ traceless but **skew** in the first pair | (11.6) | **rel $1.000$** |
| J2 closed form $=$ general `compute_tangent` | (11.12) | rel $1.2\times10^{-16}$ |
| $e_c/e_t$ uniaxial-strain asymmetry | (3.3) | $0.189598 / 0.277190 = 1.462$ |

The skew row is the point of the four preconditions in §11.3: $\mathbb{X}$
traceless in both index pairs is not sufficient, and the failure is total rather
than marginal.

The closed forms are also checked continuously against central differences
through the graph by `tangent_checker` — see `plasticity.md` §5 for the load
paths and what that check can and cannot see.

---

## 14. What the derivations assume

Collected, because each one is a place where a plausible extension silently
produces wrong numbers.

**Isotropic elasticity, in three independent places.**
$G_{\mathrm{eff}} = \boldsymbol{M}:\mathbb{C}_e:\boldsymbol{N}$ is a material
constant only for isotropic $\mathbb{C}_e$ — otherwise the *residual* is wrong,
not just the tangent. The stress shortcut (2.2) is the second. The $4G^2$
collapse (11.6) is the third. Measured with $\mathbb{C}_e$ made mildly
orthotropic ($1.6\times$ on one shear component): the closed-form tangent misses
a central difference by 2.3 %, the general `compute_tangent` by 1.9 %, and the
stress shortcut by $5.19$ absolute. Percent-level errors with no diagnostic.
This is why the materials build $\mathbb{C}_e$ themselves.

**The volumetric part of $\boldsymbol{N}$ must be constant in
$\boldsymbol{\sigma}$.** This is the operative form of the $4G^2$ precondition
for a *future* yield function, and it is narrower than "deviatoric". A
pressure-dependent dilatancy $\beta(p) = \beta_0 + c\,p$ adds
$\tfrac{c}{9}\boldsymbol{I}\otimes\boldsymbol{I}$ to
$\partial\boldsymbol{N}/\partial\boldsymbol{\sigma}$, traceless in neither index
pair; $c = 10^{-3}$ measures a **35 %** tangent error, and the error scales with
$K/G$ rather than with $c$ relative to $\beta$. Cap models, Matsuoka–Nakai and
Lade surfaces, and any smoothed cone tip share the property. Nothing checks it.

**Small strain.** No objective rate, no frame rotation. Nothing detects the
violation.

**$H' > -G_{\mathrm{eff}}$.** At or below that the yield residual has positive
slope and the return map has no admissible solution. Both materials throw rather
than clamp: clamping $\Delta\lambda$ to zero returns the elastic stress, and on a
uniaxial path with $H' = -300$ against $3G = 230.8$ the equivalent stress climbed
past $\sigma_0 = 50$ to $76.9$ with $\alpha$ identically zero — 54 % outside the
yield surface, silently. Moderate softening works. The *apex* criterion also
inverts under softening, giving a missed apex with $q < 0$; that is still
unguarded.

**$D = 3$ for the pressure terms.** $\operatorname{dev}$ divides the trace by
$D$, but $p = \operatorname{tr}(\boldsymbol{\sigma})/3$ and the
$\tfrac{\eta}{3}\boldsymbol{I}$, $\tfrac{\beta}{3}\boldsymbol{I}$ terms in (4.4)
use a hard $3$. At $D = 3$ they agree. At $D = 2$ they do not, so the DP normals
and the deviator would use different conventions. `material_policy_2d` exists but
is registered by nothing and instantiated by no test, so this is latent — as was
the $3K\,\mathbb{I}^{\mathrm{vol}}$ spelling of §2 until it was measured at 30 %
and fixed. If 2D is ever wanted, these are the lines to revisit.

**The apex tangent is a branch tangent.** Verified consistent *on* the branch to
$3.8\times10^{-8}$; a perturbation leaving the apex back onto the smooth cone is
not covered and cannot be by a central difference. At the true branch boundary
$C_{0101}$ jumps from $0$ to $71.4$ — inherent to a non-smooth return map.
