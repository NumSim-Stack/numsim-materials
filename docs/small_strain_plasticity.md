# Small-Strain Plasticity — Equations and Implementation

This document describes the return-mapping algorithm and consistent algorithmic
tangent implemented in `small_strain_plasticity`. The framework uses a
**yield function policy** so that a single material class can handle both
J2/von Mises and Drucker-Prager plasticity.

The notation below avoids overloading the symbol `alpha`: the scalar hardening
variable is denoted by $\kappa$, while the Drucker-Prager pressure/friction
coefficient is denoted by $\eta$.

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $\boldsymbol{\varepsilon}$ | Total strain tensor, input |
| $\boldsymbol{\varepsilon}^p$ | Plastic strain tensor, history |
| $\kappa$ | Scalar hardening variable / accumulated plasticity measure, history |
| $\boldsymbol{\sigma}$ | Cauchy stress tensor |
| $\mathbf{s}$ | Deviatoric stress: $\mathbf{s} = \boldsymbol{\sigma} - \tfrac{1}{3}\mathrm{tr}(\boldsymbol{\sigma})\,\mathbf{I}$ |
| $J_2$ | Second deviatoric invariant: $J_2 = \tfrac{1}{2}\,\mathbf{s}:\mathbf{s}$ |
| $q$ | Drucker-Prager deviatoric stress measure: $q = \sqrt{J_2}$ |
| $I_1$ | First stress invariant: $I_1 = \mathrm{tr}(\boldsymbol{\sigma})$ |
| $p$ | Mean stress variable used by the DP model: $p = I_1/3$ |
| $\sigma_\mathrm{eq}$ | J2/von Mises equivalent stress: $\sigma_\mathrm{eq} = \sqrt{3J_2}$ |
| $\tilde{\sigma}$ | Modified equivalent stress used in the scalar residual |
| $\mathbb{C}$ | Fourth-order elastic stiffness tensor |
| $G$ | Shear modulus |
| $K$ | Bulk modulus |
| $Y_0$ | Initial yield resistance: $\sigma_0$ for J2, $k$ for DP |
| $\sigma_0$ | Initial von Mises yield stress |
| $k$ | Drucker-Prager cohesion / initial yield resistance in the $q + \eta p$ normalization |
| $H(\kappa)$ | Isotropic hardening stress |
| $H' = dH/d\kappa$ | Isotropic hardening modulus |
| $\Delta\lambda$ | Plastic multiplier / hardening-variable increment for the smooth return |
| $\Delta\kappa$ | Scalar hardening-variable increment, especially in the apex return |
| $\eta$ | Drucker-Prager pressure/friction coefficient in the yield function |
| $\beta$ | Drucker-Prager dilatancy coefficient in the plastic potential |
| $\mathbf{M}$ | Yield normal: $\mathbf{M} = \partial F / \partial\boldsymbol{\sigma}$ |
| $\mathbf{N}$ | Flow normal: $\mathbf{N} = \partial G_p / \partial\boldsymbol{\sigma}$ |
| $\mathbb{I}^\mathrm{sym}$ | Fourth-order symmetric identity |
| $\mathbb{I}^\mathrm{dev}$ | Deviatoric projector: $\mathbb{I}^\mathrm{dev} = \mathbb{I}^\mathrm{sym} - \tfrac{1}{3}\mathbf{I}\otimes\mathbf{I}$ |

For **associative** plasticity, $\mathbf{M} = \mathbf{N}$. For
**non-associative** plasticity, they generally differ.

### Stress sign convention

The Drucker-Prager equations below use

$$p = \frac{1}{3} I_1 = \frac{1}{3}\mathrm{tr}(\boldsymbol{\sigma})$$

and the pressure term enters the yield function as $+\eta p$. Thus positive
$p$ is the sign that increases the Drucker-Prager yield function. If a
tension-positive Cauchy stress convention is used but compression should
increase pressure-dependent yielding, either define $p = -I_1/3$ or use the
corresponding opposite sign for the pressure coefficient.

---

## 2. Yield Functions

### 2.1 J2 / von Mises

**Equivalent stress**

$$\sigma_\mathrm{eq}
  = \sqrt{\frac{3}{2}\,\mathbf{s}:\mathbf{s}}
  = \sqrt{3J_2}$$

**Yield function**

$$F = \sigma_\mathrm{eq} - \sigma_0 - H(\kappa)$$

**Modified equivalent stress**

For J2, the modified equivalent stress is simply

$$\tilde{\sigma} = \sigma_\mathrm{eq}$$

**Flow / yield normal**

J2 is associative, so $\mathbf{M}=\mathbf{N}$:

$$\mathbf{N}
  = \mathbf{M}
  = \frac{\partial F}{\partial\boldsymbol{\sigma}}
  = \frac{3}{2}\,\frac{\mathbf{s}}{\sigma_\mathrm{eq}}$$

**Effective modulus**

$$G_\mathrm{eff} = 3G$$

This follows from

$$\mathbf{M}:\mathbb{C}:\mathbf{N}
  = \mathbf{N}:\mathbb{C}:\mathbf{N}
  = 2G\,\mathbf{N}:\mathbf{N}
  = 3G$$

because $\mathbf{N}:\mathbf{N}=3/2$.

Equivalently, the radial deviatoric return reduces $\sigma_\mathrm{eq}$ by
$3G\,\Delta\lambda$.

**Flow normal derivative**

Away from $\sigma_\mathrm{eq}=0$,

$$\frac{\partial \mathbf{N}}{\partial \boldsymbol{\sigma}}
  = \frac{1}{\sigma_\mathrm{eq}}
    \left(
      \frac{3}{2}\,\mathbb{I}^\mathrm{dev}
      - \mathbf{N}\otimes\mathbf{N}
    \right)$$

---

### 2.2 Drucker-Prager

**Parameters**

| Symbol | Meaning |
|--------|---------|
| $\eta$ | Pressure/friction coefficient in the yield function |
| $\beta$ | Dilatancy coefficient in the plastic potential |
| $k$ | Cohesion / initial yield resistance |
| $K$ | Bulk modulus |

When $\eta \neq \beta$, the flow rule is non-associative. When
$\eta = \beta$, the flow rule is associative.

When $\eta = \beta = 0$, the Drucker-Prager surface becomes a circular
cylinder in deviatoric stress space. It is equivalent to von Mises only up to
normalization. Specifically,

$$F_\mathrm{DP} = \sqrt{J_2} - k - H_\mathrm{DP}$$

matches

$$F_\mathrm{J2} = \sqrt{3J_2} - \sigma_0 - H_\mathrm{J2}$$

when the yield resistance and hardening are scaled consistently, for example

$$\sigma_0 + H_\mathrm{J2} = \sqrt{3}\,\bigl(k + H_\mathrm{DP}\bigr)$$

The corresponding flow normals also differ by a constant factor, so the
plastic multiplier normalization must be treated consistently.

**Deviatoric stress measure**

$$q = \sqrt{J_2} = \sqrt{\frac{1}{2}\,\mathbf{s}:\mathbf{s}}$$

**Modified equivalent stress**

$$\tilde{\sigma}
  = q + \eta p
  = \sqrt{J_2} + \frac{\eta}{3}I_1$$

The factor $1/3$ appears because

$$\frac{\partial p}{\partial\boldsymbol{\sigma}}
  = \frac{1}{3}\mathbf{I}$$

**Yield function**

$$F = q + \eta p - k - H(\kappa)
    = \tilde{\sigma} - k - H(\kappa)$$

**Yield normal**

$$\mathbf{M}
  = \frac{\partial F}{\partial\boldsymbol{\sigma}}
  = \frac{\mathbf{s}}{2\sqrt{J_2}} + \frac{\eta}{3}\,\mathbf{I}
  = \frac{\mathbf{s}}{2q} + \frac{\eta}{3}\,\mathbf{I}$$

**Flow normal**

The plastic potential has pressure coefficient $\beta$, giving

$$\mathbf{N}
  = \frac{\partial G_p}{\partial\boldsymbol{\sigma}}
  = \frac{\mathbf{s}}{2\sqrt{J_2}} + \frac{\beta}{3}\,\mathbf{I}
  = \frac{\mathbf{s}}{2q} + \frac{\beta}{3}\,\mathbf{I}$$

Since $\mathrm{tr}(\mathbf{s}) = 0$,

$$\mathrm{tr}(\mathbf{N}) = \beta$$

**Effective modulus**

For isotropic elasticity,

$$G_\mathrm{eff}
  = \mathbf{M}:\mathbb{C}:\mathbf{N}
  = G + K\eta\beta$$

Derivation:

1. The deviatoric part of the return reduces $q = \sqrt{J_2}$ by
   $G\,\Delta\lambda$.
2. The volumetric part of the return changes the mean stress by
   $$\Delta p = -K\,\mathrm{tr}(\mathbf{N})\,\Delta\lambda
     = -K\beta\,\Delta\lambda$$
3. The pressure term $\eta p$ in the yield function decreases by
   $K\eta\beta\,\Delta\lambda$.

Therefore the total decrease in $q + \eta p$ is

$$(G + K\eta\beta)\Delta\lambda$$

**Flow normal derivative**

The volumetric term $(\beta/3)\mathbf{I}$ is constant with respect to stress,
so only the deviatoric part contributes. Away from $J_2=0$,

$$\frac{\partial\mathbf{N}}{\partial\boldsymbol{\sigma}}
  = \frac{\partial}{\partial\boldsymbol{\sigma}}
    \left(\frac{\mathbf{s}}{2\sqrt{J_2}}\right)
  = \frac{1}{2\sqrt{J_2}}
    \left(
      \mathbb{I}^\mathrm{dev}
      - \frac{\mathbf{s}\otimes\mathbf{s}}{2J_2}
    \right)$$

Equivalently, with $q=\sqrt{J_2}$,

$$\frac{\partial\mathbf{N}}{\partial\boldsymbol{\sigma}}
  = \frac{1}{2q}
    \left(
      \mathbb{I}^\mathrm{dev}
      - \frac{\mathbf{s}\otimes\mathbf{s}}{2q^2}
    \right)$$

Implementation note: this derivative must be computed from $\mathbf{s}$,
$q$, or the deviatoric part of $\mathbf{N}$. It must not treat the full
non-associative $\mathbf{N}$ as if it were purely deviatoric, because the
term $(\beta/3)\mathbf{I}$ has zero stress derivative.

---

## 3. Smooth Return-Mapping Algorithm

Given:

- $\boldsymbol{\varepsilon}$,
- old plastic strain $\boldsymbol{\varepsilon}^p_n$,
- old scalar hardening variable $\kappa_n$,
- elastic stiffness $\mathbb{C}$,
- material parameters.

### Step 1: Trial state

$$\boldsymbol{\sigma}^\mathrm{trial}
  = \mathbb{C}:
    \left(\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p_n\right)$$

Compute the trial quantities required by the yield-function policy:

$$\mathbf{s}^\mathrm{trial},
  \qquad
  J_2^\mathrm{trial},
  \qquad
  \tilde{\sigma}^\mathrm{trial},
  \qquad
  \mathbf{M}^\mathrm{trial},
  \qquad
  \mathbf{N}^\mathrm{trial},
  \qquad
  F^\mathrm{trial}$$

For J2,

$$\tilde{\sigma}^\mathrm{trial}
  = \sigma_\mathrm{eq}^\mathrm{trial}$$

For Drucker-Prager,

$$\tilde{\sigma}^\mathrm{trial}
  = q^\mathrm{trial} + \eta p^\mathrm{trial}$$

### Step 2: Yield check

If $F^\mathrm{trial} \leq 0$ then the step is elastic:

$$\boldsymbol{\sigma} = \boldsymbol{\sigma}^\mathrm{trial}, \quad
  \mathbb{C}_\mathrm{ep} = \mathbb{C}, \quad
  \boldsymbol{\varepsilon}^p_{n+1} = \boldsymbol{\varepsilon}^p_n, \quad
  \kappa_{n+1} = \kappa_n$$

### Step 3: Scalar Newton iteration

For a plastic smooth-return step, find $\Delta\lambda$ such that

$$r(\Delta\lambda)
  = \tilde{\sigma}^\mathrm{trial}
  - G_\mathrm{eff}\,\Delta\lambda
  - Y_0
  - H(\kappa_n + \Delta\lambda)
  = 0$$

where

$$Y_0 =
  \begin{cases}
    \sigma_0, & \text{J2} \\
    k, & \text{Drucker-Prager}
  \end{cases}$$

The Newton update is

$$\Delta\lambda
  \leftarrow
  \Delta\lambda
  - \frac{r}{dr/d\Delta\lambda}$$

with

$$\frac{dr}{d\Delta\lambda}
  = -G_\mathrm{eff} - H'$$

The trial quantities $\tilde{\sigma}^\mathrm{trial}$,
$\mathbf{M}^\mathrm{trial}$, and $\mathbf{N}^\mathrm{trial}$ are held fixed
during this scalar Newton iteration. Only $H$ and $H'$ are re-evaluated at
$\kappa = \kappa_n + \Delta\lambda$.

### Step 4: State update

The smooth-return plastic strain update is

$$\boldsymbol{\varepsilon}^p_{n+1}
  = \boldsymbol{\varepsilon}^p_n
  + \Delta\lambda\,\mathbf{N}^\mathrm{trial}$$

The scalar hardening variable is updated as

$$\kappa_{n+1} = \kappa_n + \Delta\lambda$$

The stress is recomputed from the elastic law:

$$\boldsymbol{\sigma}
  = \mathbb{C}:
    \left(
      \boldsymbol{\varepsilon}
      - \boldsymbol{\varepsilon}^p_{n+1}
    \right)$$

For J2, the trial and converged deviatoric directions coincide. For
Drucker-Prager, this smooth-return algorithm intentionally uses the frozen
trial flow direction.

---

## 4. Consistent Algorithmic Tangent for the Smooth Return

The tangent

$$\mathbb{C}_\mathrm{ep}
  = \frac{d\boldsymbol{\sigma}}{d\boldsymbol{\varepsilon}}$$

is derived by applying the implicit function theorem to the converged scalar
residual

$$r\left(\Delta\lambda(\boldsymbol{\varepsilon}),
    \boldsymbol{\varepsilon}\right) = 0$$

The tangent below is consistent with the implemented algorithm, where the
plastic strain update uses $\mathbf{N}^\mathrm{trial}(\boldsymbol{\varepsilon})$.

### 4.1 Stress as a function of strain and plastic multiplier

Suppressing constants from the previous converged step,

$$\boldsymbol{\sigma}
  = \mathbb{C}:
    \left(
      \boldsymbol{\varepsilon}
      - \boldsymbol{\varepsilon}^p_n
      - \Delta\lambda\,\mathbf{N}^\mathrm{trial}(\boldsymbol{\varepsilon})
    \right)$$

The trial flow normal depends on strain through

$$\boldsymbol{\sigma}^\mathrm{trial}
  = \mathbb{C}:
    \left(
      \boldsymbol{\varepsilon}
      - \boldsymbol{\varepsilon}^p_n
    \right)$$

### 4.2 Partial derivatives

**Stress with respect to strain at fixed $\Delta\lambda$**

$$\mathbf{A}
  = \left.
      \frac{\partial\boldsymbol{\sigma}}
           {\partial\boldsymbol{\varepsilon}}
    \right|_{\Delta\lambda}
  = \mathbb{C}
    - \Delta\lambda\;
      \mathbb{C}:
      \frac{\partial\mathbf{N}}{\partial\boldsymbol{\sigma}}:
      \mathbb{C}$$

This term accounts for rotation of the trial flow direction under
perturbations of the strain.

**Stress with respect to $\Delta\lambda$ at fixed strain**

$$\frac{\partial\boldsymbol{\sigma}}{\partial\Delta\lambda}
  = -\mathbb{C}:\mathbf{N}^\mathrm{trial}$$

**Residual with respect to strain**

The residual uses the trial modified equivalent stress:

$$r
  = \tilde{\sigma}^\mathrm{trial}
  - G_\mathrm{eff}\Delta\lambda
  - Y_0
  - H(\kappa_n + \Delta\lambda)$$

Therefore,

$$\frac{\partial r}{\partial\boldsymbol{\varepsilon}}
  = \frac{\partial\tilde{\sigma}^\mathrm{trial}}
         {\partial\boldsymbol{\varepsilon}}
  = \mathbf{M}^\mathrm{trial}:\mathbb{C}$$

This is a second-order tensor. The key point is that this derivative is
$\mathbf{M}:\mathbb{C}$, not $\mathbf{M}:\mathbf{A}$, because
$\tilde{\sigma}^\mathrm{trial}$ depends on $\boldsymbol{\varepsilon}$ only
through the trial stress.

**Residual with respect to $\Delta\lambda$**

$$\frac{\partial r}{\partial\Delta\lambda}
  = -G_\mathrm{eff} - H'$$

Using the consistency requirement $G_\mathrm{eff} = \mathbf{M}:\mathbb{C}:\mathbf{N}$
this may also be written as

$$\frac{\partial r}{\partial\Delta\lambda}
  = -\left(\mathbf{M}:\mathbb{C}:\mathbf{N} + H'\right)$$

### 4.3 Implicit function theorem

From $r\left(\Delta\lambda(\boldsymbol{\varepsilon}), \boldsymbol{\varepsilon}\right)=0$
one obtains

$$\frac{d\Delta\lambda}{d\boldsymbol{\varepsilon}}
  = -\left(
      \frac{\partial r}{\partial\Delta\lambda}
    \right)^{-1}
    \frac{\partial r}{\partial\boldsymbol{\varepsilon}}$$

and hence

$$\frac{d\Delta\lambda}{d\boldsymbol{\varepsilon}}
  = \frac{\mathbf{M}:\mathbb{C}}{G_\mathrm{eff} + H'}$$

### 4.4 Algorithmic tangent

Combining the partial derivatives gives

$$\boxed{
  \mathbb{C}_\mathrm{ep}
  = \mathbf{A}
    - \frac{
        (\mathbb{C}:\mathbf{N})
        \otimes
        (\mathbf{M}:\mathbb{C})
      }
      {G_\mathrm{eff} + H'}
}$$

where

$$\mathbf{A}
  = \mathbb{C}
    - \Delta\lambda\;
      \mathbb{C}:
      \frac{\partial\mathbf{N}}{\partial\boldsymbol{\sigma}}:
      \mathbb{C}$$

All quantities in this smooth-return tangent are evaluated at the **trial
state** unless explicitly stated otherwise.

For non-associative Drucker-Prager plasticity, $\mathbf{M}\neq\mathbf{N}$,
so the algorithmic tangent is generally non-symmetric.

### 4.5 Specialization to J2

For J2,

$$\mathbf{M} = \mathbf{N}
  = \frac{3}{2}\frac{\mathbf{s}}{\sigma_\mathrm{eq}}$$

and $G_\mathrm{eff}=3G$. The flow direction is purely deviatoric, and the
trial and converged deviatoric directions coincide because the J2 return is
radial.

### 4.6 Why trial state, not converged state

The implementation updates plastic strain using the frozen trial flow
direction:

$$\boldsymbol{\varepsilon}^p_{n+1}
  = \boldsymbol{\varepsilon}^p_n
  + \Delta\lambda\,\mathbf{N}^\mathrm{trial}$$

Therefore the tangent must differentiate this exact algorithm. For J2 the
distinction is immaterial because radial return preserves the deviatoric
direction. For Drucker-Prager, however, the deviatoric direction can rotate
during correction, so using converged-state quantities in the smooth tangent
would produce an inconsistent algorithmic tangent.

---

## 5. Implementation Map

| Equation / operation | File | Function |
|----------------------|------|----------|
| Trial state evaluation | `plasticity_utils.h` | `evaluate_at_state()` |
| Yield check + trial state | `plasticity_utils.h` | `compute_trial()` |
| Smooth algorithmic tangent | `plasticity_utils.h` | `compute_tangent()` |
| Newton iteration + state update | `small_strain_plasticity.h` | `compute()` |
| J2 yield function policy | `yield_functions.h` | `j2_yield_function` |
| Drucker-Prager yield function policy | `drucker_prager_yield_function.h` | `drucker_prager_yield_function` |
| DP apex return | `drucker_prager_yield_function.h` | `needs_apex_return()`, `apex_*()` |
| Scalar Newton solver | `backward_euler.h` | `backward_euler::solve()` |

### Yield function policy interface

Each yield function policy should provide the following operations:

```text
equivalent_stress(s)
    -> scalar equivalent_stress_measure
       J2: sigma_eq = sqrt(3 J2)
       DP: q = sqrt(J2)

modified_equivalent_stress(sigma, equivalent_stress)
    -> scalar
       J2: sigma_eq
       DP: q + eta p

trial_yield(sigma, equivalent_stress, Y0, H)
    -> scalar F

residual(sigma_tilde_trial, delta_lambda, G_eff, Y0, H)
    -> scalar r

jacobian(G_eff, H_prime)
    -> scalar dr/d(delta_lambda)

effective_modulus(G)
    -> scalar G_eff
       J2: 3G
       DP: G + K eta beta   (K, eta, beta are stored as policy members)

flow_normal(s, equivalent_stress)
    -> tensor2 N = dG_p/dsigma

yield_normal(s, equivalent_stress)
    -> tensor2 M = dF/dsigma

flow_normal_stress_derivative(s, equivalent_stress)
    -> tensor4 dN/dsigma
       Takes the deviatoric stress s (not the full N) to avoid
       cancellation when reconstructing s from a non-associative N.
```

The DP policy stores its own `K_bulk`, `eta`, and `beta` so that
`effective_modulus(G)` only needs the shear modulus from the caller. If a
yield function policy depended on `K` from outside, the signature should
become `effective_modulus(G, K)` instead.

For Drucker-Prager, `flow_normal_stress_derivative` must use `s` (or `q`),
not the full non-associative `N`. The volumetric term `(β/3)·I` in `N` has
zero stress derivative, but reconstructing `s` from `N` introduces
cancellation error when `q` is small.

Policies that support an apex return may additionally provide:

```text
needs_apex_return(G, delta_lambda, equivalent_stress)
apex_modified_sig_eq(sigma)
apex_effective_modulus()
apex_plastic_strain(eps, eps_p_old, delta_kappa)
apex_tangent(H_prime)
```

The main material class can dispatch to these methods using
`if constexpr (requires { ... })`, so the apex branch is compiled only for
yield functions that provide apex support. J2 has no apex and never triggers
this path.

---

## 6. Consistency Requirements

The following relationships must hold for the residual, stress update, and
tangent to be mutually consistent.

### 6.1 Yield normal consistency

The yield normal must be the exact stress gradient of the yield function:

$$\mathbf{M}
  = \frac{\partial F}{\partial\boldsymbol{\sigma}}
  = \frac{\partial\tilde{\sigma}}{\partial\boldsymbol{\sigma}}$$

For Drucker-Prager, if the modified equivalent stress is

$$\tilde{\sigma} = q + \eta p$$

with $p = \tfrac{1}{3}I_1$, then

$$\frac{\partial p}{\partial\boldsymbol{\sigma}}
  = \frac{1}{3}\mathbf{I}$$

and therefore

$$\mathbf{M}
  = \frac{\mathbf{s}}{2q} + \frac{\eta}{3}\mathbf{I}$$

A mismatch between the pressure term in `modified_equivalent_stress()` and
the pressure term in `yield_normal()` will directly corrupt the tangent.

For example, the following pair is inconsistent:

$$\tilde{\sigma} = q + \eta I_1$$

but

$$\mathbf{M} = \frac{\mathbf{s}}{2q} + \frac{\eta}{3}\mathbf{I}$$

because the gradient of $\eta I_1$ is $\eta\mathbf{I}$, not
$(\eta/3)\mathbf{I}$.

### 6.2 Effective modulus consistency

The effective modulus used in the scalar residual must satisfy

$$G_\mathrm{eff}
  = \mathbf{M}:\mathbb{C}:\mathbf{N}$$

For isotropic elasticity this gives

$$G_\mathrm{eff} = 3G \qquad\text{for J2}$$

and

$$G_\mathrm{eff} = G + K\eta\beta \qquad\text{for Drucker-Prager}$$

The residual Jacobian is then

$$\frac{dr}{d\Delta\lambda}
  = -\left(G_\mathrm{eff} + H'\right)
  = -\left(\mathbf{M}:\mathbb{C}:\mathbf{N} + H'\right)$$

### 6.3 Trial-state tangent consistency

The smooth-return tangent must use the same trial-state quantities used by
the update:

$$\mathbf{N}^\mathrm{trial},
  \qquad
  \mathbf{M}^\mathrm{trial},
  \qquad
  \frac{\partial\mathbf{N}^\mathrm{trial}}
       {\partial\boldsymbol{\sigma}^\mathrm{trial}}$$

This is required because the algorithm updates plastic strain with
$\mathbf{N}^\mathrm{trial}$, not the converged $\mathbf{N}$.

### 6.4 Smoothness requirements

The smooth-return tangent assumes the active return is differentiable. The
derivative formulas for J2 and Drucker-Prager require

$$\sigma_\mathrm{eq} > 0
  \qquad\text{or}\qquad
  q = \sqrt{J_2} > 0$$

respectively. At the Drucker-Prager apex, the smooth-return tangent is not
valid and the apex tangent in Section 7 must be used.

---

## 7. Drucker-Prager Apex Return

The Drucker-Prager yield surface is a cone in the $(q,p)$ plane. With

$$F = q + \eta p - k - H(\kappa)$$

its apex occurs at $q = 0$ and

$$p = \frac{k + H(\kappa)}{\eta}$$

provided $\eta \neq 0$ and the sign convention is consistent with the
definition of $p$.

The apex branch assumes that the denominator appearing in the scalar solve is
nonzero:

$$K\eta\beta + H' \neq 0$$

For the usual pressure-sensitive case, one typically has $\eta>0$. The sign
and magnitude of $\beta$ determine the volumetric plastic flow direction.

### 7.1 Apex detection

After the standard smooth-return Newton solve gives $\Delta\lambda$, check
whether the deviatoric correction would overshoot the cone apex:

$$G\,\Delta\lambda \geq q^\mathrm{trial}$$

where $q^\mathrm{trial}=\sqrt{J_2^\mathrm{trial}}$.

If this condition holds, the corrected value of $q$ from the smooth return
would be non-positive, so the standard cone return is invalid and the apex
return must be used.

This situation typically occurs under loading paths where the pressure
contribution dominates the deviatoric stress.

### 7.2 Apex return algorithm

At the apex, the deviatoric stress vanishes:

$$\mathbf{s} = \mathbf{0}$$

and the stress is purely hydrostatic:

$$\boldsymbol{\sigma} = p\,\mathbf{I}$$

The yield condition reduces to

$$\eta p = k + H(\kappa_n + \Delta\kappa)$$

The pressure correction uses only the volumetric part of the flow rule.
Since $\mathrm{tr}(\mathbf{N}) = \beta$, the pressure after the volumetric
correction is

$$p = p^\mathrm{trial} - K\beta\,\Delta\kappa$$

Substituting into the apex yield condition gives the scalar apex residual:

$$r_\mathrm{apex}(\Delta\kappa)
  = \eta p^\mathrm{trial}
  - K\eta\beta\,\Delta\kappa
  - k
  - H(\kappa_n + \Delta\kappa)
  = 0$$

with Jacobian

$$\frac{dr_\mathrm{apex}}{d\Delta\kappa}
  = -K\eta\beta - H'$$

This is the same residual structure as the smooth return, but with

$$\tilde{\sigma}^\mathrm{apex} = \eta p^\mathrm{trial}$$

and

$$G_\mathrm{eff}^\mathrm{apex} = K\eta\beta$$

The $G$ term is dropped because the final apex stress has no deviatoric part.

### 7.3 Apex plastic strain update

At the apex, all elastic deviatoric strain is removed so that the final
deviatoric stress vanishes. This is enforced by setting

$$\mathrm{dev}\left(\boldsymbol{\varepsilon}^p_{n+1}\right)
  = \mathrm{dev}\left(\boldsymbol{\varepsilon}\right)$$

The volumetric plastic strain is updated using the volumetric part of the
flow rule:

$$\mathrm{tr}\left(\boldsymbol{\varepsilon}^p_{n+1}\right)
  = \mathrm{tr}\left(\boldsymbol{\varepsilon}^p_n\right)
  + \beta\,\Delta\kappa$$

Therefore,

$$\boldsymbol{\varepsilon}^p_{n+1}
  = \mathrm{dev}(\boldsymbol{\varepsilon})
  + \frac{
      \mathrm{tr}(\boldsymbol{\varepsilon}^p_n)
      + \beta\,\Delta\kappa
    }{3}\,\mathbf{I}$$

The resulting stress is hydrostatic:

$$\boldsymbol{\sigma}
  = \mathbb{C}:
    \left(
      \boldsymbol{\varepsilon}
      - \boldsymbol{\varepsilon}^p_{n+1}
    \right)
  = p\,\mathbf{I}$$

where equivalently

$$p = p^\mathrm{trial} - K\beta\,\Delta\kappa$$

or, after convergence,

$$p = \frac{k + H(\kappa_n + \Delta\kappa)}{\eta}$$

provided $\eta \neq 0$.

In the apex branch, $\Delta\kappa$ denotes the increment of the scalar
hardening variable. It is not necessarily the norm of the full plastic
strain increment. The deviatoric plastic strain is set directly to enforce
$\mathbf{s}=\mathbf{0}$.

This apex treatment should be interpreted as the implemented **projection
algorithm**: the deviatoric plastic strain is set to enforce $q=0$, while
$\Delta\kappa$ is determined by the apex consistency condition
$\eta\,p = k + H(\kappa_n + \Delta\kappa)$. It differs from a classical
single-plastic-multiplier flow-rule update where the deviatoric plastic
correction and hardening increment would both be tied strictly to the same
plastic multiplier $\Delta\lambda$.

### 7.4 Apex tangent

At the apex, the return map is nonsmooth. The tangent below is the
**algorithmic tangent for perturbations that remain on the active apex
branch** — that is, perturbations small enough that the next return still
falls on the apex. For perturbations that leave the apex and return to the
smooth cone, the smooth-branch tangent in Section 4 applies instead. This
branch tangent is therefore not a unique classical derivative of the full
return map; it is the consistent tangent of the projection algorithm
restricted to the apex branch.

For such on-branch perturbations, the deviatoric stress remains zero and
only the hydrostatic pressure changes.

The apex residual is

$$r_\mathrm{apex}
  = \eta p^\mathrm{trial}
  - K\eta\beta\,\Delta\kappa
  - k
  - H(\kappa_n + \Delta\kappa)$$

Its strain derivative is

$$\frac{\partial r_\mathrm{apex}}
      {\partial\boldsymbol{\varepsilon}}
  = \eta\,
    \frac{\partial p^\mathrm{trial}}
         {\partial\boldsymbol{\varepsilon}}
  = \eta K\,\mathbf{I}$$

and its scalar derivative is

$$\frac{\partial r_\mathrm{apex}}{\partial\Delta\kappa}
  = -K\eta\beta - H'$$

Therefore,

$$\frac{d\Delta\kappa}{d\boldsymbol{\varepsilon}}
  = \frac{\eta K}{K\eta\beta + H'}\,\mathbf{I}$$

Since $p = p^\mathrm{trial} - K\beta\,\Delta\kappa$, one obtains

$$\frac{dp}{d\boldsymbol{\varepsilon}}
  = K\mathbf{I}
  - K\beta\,
    \frac{d\Delta\kappa}{d\boldsymbol{\varepsilon}}$$

and hence

$$\frac{dp}{d\boldsymbol{\varepsilon}}
  = K\mathbf{I}
    \left(
      1 - \frac{K\eta\beta}{K\eta\beta + H'}
    \right)
  = \frac{KH'}{K\eta\beta + H'}\,\mathbf{I}$$

Because $\boldsymbol{\sigma}=p\mathbf{I}$, the apex tangent is

$$\boxed{
  \mathbb{C}_\mathrm{ep}^\mathrm{apex}
  = \frac{K H'}{K\eta\beta + H'}\;
    \mathbf{I}\otimes\mathbf{I}
}$$

This tangent is rank one and purely volumetric. All deviatoric stiffness
vanishes on the apex branch.

For perfect plasticity, $H'=0$, and if $K\eta\beta\neq0$, the apex tangent
becomes

$$\mathbb{C}_\mathrm{ep}^\mathrm{apex}=\mathbf{0}$$

This zero tangent is mathematically consistent with the idealized apex
return, but it can make the global Newton solve more difficult.

### 7.5 Physical interpretation

The apex return occurs when the pressure contribution to the Drucker-Prager
yield condition dominates the deviatoric stress. In such a case, the smooth
cone return would attempt to reduce $q$ below zero, which is impossible
because

$$q = \sqrt{J_2} \geq 0$$

Under confined loading, for example, the mean stress can grow rapidly
relative to the deviatoric stress. Once the smooth correction would remove
all deviatoric stress, the stress state must return to the apex instead of to
a smooth point on the cone.

The two kinks often visible in the stress-strain response correspond to:

1. **Elastic to smooth cone:** standard return mapping activates.
2. **Smooth cone to apex:** deviatoric stress vanishes and the response
   becomes purely hydrostatic.

### 7.6 Apex implementation map

| Equation / operation | File | Function |
|----------------------|------|----------|
| Apex detection | `drucker_prager_yield_function.h` | `needs_apex_return()` |
| Apex modified equivalent stress | `drucker_prager_yield_function.h` | `apex_modified_sig_eq()` |
| Apex effective modulus | `drucker_prager_yield_function.h` | `apex_effective_modulus()` |
| Apex plastic strain update | `drucker_prager_yield_function.h` | `apex_plastic_strain()` |
| Apex tangent | `drucker_prager_yield_function.h` | `apex_tangent()` |
| Dispatch between smooth and apex return | `small_strain_plasticity.h` | `compute()` |

---

## 8. Summary of Key Checks

For a correct implementation, verify the following identities numerically and
analytically.

### J2

$$\mathbf{M}=\mathbf{N}
  = \frac{3}{2}\frac{\mathbf{s}}{\sigma_\mathrm{eq}}$$

$$\mathbf{M}:\mathbb{C}:\mathbf{N}=3G$$

$$\frac{dr}{d\Delta\lambda}=-(3G+H')$$

### Drucker-Prager smooth return

$$\tilde{\sigma}=q+\eta p$$

$$\mathbf{M}=\frac{\mathbf{s}}{2q}+\frac{\eta}{3}\mathbf{I}$$

$$\mathbf{N}=\frac{\mathbf{s}}{2q}+\frac{\beta}{3}\mathbf{I}$$

$$\mathbf{M}:\mathbb{C}:\mathbf{N}=G+K\eta\beta$$

$$\frac{dr}{d\Delta\lambda}=-(G+K\eta\beta+H')$$

### Drucker-Prager apex return

Apex detection:

$$G\Delta\lambda \geq q^\mathrm{trial}$$

Apex residual:

$$r_\mathrm{apex}
  = \eta p^\mathrm{trial}
  - K\eta\beta\Delta\kappa
  - k
  - H(\kappa_n+\Delta\kappa)$$

Apex tangent:

$$\mathbb{C}_\mathrm{ep}^\mathrm{apex}
  = \frac{KH'}{K\eta\beta+H'}\,\mathbf{I}\otimes\mathbf{I}$$

---

## 9. Common Failure Modes

### 9.1 Pressure factor mismatch

If the modified equivalent stress is implemented as

$$\tilde{\sigma}=q+\eta I_1$$

but the yield normal is implemented as

$$\mathbf{M}=\frac{\mathbf{s}}{2q}+\frac{\eta}{3}\mathbf{I}$$

then the tangent will be wrong by a factor of three in the volumetric
coupling. The two consistent choices are either

$$\tilde{\sigma}=q+\eta p = q+\frac{\eta}{3}I_1$$

with

$$\mathbf{M}=\frac{\mathbf{s}}{2q}+\frac{\eta}{3}\mathbf{I}$$

or

$$\tilde{\sigma}=q+\eta I_1$$

with

$$\mathbf{M}=\frac{\mathbf{s}}{2q}+\eta\mathbf{I}$$

The first convention is the one used in this document.

### 9.2 Using the full DP flow normal in the derivative

For Drucker-Prager,

$$\mathbf{N}=\frac{\mathbf{s}}{2q}+\frac{\beta}{3}\mathbf{I}$$

but

$$\frac{\partial\mathbf{N}}{\partial\boldsymbol{\sigma}}
  = \frac{\partial}{\partial\boldsymbol{\sigma}}
    \left(\frac{\mathbf{s}}{2q}\right)$$

The derivative of the volumetric term is zero. Therefore the implementation
must not use a J2-style formula involving the full
$\mathbf{N}\otimes\mathbf{N}$ for the Drucker-Prager derivative.

### 9.3 Using converged-state quantities in the smooth tangent

The smooth-return update uses $\mathbf{N}^\mathrm{trial}$, so the smooth
tangent must also use trial-state quantities. Using converged-state
$\mathbf{N}$, $\mathbf{M}$, or
$\partial\mathbf{N}/\partial\boldsymbol{\sigma}$ gives a tangent for a
different algorithm.

### 9.4 Applying the smooth tangent at the apex

The smooth Drucker-Prager derivative contains factors of $1/q$ and is
singular at $q=0$. Once the apex branch is active, use the apex tangent
instead.
