# Review — CalculiX external-behaviour adapter

> **STATUS: all findings addressed.** See "Resolution" at the end. The three
> bugs that could produce a silently wrong result (C1, H1, H2) each now have a
> regression test that was *verified to fail* against the old code before the
> fix was restored.


Scope: `include/numsim-materials/umat/calculix_interface.h` and
`tests/test_calculix_interface.cpp` (the native `call_external_umat_user` hook).

Method: 3-lens parallel review (architecture / C++-ABI / code-quality &
coverage). Every Critical and High finding below was independently
re-verified by me against the CalculiX 2.22 source before landing here — one
agent's "packing is correct" was **wrong** (see H1), so the severities are mine,
not a straight merge of the agents'.

The convention translation is correct **for the one configuration the single
test exercises** (`nstatv==0`, `iel=iint=1`, `iorien=0`, `time=0`, `emec0=0`,
one increment, symmetric tangent). Every finding below is a case that leaves
that configuration — and all are currently masked by it.

---

## Critical

### C1 — STATEV is addressed at point (1,1) for *every* integration point
`calculix_interface.h:90-91,103` (macro drops `IEL`/`IINT`/`MI`, `:149,156`)

CalculiX passes the **whole** state arrays' base to the external hook, not a
per-point slice. Verified:
- `umat_main.f:40` — `real*8 xstate(nstate_,mi(1),*), xstateini(nstate_,mi(1),*)`
- `umat_main.f:206` (native `umat_user`) and the `@` branch at `:233` both pass
  bare `xstateini,xstate` — **no** `(1,iint,iel)` slice, unlike
  `umat_abaqus.f:295` which slices `xstate(1,iint,iel)` before its call.

So the external `.so` must index state itself. The adapter uses offset 0
(`std::copy_n(statev_old, …, statev_new)` and `a.statev = statev_new`), so every
Gauss point of every element reads and writes the **same** state slot; all other
points are never seeded or updated. Silent wrong physics for any model with
`nstatv > 0`. Invisible today only because linear-elastic has `nstatv == 0`
(the copy is a no-op) and the test uses `iel=iint=1` (offset 0).

`emec`/`emec0` (6) and `stre`/`stiff` (6/21) are per-point vectors in
`umat_main.f:36`, **not** 3-D arrays — so only STATEV needs slicing.

**Fix:** wire `IEL`, `IINT`, `MI` through the macro; compute
`offset = nstatv_v * ((iint-1) + (*mi)*(iel-1))` and use `statev_old+offset` /
`statev_new+offset` for both the seed copy and `a.statev`. Add a stateful test
at a non-`(1,1)` point so a regression cannot hide at offset 0.

---

## High

### H1 — Tangent is transposed for major-asymmetric materials
`calculix_interface.h:131` + `tensor_conversion.h:146`, `material_point_evaluator.h:82,181`

`umat_dispatch` writes `a.ddsdde` (my `ddsdde36`) **column-major**:
`material_point_evaluator.h:82` documents it, `narrow_matrix`
(`tensor_conversion.h:146`) does `host[a + b*n] = C(a,b)`. So
`ddsdde36[i + j*6] == C(i,j)`, and my packing read `ddsdde36[i*6 + j]` is
`C(j,i)` — the transpose. Harmless for a major-symmetric tangent (linear
elastic), silently wrong for non-associative plasticity / damage. The comment
"row- vs column-major indexing … is moot" asserts an unenforced assumption.

The cross-check test cannot catch it: `ref_ddsdde` is filled by the same
column-major path and read with the same `[i*6+j]`, so both sides carry the same
transpose; the shear test probes only the diagonal `(3,3)`, index-invariant.

**Fix:** symmetrize exactly as the reference `umat_abaqus.f:335-355` does
(`stiff = (ddsdde(i,j)+ddsdde(j,i))/2`):
`stiff[i+j*(j+1)/2] = 0.5*(ddsdde36[i + j*6] + ddsdde36[j + i*6])`.
For symmetric tangents this is identity; for asymmetric it matches ccx (which
keeps only the symmetric part in `stiff(21)`). Add a test with a known
asymmetric `C` asserting an off-diagonal lands at a specific `stiff` index.

### H2 — TIME(1)/TIME(2) mapping is wrong (and the comment claims it is right)
`calculix_interface.h:97-99`

Verified against `umat_abaqus.f:187-188`:
`abqtime(1) = time - dtime`, `abqtime(2) = ttime + time - dtime`, where ccx
`time` = step time at the **end** of the increment, `ttime` = total time at the
**start of the step**. The adapter passes `{time, ttime}`, so TIME(1) is off by
`dtime` and TIME(2) is short by `(time - dtime)`. Correct only on the first
increment of the first step. Any time/rate-dependent model (creep,
viscoelasticity, the `external_scalar_source` time consumers) gets the wrong
absolute time on every later increment. The comment on `:98` is a false claim.

**Fix:** `dt = dtime?*dtime:0; t = time?*time:0; tt = ttime?*ttime:0;`
`time2 = { t - dt, tt + t - dt }`. Correct the comment.

### H3 — `iorien != 0` silently ignored; the native hook makes rotation the user's job
`calculix_interface.h:44-49` (macro drops `IORIEN`/`PGAUSS`/`ORAB`, `:156`)

For `umat_abaqus.f` ccx rotates strain in / stress+stiffness out around the
call, so a UMAT never sees orientation. The native `umat_user` has **no** such
wrapper: `umat_user.f:86-104` requires results in the material frame and tells
the user to call `transformatrix(orab(1,iorien),…)`; `umat_main.f` does no
rotation around the external call. So with `*ORIENTATION` in the deck,
`iorien != 0` reaches the adapter and results come back in the wrong frame —
silently wrong. The header reuses the Abaqus-entry justification, which does not
transfer.

**Fix:** wire `IORIEN`; `throw fatal_error` when `*iorien != 0` until real
rotation exists. Converts a silent wrong answer into a hard stop; cheap.

### H4 — Stress/strain measure is small-strain-only, unguarded
`calculix_interface.h:31,44-49`

`emec` is Green-Lagrange, `stre` is PK2; the models consume/emit them as small
strain. For a `C:E` elastic law this coincidentally *is* St-Venant–Kirchhoff, so
it is correct. Under `NLGEOM` ccx passes large Green-Lagrange strains; the first
inelastic model then feeds finite strain into a small/log-strain return map and
labels the output PK2 — a wrong converged stress, not merely a slow tangent.
The single test (`emec0=0`, one increment) cannot see this.

**Fix:** document the scope as *geometrically-linear / small-strain only* far
more forcefully than "targets first"; consider a magnitude guard.

---

## Medium

- **M1 — No real STATEV-flow test.** All tests are single-call linear-elastic,
  so the `xstateini→xstate` seeding is a no-op and untested. Add a
  multi-increment J2/hardening driver committing `statev_new→statev_old` between
  steps, cross-checked vs the direct evaluator. (Also guards C1.)
  `test_calculix_interface.cpp`
- **M2 — Nonzero `emec0` never tested.** Every case has `emec0=0`, so
  `stran = 2·emec0 = 0` and the stran/dstran split is unverified; only the sum
  is exercised. Add a case with `emec0` nonzero in all six slots.
- **M3 — 16-arg positional raw-pointer API.** `calculix_dispatch` abandons the
  misuse-resistant `dispatch_args<T>` aggregate the Abaqus side uses; seven
  `const double*` in a row make a `time`/`ttime` or `emec`/`emec0` swap compile
  cleanly — exactly the class of bug H2 is. Consider a `calculix_args<T>`
  aggregate or do the translation inside the macro. `calculix_interface.h:57-64`
- **M4 — `beta` and `ielas` dropped without a guard.** `beta` =
  `*INITIAL CONDITIONS,TYPE=STRESS` (`umat_user.f:52`) — a preloaded model is
  wrong from step 1; `ielas==1` requests an elastic response ccx will later
  mis-use. Wire and guard (fatal if `beta` nonzero) rather than leave un-named.
- **M5 — No error-path coverage.** The sibling suite's `FatalProbe` +
  `set_fatal_handler` (`test_umat_interface.cpp:322-335`) is available. Cover:
  unknown model → fatal + zeroed outputs; too-few constants → `require_props`
  fatal (the `nconst=-kode-100` decode is otherwise happy-path only); a
  convergence failure → `pnewdt` cutback propagated back through the adapter.
- **M6 — `emec`/`emec0` inconsistently null-guarded** vs every other pointer;
  a null there segfaults rather than degrading. Guard for consistency.
  `calculix_interface.h:78-84`

## Low

- **L1 — `nstatv_v` not clamped ≥ 0** before flowing to
  `umat_interface.h:440`'s `static_cast<size_t>`, where a negative wraps to a
  huge span. Shared with the pre-existing `umat_` path; unreachable via a valid
  deck. Clamp `*nstatv > 0 ? *nstatv : 0`.
- **L2 — `-(*kode)-100` is UB if `*kode==INT_MIN`.** Theoretical; widen to
  `long long` if desired. `:71`
- **L3 — Fortran hidden-length hardcoded `int`** vs the configurable
  `NUMSIM_MATERIALS_FORTRAN_STRLEN` the Abaqus macro exposes. Benign (last arg,
  `amat` unused) but inconsistent with the stated lesson.
- **L4 — Magic `6`/`36`/`21`.** A `constexpr` would self-document the packing.
- **L5 — Doc overclaim** `:35` "verified bit-identical against ccx built-in
  *ELASTIC" — that verification was the compiled-in `umat_user_` target
  (S11=0.027), not this external `.so` path. Soften or cite.
- **L6 — Abaqus concepts behind a CalculiX-named file.** The `umat_dispatch`
  reuse is the right seam, but the header should name the inherited
  error/cutback semantics (`pnewdt=0.25`, zero-outputs) so a reader doesn't
  assume this file owns them.

---

## Suggested sequencing (one PR)

1. **C1 + H1 + H2 + H3** — the code-correctness fixes; all small, all currently
   masked. C1 changes the macro signature (wire `IEL`/`IINT`/`MI`), so do it
   first.
2. **M1 + M2 + M5** — the tests that would have caught C1/H1/H2 and prevent
   regressions (stateful multi-increment at a non-(1,1) point; nonzero `emec0`;
   asymmetric-tangent packing; error paths).
3. **H4 + M3 + M4 + L*** — hardening and documentation; M3 (aggregate) is the
   structural change that closes the transposition class M-wide.

---

## Resolution

All findings applied in `calculix_interface.h` / `test_calculix_interface.cpp`.
Full suite: **293 tests, 0 failures** (282 before, 11 new).

| # | Fix |
|---|---|
| C1 | `IEL`/`IINT`/`MI` wired through the macro; state sliced at `nstatv·((iint-1) + mi1·(iel-1))`, with the index triple validated (`iint ≤ mi1`, all ≥ 1) and a fatal on inconsistency. |
| H1 | `stiff(21)` now carries the **symmetrized** tangent, `0.5·(D[i+j·6] + D[j+i·6])`, exactly as `umat_abaqus.f:335-355`. Reads the buffer column-major, matching `narrow_matrix`. |
| H2 | TIME rebased onto the start of the increment: `{time-dtime, ttime+time-dtime}`, mirroring `umat_abaqus.f:187-188`. Comment corrected. |
| H3 | `iorien != 0` now zeroes outputs and reports a fatal instead of returning wrong-frame results. |
| H4 | Header carries an explicit **SCOPE — GEOMETRICALLY LINEAR (SMALL STRAIN) ONLY** section stating the PK2/Green-Lagrange pairing is exact only for `C:E`, and that an inelastic model under NLGEOM would be silently wrong. |
| M1 | `IndexesStateByElementAndIntegrationPoint`: 40-increment J2 driven at `iel=2, iint=3`, cross-checked against the Abaqus path, asserting state landed in the right block, that it is non-trivial, and that **no other block was touched**. |
| M2 | The cross-check now uses a nonzero `emec0` in all six slots, so the `stran`/`dstran` split is observable. |
| M3 | Introduced the `calculix_args` aggregate; the 16-arg positional list is gone, closing the transposition class that produced H2. |
| M4 | `beta` wired and guarded (nonzero ⇒ fatal), with a test that zero `beta` still passes. `ielas` wired and documented as ignored. |
| M5 | Error paths covered: unknown model, `kode`-decoded constant shortfall, orientation, initial stress — all via `FatalProbe`. |
| M6 | `emec`/`emec0` null-guarded consistently with the rest; pointer contract documented. |
| L1 | `nstatv` clamped to ≥ 0 before it reaches the `size_t` cast. |
| L2 | `-kode-100` widened to `long long` before negation; count clamped ≥ 0. |
| L3 | Resolved *opposite* to the suggestion: the trailing length is `int` because `call_external_umat_user.c`'s C typedef fixes it — unlike Abaqus, where the Fortran compiler chooses. Documented; deliberately **not** made configurable. |
| L4 | `calculix_ntens` / `calculix_nstiff` replace the magic 6 / 21 / 36. |
| L5 | The "verified bit-identical against ccx built-in *ELASTIC" claim removed — that run was the compiled-in `umat_user_` target, not this `.so` path. |
| L6 | Header documents the inherited `umat_dispatch` error semantics (fatal-zeroes-and-terminates, `PNEWDT = 0.25` cutback, zeroed 6×6 packed through). |

### Regression tests verified to fail against the old code

Each was re-run with the fix reverted, to prove it is not vacuous:

- **C1** → `state written outside this point's block at 4`
- **H1** → `stiff(0,1)`, `stiff(0,2)`, `stiff(1,2)` mismatched
- **H2** → `passing ttime straight through would give 5.0 here`

### Still open

- No test forces a genuine convergence failure to assert the `PNEWDT` cutback
  round-trips through the adapter. The path is inherited unchanged from
  `umat_dispatch` (which the sibling suite covers), so this is a gap in
  *adapter-level* coverage only.
- `ielas == 1` (elastic-iteration request) is accepted and ignored. Harmless for
  an elastic model; an inelastic one will need to honour it.
- End-to-end validation against a real ccx run through the `@LIB,FUNC` deck path
  has not been done for this adapter.
