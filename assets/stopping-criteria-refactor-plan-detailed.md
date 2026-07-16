# Detailed plan: unify stopping criteria + fix AL inner-tolerance schedule

**Repo:** `topohedral-optimize`  **Branch:** `feat/use-bound-in-auglag`
**Source:** derived from and verified against `assets/stopping-criteria-refactor-plan.md`, with three corrections found during code verification (see "Deviations" below).

## Context

Two independent problems currently make convergence checking inconsistent across solvers:

1. **`asa.rs` mixes norms within its own stopping test.** Its initial reference (`norm_grad_fx_init`) is computed with `abs_max()` (L∞) but the per-iteration residual it's compared against uses `.norm()` (L2). This makes the rtol test dimension-dependent (an L2 residual can be up to √n larger than the L∞ one) and inconsistent with BFGS-B, which already does this correctly end-to-end. QuasiNewton/ConjugateGradient are internally consistent but use L2 throughout (fine for unconstrained problems, but means the same numeric `grad_atol` means something different depending which solver runs it).
2. **The Augmented Lagrangian inner-tolerance formula can never reach tight user tolerances.** `set_inner_tolerances` recomputes an inner `grad_atol` from scratch every outer iteration via `clamp(0.1 * max(‖c‖, 1/penalty_max)^1.5, 1e-6, 1e-2)`. The `1e-6` floor means a user requesting `grad_atol = 1e-8` can never get an inner solve tight enough to deliver that, and because it's recomputed from scratch (not carried forward) it isn't monotone — it can get looser on a later outer iteration than it was on an earlier one.

Goal: make every solver's stationarity test use the **projected-gradient L∞ norm** (`‖P[x - ∇f] - x‖∞`, matching Byrd et al. 1995 / Birgin–Martínez), and make the AL inner tolerance a monotone sequence that floors at the user's actual `grad_atol` instead of a hardcoded constant. Per your instruction: tests should assert **convergence and correctness** (solution accuracy, KKT residuals), not specific iteration/eval counts — those are free to change and should be re-baselined from fresh runs rather than preserved by tuning inputs.

Work in two code phases (each independently compilable/testable, separate commits so regressions are bisectable), followed by a test re-baseline pass and new regression tests.

---

## Deviations from the drafted plan (found during verification)

1. **`src/line_search/mod.rs:72` is a required edit the draft missed.** `IterData.norm_grad_fx` for QuasiNewton/ConjugateGradient's iterations `k ≥ 1` is **not** produced by `IterData::new` (only `k = 0` is) — every subsequent iterate comes from `line_search::search()`, which independently computes `let new_norm_grad_fx = new_grad_fx.norm();` at line 72 and stuffs it into the returned `IterData` directly. If only `common.rs::IterData::new` is patched, QN/CG would compare L∞ on iteration 0 against L2 on every later iteration — silently worse than the current bug. This edit is now folded into Phase 1.
2. **AL's per-constraint penalty/multiplier logic should not be replaced.** `EqPenalty`/`IeqPenalty::update_penalties_shifts` already decide, **per constraint component**, whether to grow that component's penalty or update its shift — gated by `constraint_val > constraint_improvement_factor * previous_max_violation` (`update_max_violations`, `augmented_lagrangian.rs:163-202`). This is already a Birgin–Martínez-style adaptive scheme, just finer-grained (per-component) than the draft's proposed single global `η_k` gate. You confirmed (via the design-choice question) to keep this untouched and scope Phase 2 to just the inner-tolerance (`ω_k`) formula, which is where the actual reported bug lives. This makes Phase 2 much smaller than drafted: no `eta_k`/`mu_k` state, no split of `update_lagrangian`, no new penalty-struct methods.
3. **BFGS-B needs zero changes.** Confirmed `src/bound_constrained/bfgsb.rs` already uses a dedicated `projected_gradient_inf_norm()` helper (L∞) for both `is_converged` and `print_status`, end-to-end, and its `IterData.norm_grad_fx` field is set directly to `gk.norm()` (bypassing `IterData::new`) purely for display/trace — never read by its convergence logic. Included in the test gate only as a regression check.

---

## Phase 0 — Baseline (no code changes)

```
cargo test 2>&1 | tee /tmp/baseline_tests.txt
```
Just for diffing after each phase — no assertions on this output.

---

## Phase 1 — Unify stopping criteria on projected-gradient L∞

### 1.1 — `src/common.rs:113`, `IterData::new`

```rust
// BEFORE
let norm_grad_fx = grad_fx.norm();
// AFTER
let norm_grad_fx = grad_fx.abs_max().unwrap_or(0.0);
```
`VectorOps` (providing `.abs_max()`) is already imported in this file — no new imports needed.

### 1.2 — `src/line_search/mod.rs:72`, `search()`

```rust
// BEFORE
let new_norm_grad_fx = new_grad_fx.norm();
// AFTER
let new_norm_grad_fx = new_grad_fx.abs_max().unwrap_or(0.0);
```
This is the actual per-iteration value QuasiNewton/ConjugateGradient read via `iter_k.norm_grad_fx` for `k ≥ 1` — see Deviation 1 above. `VectorOps` already imported.

### 1.3 — `src/bound_constrained/asa.rs`: fix the mixed-norm bug

**Line 227, `is_converged`:**
```rust
// BEFORE
let projected_grad_norm = projected_grad.norm();
// AFTER
let projected_grad_norm = projected_grad.abs_max().unwrap_or(0.0);
```

**Line 442, `print_status`** (same change, trace-consistency only):
```rust
// BEFORE
let projected_grad_norm = projected_grad.norm();
// AFTER
let projected_grad_norm = projected_grad.abs_max().unwrap_or(0.0);
```

**Line 350** (`ngpa_step`, feeds the `IterData` it returns — diagnostic field only, not read by `is_converged`, but keep it consistent with the now-L∞ `IterData.norm_grad_fx` semantics):
```rust
// BEFORE
let norm_grad_fx_new = grad_fx_new.norm();
// AFTER
let norm_grad_fx_new = grad_fx_new.abs_max().unwrap_or(0.0);
```

**Line 660** (UA-phase branch of `minimize`, same rationale as 350):
```rust
// BEFORE
iter_k.norm_grad_fx = iter_k.grad_fx.norm();
// AFTER
iter_k.norm_grad_fx = iter_k.grad_fx.abs_max().unwrap_or(0.0);
```

**Do not touch** (verified internally-consistent, not stopping-criterion related):
- Line 192 — `norm_grad_fx_init` init, already `.abs_max()`.
- Lines 307–308 — `_d_norm`/`_d_inf_norm` NGPA step diagnostics (both unused/trace-only, one already `.abs_max()`).
- Line 314 — degenerate-direction guard, already `.abs_max()`.
- Lines 547/549 — NGPA→UA phase switch (`μ`-test): compares two L2 sub-norms to each other; internally consistent under either norm, not a stopping comparison.
- Lines 665/667 — UA→NGPA phase switch, same rationale.

### 1.4 — QuasiNewton / ConjugateGradient: reference norm

**`src/unconstrained/quasi_newton.rs:63`, `fn new`:**
```rust
// BEFORE
let norm_grad_0 = grad_0.norm();
// AFTER
let norm_grad_0 = grad_0.abs_max().unwrap_or(0.0);
```

**`src/unconstrained/conjugate_gradient.rs:61`, `fn new`:**
```rust
// BEFORE
let norm_grad_0 = grad_0.norm();
// AFTER
let norm_grad_0 = grad_0.abs_max().unwrap_or(0.0);
```
No other changes needed in these two files — `is_converged` in both already just receives whatever `f64` it's handed (`iter_k.norm_grad_fx`), which 1.1/1.2 make L∞.

### 1.5 — Re-baseline tests

Because `‖·‖∞ ≤ ‖·‖₂` always, switching the same numeric `grad_atol`/`grad_rtol` from an L2 to an L∞ comparison makes solvers trigger convergence **at least as early** — expect iteration/eval counts to drop or stay flat, and `ConvergedReason` may flip between `Rtol`/`Atol` for some cases.

- **`tests/quasi_newton.rs`, `tests/conjugate_gradient.rs`:** These use a shared `assert_returns` helper that does `assert_eq!` on `num_fun_evals`, `num_grad_evals`, and `reason`, plus tolerance checks on `xmin`/`fmin`. Run the suite, and for every failing case:
  - If it fails only on `num_fun_evals`/`num_grad_evals`/`num_iterations`/`reason` — update those literals in the `#[case(...)]` attribute to the new actual run output. This is expected and desired per your instruction.
  - If the `xmin`/`fmin` tolerance assertion itself fails (the solver now stops too early to be within tolerance of the known analytical optimum) — tighten the `grad_atol`/`grad_rtol` **input** in that case's `Options` so the solver still reaches the required accuracy. Do not loosen the `xmin_tol`/`fmin_tol` assertion threshold to paper over this; that's weakening a correctness check rather than updating a captured reference.
- **`tests/asa.rs`:** No eval/iteration-count assertions exist here, so most tests should just start passing (the mixed-norm bug is exactly what was making ASA converge inconsistently). One exception: `asa_minimizes_ten_dimensional_rosenbrock_outside_box` hardcodes an `expected_x`/`expected_f` captured from a prior run (not a closed-form solution, since the active-bound KKT point of a bound-constrained Rosenbrock has no simple analytical form). Re-run, capture the new converged point, and **before** updating the literal, confirm the existing `kkt_residual(...)  <= 1e-6` assertion in that test still passes on the new point — this guards against accidentally baking in a non-optimal result.
- **`tests/bfgsb.rs`:** Expected to be unaffected (BFGS-B untouched). Run as a pure regression check; if anything changes here, treat it as a signal something in Phase 1 leaked into a shared path and investigate rather than just updating the literal.

### 1.6 — Gate

```
cargo build && cargo test --test quasi_newton --test conjugate_gradient --test asa --test bfgsb
```
All four green before starting Phase 2.

---

## Phase 2 — AL inner-tolerance (`ω_k`) monotone floor fix

**File:** `src/constrained/augmented_lagrangian.rs`

Scope, per Deviation 2: only `set_inner_tolerances` and its supporting state change. `update_lagrangian`, `EqPenalty`/`IeqPenalty::update_penalties_shifts`, and `is_converged` (already L∞-correct, verified) are untouched.

### 2.1 — Add `omega_k` state to the struct

Struct is at line 763:
```rust
pub struct AugmentedLagrangian<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn>
{
    fcn: Arc<Mutex<CountingRealFn<AugmentedLagrangianFcn<F1, F2, F3>>>>,
    x_init: Vector,
    bounds: Option<BoundsConstraints>,
    opts: Options,
    omega_k: f64,   // ADD: monotone ceiling for the inner stationarity tolerance
}
```

Add near the other module consts:
```rust
/// Ceiling for ω_k (inner stationarity tolerance) on the very first outer
/// iteration — matches the old `set_inner_tolerances` atol upper clamp.
/// From the second outer iteration on, the previous ω_k becomes the ceiling
/// instead, which is what makes the sequence monotone non-increasing.
const OMEGA_INIT_CEIL: f64 = 1e-2;
```

In `new()` (line 775), compute the user's floor before `opts` is moved into `Self`:
```rust
// BEFORE (near end of `new`)
Self {
    fcn: fcn_shared,
    x_init: x0,
    bounds,
    opts,
}

// AFTER
let user_omega = opts.constrained_opts.base_opts.grad_atol;
Self {
    fcn: fcn_shared,
    x_init: x0,
    bounds,
    opts,
    omega_k: OMEGA_INIT_CEIL.max(user_omega),
}
```
(`.max(user_omega)` only matters if someone configures `grad_atol` looser than `1e-2`; otherwise it's `1e-2`, identical to today's first-iteration behavior.)

### 2.2 — Add `compute_omega_k`, replace `set_inner_tolerances`

Replace the whole body of `set_inner_tolerances` (currently lines 951–1061) with:

```rust
/// Computes this outer iteration's inner stationarity tolerance ω_k and
/// advances the stored ceiling for the next call.
///
/// Keeps the existing LANCELOT-style scaling
/// `0.1 * max(‖c‖∞, 1/penalty_max)^1.5`, but clamps to
/// `[user_grad_atol, previous ω_k]` instead of the old fixed `[1e-6, 1e-2]`.
/// That makes the sequence monotone non-increasing (never loosens between
/// outer iterations) and lets it reach tolerances tighter than the old
/// 1e-6 floor when the user asks for them via `grad_atol`.
#[trace_fn]
fn compute_omega_k(&mut self) -> f64
{
    let user_omega = self.opts.constrained_opts.base_opts.grad_atol;
    let prev_omega_k = self.omega_k;

    let omega_k = self.fcn.lock().unwrap().with_inner_mut(|fcn| {
        let (norm_eq, max_penalty_eq) = if let Some(eq_penalty) = &fcn.eq_penalty
        {
            (
                eq_penalty.data.values.abs_max().unwrap(),
                eq_penalty.data.penalties.allmax().unwrap(),
            )
        }
        else
        {
            (0.0, 1.0)
        };
        let (norm_ieq, max_penalty_ieq) = if let Some(ieq_penalty) = &fcn.ieq_penalty
        {
            (
                ieq_penalty.data.values.posed().abs_max().unwrap(),
                ieq_penalty.data.penalties.allmax().unwrap(),
            )
        }
        else
        {
            (0.0, 1.0)
        };

        let residual_primal = norm_eq.max(norm_ieq);
        let penalty_max = max_penalty_eq.max(max_penalty_ieq);
        let omega_target = 0.1 * residual_primal.max(1.0 / penalty_max).powf(1.5);
        omega_target.max(user_omega).min(prev_omega_k)
    });

    self.omega_k = omega_k;
    //{{{ trace
    info!(target: "aug", "Inner ω_k = {omega_k:.4e} (prev = {prev_omega_k:.4e}, user floor = {user_omega:.4e})");
    //}}}
    omega_k
}

#[trace_fn]
fn set_inner_tolerances(&mut self) -> InnerMethod
{
    let inner_method = self.opts.inner_method.clone();
    let is_constrained = self.fcn.lock().unwrap().with_inner_mut(|fcn| fcn.is_constrained());

    match inner_method
    {
        InnerMethod::Unconstrained(mut uncon_method) =>
        {
            if !is_constrained
            {
                uncon_method.uncon_opts_mut().grad_rtol =
                    self.opts.constrained_opts.base_opts.grad_rtol;
                uncon_method.uncon_opts_mut().grad_atol =
                    self.opts.constrained_opts.base_opts.grad_atol;
            }
            else
            {
                let omega_k = self.compute_omega_k();
                uncon_method.uncon_opts_mut().grad_rtol = 0.0;
                uncon_method.uncon_opts_mut().grad_atol = omega_k;
            }
            InnerMethod::Unconstrained(uncon_method)
        }
        InnerMethod::BoundConstrained(mut bcon_method) =>
        {
            if !is_constrained
            {
                bcon_method.bound_opts_mut().base_opts.grad_rtol =
                    self.opts.constrained_opts.base_opts.grad_rtol;
                bcon_method.bound_opts_mut().base_opts.grad_atol =
                    self.opts.constrained_opts.base_opts.grad_atol;
            }
            else
            {
                let omega_k = self.compute_omega_k();
                bcon_method.bound_opts_mut().base_opts.grad_rtol = 0.0;
                bcon_method.bound_opts_mut().base_opts.grad_atol = omega_k;
            }
            InnerMethod::BoundConstrained(bcon_method)
        }
    }
}
```

Notes:
- `set_inner_tolerances` changes from `&self` to `&mut self`. Its single call site (`minimize`, line 1099: `let uncon_method = self.set_inner_tolerances();`) needs **no edit** — it already runs inside `minimize(&mut self)`.
- This also removes the near-duplicate atol-formula code that existed independently in the `Unconstrained` and `BoundConstrained` branches (flagged during verification) by hoisting the shared math into `compute_omega_k` and the `is_constrained` check into one place above the `match`.
- `VectorOps` (for `.abs_max()`/`.allmax()`/`.posed()`) is already imported and used elsewhere in this file — no new imports.

### 2.3 — Leave `update_lagrangian`, `is_converged`, and the penalty structs untouched

Verified `is_converged` (lines 809–885) already measures the projected classical-Lagrangian gradient in `abs_max()` (L∞) and the constraint residual via `.posed().abs_max()` — no changes needed. `update_lagrangian` (line 918) and `EqPenalty`/`IeqPenalty::update_penalties_shifts` keep their existing per-constraint penalty-vs-shift decision exactly as-is.

*(Optional, low-priority polish — not required for correctness): the `info!` at `minimize`, line 1114 logs `iter_k.norm_grad_fx`, which is an L∞-but-unprojected, non-classical-Lagrangian gradient norm — cosmetically misleading next to the actual convergence measure. Skip unless you want to touch it; it has no effect on behavior or tests.)*

### 2.4 — Re-baseline `tests/augmented_lagrangian.rs`

This file uses `assert_answer` (relative-tolerance `xmin`/`fmin` check — these are geometrically/analytically derived and should be unaffected) and `assert_counts` (`<=` upper bounds on `num_fun_evals`/`num_grad_evals`, not exact). Run the full suite after Phase 1 + Phase 2 together (AL's behavior depends on both):
- If a case's `num_fun_evals`/`num_grad_evals` now exceeds its `#[case(...)]` cap, raise the cap to the new actual count (plus don't add artificial slack — use the real observed number, consistent with your instruction that these are free to change).
- If a case's count drops well below its existing cap, it's fine to leave the cap as-is (test still passes) or tighten it down to the new count for a more meaningful regression guard — your call, not required.
- If any `assert_answer` (accuracy) assertion fails, treat it the same as in Phase 1: first try tightening `grad_atol`/`constraint_tol` inputs for that case; don't loosen the xmin/fmin tolerance without first checking whether the solve is actually converging correctly.

### 2.5 — Gate

```
cargo build && cargo test --test augmented_lagrangian
```

---

## Phase 3 — New regression tests (guard the fixed bugs)

Add to `tests/augmented_lagrangian.rs`:
1. **Tight-tolerance regression:** a case with `grad_atol = 1e-8` (tighter than the old `1e-6` floor) on a problem with a known solution (reuse one of the existing quadratic/circle-bound problems). Assert convergence succeeds (not `MaxIterations`) and the achieved solution meets `xmin`/`fmin` accuracy consistent with `1e-8`-level stationarity — no iteration/eval-count assertion. This would have failed to converge to that accuracy under the old clamp and guards against reintroducing it.
2. *(Optional strengthening, matches the drafted plan's Phase 3 item 1)* pick one existing bound-constrained AL case and add an explicit assertion on the KKT residual directly (projected classical-Lagrangian gradient `abs_max` ≤ `grad_atol`, constraint residual `abs_max` ≤ `constraint_tol`) using the same pattern as `kkt_residual` in `tests/asa.rs`/`tests/bfgsb.rs`, rather than relying only on the relative `xmin`/`fmin` check.

Add to `tests/asa.rs`:
3. **Dimension-independence regression:** a larger-`n` (e.g. `n = 500`) bound-constrained problem (mirrors an existing smaller test, e.g. the shifted-quadratic-with-active-bound case scaled up). Assert convergence succeeds and `kkt_residual(...)` is small (same bound as the existing tests, e.g. `≤ 1e-6`) — do **not** assert a specific iteration count or compare it numerically to the small-`n` case's iteration count; just confirm it converges within the configured `max_iter`. This guards against the dimension-dependent-rtol bug (L2-numerator-vs-L∞-reference) that Phase 1 fixed, without pinning a fragile count.

### Gate

```
cargo test
```
Full suite green.

---

## Order-of-operations checklist

- [ ] Phase 0 baseline saved
- [ ] 1.1 `common.rs:113` → `abs_max`
- [ ] 1.2 `line_search/mod.rs:72` → `abs_max` (deviation from drafted plan — required)
- [ ] 1.3 `asa.rs` lines 227, 442, 350, 660 → `abs_max`; lines 192/307-308/314/547/549/665/667 untouched
- [ ] 1.4 `quasi_newton.rs:63`, `conjugate_gradient.rs:61` → `abs_max`
- [ ] 1.5 re-baseline `tests/quasi_newton.rs`, `tests/conjugate_gradient.rs`, `tests/asa.rs`; verify `tests/bfgsb.rs` unaffected
- [ ] 1.6 gate green
- [ ] 2.1 `omega_k` field + `OMEGA_INIT_CEIL` const + `new()` init
- [ ] 2.2 `compute_omega_k` + rewritten `set_inner_tolerances` (now `&mut self`)
- [ ] 2.3 confirm `update_lagrangian`/`is_converged`/penalty structs untouched
- [ ] 2.4 re-baseline `tests/augmented_lagrangian.rs`
- [ ] 2.5 gate green
- [ ] Phase 3: tight-tolerance AL regression test, optional KKT-residual strengthening, large-n ASA regression test
- [ ] Full `cargo test` green

## Invariants to check while editing

- `omega_k` is non-increasing across outer iterations and always `≥` `user_omega` (`opts.constrained_opts.base_opts.grad_atol`).
- After Phase 1, every solver's stopping comparison (`is_converged`) uses `abs_max()` on a projected (bounds-aware where applicable) gradient — confirm by grepping for `.norm()` in `is_converged` functions across `src/`; none should remain except the internally-consistent NGPA/UA phase-switch comparisons in `asa.rs` (lines 547/549/665/667), which are deliberately left alone.
- `cargo build` with no warnings about unused `atol_min`/`atol_max` locals (removed in 2.2) or unused `alpha`/`beta` bindings in `minimize` (both still used — `update_lagrangian` is untouched, so no cleanup needed there).
