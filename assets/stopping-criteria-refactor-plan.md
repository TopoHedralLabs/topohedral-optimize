# Refactor plan: unify stopping criteria + fix AL↔inner tolerance coupling

**Repo:** `topohedral-optimize`  **Branch:** `feat/use-bound-in-auglag`
**Goal:** (1) make every solver's KKT/stationarity test use the **projected-gradient L∞ norm**, and (2) replace the AL inner-tolerance heuristic with a proper monotone `(η_k, ω_k)` Birgin–Martínez / LANCELOT schedule that converges to the user's requested tolerances.

Work in three phases, each independently compilable and testable. Do **not** combine phases in one commit — the norm change re-baselines tests, and mixing it with the AL change makes regressions impossible to bisect.

Reference norm convention adopted repo-wide: the stationarity measure is `‖ P[x - ∇f] - x ‖_∞` (projected gradient, sup-norm), matching Byrd et al. 1995, Hager–Zhang 2006, and Birgin–Martínez/Algencan. In this codebase `abs_max()` is the L∞ reduction and `norm()` is the L2 norm.

---

## Phase 0 — Preflight (no code changes)

Run and save baselines so you can diff convergence behavior after Phase 1:

```
cargo test 2>&1 | tee /tmp/baseline_tests.txt
cargo test --test asa 2>&1 | tee /tmp/baseline_asa.txt
cargo test --test augmented_lagrangian 2>&1 | tee /tmp/baseline_al.txt
```

Note which tests pass now. Phase 1 may shift iteration counts and tolerances in `tests/quasi_newton.rs`, `tests/conjugate_gradient.rs`, `tests/asa.rs`; that's expected and handled in Phase 1 step 4.

---

## Phase 1 — Unify stopping criteria on projected-gradient L∞

### 1.1 — ASA: fix the mixed-norm bug (highest priority, correctness)

**File:** `src/bound_constrained/asa.rs`

The initializer at line ~192 sets `norm_grad_fx_init` from `abs_max()` (L∞), but `is_converged` at line ~227 measures the per-iteration residual with `.norm()` (L2) and compares it against that L∞ reference. This makes the rtol test dimension-dependent (harder by up to √n) and the atol test inconsistent with BFGS-B/AL.

**Edit — the convergence test (line ~227), inside `fn is_converged`:**

```rust
// BEFORE
let projected_grad_norm = projected_grad.norm();
// AFTER
let projected_grad_norm = projected_grad.abs_max().unwrap_or(0.0);
```

**Edit — the diagnostic copy in `print_status` (line ~442):** same change, for trace consistency only:

```rust
// BEFORE
let projected_grad_norm = projected_grad.norm();
// AFTER
let projected_grad_norm = projected_grad.abs_max().unwrap_or(0.0);
```

**DO NOT TOUCH** the following `.norm()` calls in `asa.rs` — they are NGPA/UA phase-switching and μ-test quantities, not the stopping criterion. Changing them alters algorithm behavior, not convergence detection:
- line ~307 `_d_norm` (NGPA step diagnostic)
- line ~350 `norm_grad_fx_new` (feeds `IterData.norm_grad_fx`; see 1.3 for the coordinated fix)
- line ~547 / ~549 `projected_grad_norm` / `inactive_grad_norm` (NGPA switch test `‖∇f_inactive‖ < μ‖∇f_proj‖`)
- line ~660 `iter_k.norm_grad_fx` (see 1.3)
- line ~665 / ~667 `projected_grad_norm_new` / `inactive_grad_norm_new` (UA switch test)

> Rationale: the μ-test compares two gradient sub-norms to each other, so it's internally consistent under either norm. Only the *stopping* comparison (residual vs. a fixed reference/threshold) is norm-sensitive.

### 1.2 — QuasiNewton and ConjugateGradient: switch stationarity to L∞

**Files:** `src/unconstrained/quasi_newton.rs`, `src/unconstrained/conjugate_gradient.rs`

Both compute `norm_grad_0 = grad_0.norm()` in `new()` and test `iter_k.norm_grad_fx` (L2) in `is_converged`. `IterData.norm_grad_fx` is populated by `IterData::new` in `src/common.rs` via `grad_fx.norm()`. The cleanest unifying change is in `IterData` itself (1.3), which flips both unconstrained solvers automatically. **After doing 1.3**, update the reference-norm computed in each `new()`:

`quasi_newton.rs`, `fn new` (line ~55):
```rust
// BEFORE
let norm_grad_0 = grad_0.norm();
// AFTER
let norm_grad_0 = grad_0.abs_max().unwrap_or(0.0);
```

`conjugate_gradient.rs`, `fn new` (line ~60):
```rust
// BEFORE
let norm_grad_0 = grad_0.norm();
// AFTER
let norm_grad_0 = grad_0.abs_max().unwrap_or(0.0);
```

No other change needed in these two files — they read `iter_k.norm_grad_fx`, which 1.3 makes L∞.

### 1.3 — `IterData`: make `norm_grad_fx` the L∞ norm

**File:** `src/common.rs`, `impl IterData`, `fn new` (the `norm_grad_fx` field is the shared stationarity scalar consumed by QN/CG and set in several ASA spots).

```rust
// BEFORE
let norm_grad_fx = grad_fx.norm();
// AFTER
let norm_grad_fx = grad_fx.abs_max().unwrap_or(0.0);
```

**Coordinated ASA edits** so `norm_grad_fx` stays L∞ where ASA reassigns it directly:
- `asa.rs` line ~350: `let norm_grad_fx_new = grad_fx_new.abs_max().unwrap_or(0.0);`
- `asa.rs` line ~660: `iter_k.norm_grad_fx = iter_k.grad_fx.abs_max().unwrap_or(0.0);`

> Note: ASA's own stopping test uses the *projected* gradient (fixed in 1.1), not `norm_grad_fx`. These two lines only keep the `IterData` field's semantics consistent (raw-gradient L∞) for diagnostics and any downstream reader. BFGS-B already uses `projected_gradient_inf_norm` internally and needs **no change**.

### 1.4 — Re-baseline unconstrained/ASA tests

The switch from L2 to L∞ loosens the numeric threshold (‖·‖∞ ≤ ‖·‖₂), so existing tests asserting a specific `grad_atol` will now converge at a *looser* true-L2 residual. Two options, pick per test:

- If a test asserts solution accuracy (‖x - x*‖ or f - f*), tighten `grad_atol` in that test's options until the accuracy assertion passes again. Prefer this.
- If a test asserts an iteration count, update the expected count from the new run.

Files likely affected: `tests/quasi_newton.rs`, `tests/conjugate_gradient.rs`, `tests/asa.rs`. Diff against `/tmp/baseline_*.txt`. Do not weaken solution-accuracy assertions; only adjust the *tolerance inputs* and *iteration-count expectations*.

### 1.5 — Gate

```
cargo build && cargo test --test quasi_newton --test conjugate_gradient --test asa --test bfgsb
```

All four must pass before starting Phase 2.

---

## Phase 2 — AL outer/inner tolerance schedule (the robustness fix)

**File:** `src/constrained/augmented_lagrangian.rs`

### Problem being fixed
`set_inner_tolerances` (lines ~951–1061) hard-sets inner `grad_rtol = 0.0` and derives inner `grad_atol` as
`clamp(0.1 * max(‖c‖, 1/penalty_max)^1.5, 1e-6, 1e-2)`.
The `[1e-6, 1e-2]` clamp is disconnected from the user's requested `grad_atol`: tighter user tolerances can never be reached (inner floors at 1e-6), and early iterations stop too loose (1e-2), poisoning the multiplier estimate `λ = penalty * shift`. It also recomputes from scratch each outer iteration — not a monotone sequence, and it never converges to the user targets. This is the "same settings, works sometimes" failure.

### Fix: maintain monotone `(η_k, ω_k)` that floor at the user's `(constraint_tol, grad_atol)`

Standard LANCELOT/Birgin–Martínez control:
- `ω_k` = subproblem stationarity tolerance handed to the inner solver (interpreted in the **inner Lᴀ metric**: inner stops at `‖P[∇Lᴀ]‖_∞ ≤ ω_k`).
- `η_k` = feasibility tolerance gating multiplier-update vs. penalty-increase.
- Both tightened multiplicatively, floored at user targets `ω_* = grad_atol`, `η_* = constraint_tol`.

### 2.1 — Add tolerance state to the AL struct

Struct is at line ~764:
```rust
pub struct AugmentedLagrangian<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn>
{
    fcn: Arc<Mutex<CountingRealFn<AugmentedLagrangianFcn<F1, F2, F3>>>>,
    x_init: Vector,
    bounds: Option<BoundsConstraints>,
    opts: Options,
    // ADD:
    omega_k: f64,   // current inner stationarity tolerance
    eta_k: f64,     // current feasibility tolerance
    mu_k: f64,      // current max penalty (mirror of penalties for scheduling)
}
```

Add near the other module consts (line ~35):
```rust
// Schedule exponents (LANCELOT-style). alpha_* control how fast eta/omega tighten.
const ETA_TIGHTEN_EXP: f64 = 0.9;    // eta_{k+1} = max(eta_*, eta_k / mu^ETA_TIGHTEN_EXP) on success
const OMEGA_TIGHTEN_EXP: f64 = 1.0;  // omega_{k+1} = max(omega_*, omega_k / mu^OMEGA_TIGHTEN_EXP)
const ETA_RESET_EXP: f64 = 0.1;      // eta after penalty increase = max(eta_*, eta0 / mu^ETA_RESET_EXP)
const OMEGA_INIT: f64 = 1.0;         // ω_0
const ETA_INIT: f64 = 1.0;           // η_0 (scaled by 1/mu_0 below)
```

Initialize in the constructor (find `AugmentedLagrangian::new`, wherever `Self { fcn, x_init, bounds, opts }` is built) using `opts.initial_penalty`:
```rust
let mu0 = opts.initial_penalty;
Self {
    fcn,
    x_init,
    bounds,
    opts,
    mu_k: mu0,
    omega_k: OMEGA_INIT / mu0,
    eta_k: (ETA_INIT / mu0.powf(ETA_RESET_EXP))
        .max(/* user eta_* */ /* set below after opts moved */ 0.0),
}
```
Because `opts` is moved into `Self`, compute the user targets *before* the move:
```rust
let user_eta = opts.constrained_opts.constraint_tol;
let user_omega = opts.constrained_opts.base_opts.grad_atol;
let mu0 = opts.initial_penalty;
let omega_0 = (OMEGA_INIT / mu0).max(user_omega);
let eta_0 = (ETA_INIT / mu0.powf(ETA_RESET_EXP)).max(user_eta);
Self { fcn, x_init, bounds, opts, mu_k: mu0, omega_k: omega_0, eta_k: eta_0 }
```

### 2.2 — Replace `set_inner_tolerances`

Replace the whole body (lines ~951–1061) with a version that (a) drops the hard `rtol = 0.0` + clamp, (b) hands the inner solver `ω_k` as its atol, and (c) sets inner rtol to 0 *only* because we now drive convergence by the absolute `ω_k` (which itself floors at the user rtol-equivalent). Keep the unconstrained-passthrough branch (when `!fcn.is_constrained()`), but for the constrained branch use `self.omega_k`:

```rust
#[trace_fn]
fn set_inner_tolerances(&self) -> InnerMethod
{
    let inner_method = self.opts.inner_method.clone();
    match inner_method
    {
        InnerMethod::Unconstrained(mut uncon_method) =>
        {
            self.fcn.lock().unwrap().with_inner_mut(|fcn| {
                if !fcn.is_constrained()
                {
                    uncon_method.uncon_opts_mut().grad_rtol =
                        self.opts.constrained_opts.base_opts.grad_rtol;
                    uncon_method.uncon_opts_mut().grad_atol =
                        self.opts.constrained_opts.base_opts.grad_atol;
                }
                else
                {
                    // Drive the Lᴀ subproblem to ω_k in the sup-norm.
                    uncon_method.uncon_opts_mut().grad_rtol = 0.0;
                    uncon_method.uncon_opts_mut().grad_atol = self.omega_k;
                    info!(target: "aug", "Inner ω_k = {:.4e}", self.omega_k);
                }
            });
            InnerMethod::Unconstrained(uncon_method)
        }
        InnerMethod::BoundConstrained(mut bcon_method) =>
        {
            self.fcn.lock().unwrap().with_inner_mut(|fcn| {
                if !fcn.is_constrained()
                {
                    bcon_method.bound_opts_mut().base_opts.grad_rtol =
                        self.opts.constrained_opts.base_opts.grad_rtol;
                    bcon_method.bound_opts_mut().base_opts.grad_atol =
                        self.opts.constrained_opts.base_opts.grad_atol;
                }
                else
                {
                    bcon_method.bound_opts_mut().base_opts.grad_rtol = 0.0;
                    bcon_method.bound_opts_mut().base_opts.grad_atol = self.omega_k;
                    info!(target: "aug", "Inner ω_k = {:.4e}", self.omega_k);
                }
            });
            InnerMethod::BoundConstrained(bcon_method)
        }
    }
}
```

> This requires the inner bounded/unbounded stopping test to be sup-norm — which Phase 1 guaranteed. That is the whole reason Phase 1 comes first. The inner solver now stops when `‖P[∇Lᴀ]‖_∞ ≤ ω_k`, i.e. the tolerance is interpreted in the penalized-Lᴀ metric, exactly as it should be.

### 2.3 — Add the schedule update, driven by feasibility vs `η_k`

Add a method that updates `(η_k, ω_k, μ_k)` and decides multiplier-update vs penalty-increase. This *replaces* the unconditional `update_lagrangian(alpha, beta, ret)` call in `minimize`. Read the current infeasibility with the same sup-norm the convergence test uses.

```rust
/// Returns the post-update iterate. Implements the LANCELOT/Birgin–Martínez
/// outer test: if feasible enough (‖c‖∞ ≤ η_k) accept multipliers and tighten;
/// else increase penalty and reset η, ω relative to the new μ.
#[trace_fn]
fn update_schedule_and_lagrangian(
    &mut self,
    uncon_ret: Returns,
) -> IterData
{
    let user_eta = self.opts.constrained_opts.constraint_tol;
    let user_omega = self.opts.constrained_opts.base_opts.grad_atol;
    let beta = self.opts.penalty_growth_factor;

    // Current infeasibility in sup-norm (plain constraint values, posed for ieq).
    let residual_primal = self.fcn.lock().unwrap().with_inner_mut(|fcn| {
        let norm_eq = fcn.eq_penalty.as_ref()
            .map(|p| p.data.values.abs_max().unwrap()).unwrap_or(0.0);
        let norm_ieq = fcn.ieq_penalty.as_ref()
            .map(|p| p.data.values.posed().abs_max().unwrap()).unwrap_or(0.0);
        norm_eq.max(norm_ieq)
    });

    if residual_primal <= self.eta_k
    {
        // Feasibility improved enough: update multipliers (shift += c), keep μ,
        // and tighten both tolerances toward the user targets.
        info!(target: "aug", "Feasible (‖c‖∞={:.4e} ≤ η_k={:.4e}): multiplier update",
              residual_primal, self.eta_k);
        self.update_lagrangian_multipliers_only();
        self.eta_k   = (self.eta_k   / self.mu_k.powf(ETA_TIGHTEN_EXP)).max(user_eta);
        self.omega_k = (self.omega_k / self.mu_k.powf(OMEGA_TIGHTEN_EXP)).max(user_omega);
    }
    else
    {
        // Not feasible enough: increase penalty, reset tolerances relative to new μ.
        info!(target: "aug", "Infeasible (‖c‖∞={:.4e} > η_k={:.4e}): penalty increase",
              residual_primal, self.eta_k);
        self.increase_penalty(beta);
        self.mu_k *= beta;
        self.eta_k   = (ETA_INIT   / self.mu_k.powf(ETA_RESET_EXP)).max(user_eta);
        self.omega_k = (OMEGA_INIT / self.mu_k).max(user_omega);
    }

    IterData::new(self.fcn.clone(), &uncon_ret.xmin)
}
```

You now need two helpers that split the current `update_lagrangian` behavior. The existing `EqPenalty::update_penalties_shifts` / `IeqPenalty::update_penalties_shifts` (lines ~239–262 and the ieq analog) currently do **both** the shift update *and* the penalty increase inside one loop keyed on `was_violated`. Split them:

- `update_lagrangian_multipliers_only`: apply only the `shift += value` branch (the `else` arm at line ~258–259), i.e. the first-order multiplier update `λ ← λ + μ c`, for **all** constraints, no penalty change.
- `increase_penalty(beta)`: multiply `penalties *= beta` (and `shifts /= beta` to preserve `λ = penalty*shift`, matching the existing convention at lines ~253–254) for all constraints.

Suggested penalty-struct methods (add to both `EqPenalty` and `IeqPenalty`):
```rust
fn update_multipliers_only(&mut self) {
    for i in 0..self.data.values.len() {
        self.data.shifts[i] += self.data.values[i]; // λ_i ← λ_i + μ_i c_i  (λ=penalty*shift)
    }
}
fn increase_penalty(&mut self, beta: f64) {
    for i in 0..self.data.penalties.len() {
        self.data.penalties[i] *= beta;
        self.data.shifts[i]    /= beta; // keep λ = penalty*shift invariant
    }
}
```
And on the AL fcn, thin wrappers that call the above on whichever penalties exist (mirror the `with_inner_mut` pattern already used in `update_lagrangian`).

> Design note (matches your project memory + Birgin §4): the multiplier update uses `λ_{k+1} = λ_k + μ_k c(x_k)`. Keeping the `λ = penalty * shift` factoring means "shift += c" *is* the first-order update. Verify this equivalence holds given `compute_lagrange_multiplier_estimates = penalties * shifts` (line ~208) before trusting the trace. If your penalty form is PHR-shifted rather than classic, adjust the `+= c` accordingly, but keep the two operations (multiplier update vs penalty increase) mutually exclusive per outer iteration — conflating them is the current bug.

### 2.4 — Wire into `minimize`

In `fn minimize` (line ~1088), replace the update call:

```rust
// BEFORE
iter_k = self.update_lagrangian(alpha, beta, ret);
iter_prev_k.copy_from(&iter_k);

// AFTER
iter_k = self.update_schedule_and_lagrangian(ret);
iter_prev_k.copy_from(&iter_k);
```

`alpha` (`constraint_improvement_factor`) is no longer used to *decide* the branch (η_k does). You may keep `alpha` for an alternative "sufficient decrease in ‖c‖ vs previous outer iterate" gate if you prefer that flavor over the absolute η_k test; if you drop it, remove the now-unused `let alpha = ...` binding to avoid warnings.

### 2.5 — Leave `is_converged` almost as-is (it's already correct)

The outer test (lines ~809–884) already rebuilds a `Lagrangian`-type function, uses updated multipliers, projects the gradient, and measures in `abs_max` (L∞). Keep it. One consistency tweak — the final trace at line ~1114 logs `iter_k.norm_grad_fx` (raw Lᴀ gradient, now L∞ after Phase 1 but still the *penalized* gradient, not the projected plain-L residual the test used). Replace with the actual test quantity for honest diagnostics: recompute and log `residual_stationarity` (the projected plain-Lagrangian sup-norm) instead of `iter_k.norm_grad_fx`, or drop that specific line.

### 2.6 — Gate

```
cargo build && cargo test --test augmented_lagrangian 2>&1 | tee /tmp/al_after.txt
diff /tmp/baseline_al.txt /tmp/al_after.txt
```

Expect previously-flaky problems to now converge with the same user settings. If a problem regresses, first check the trace for `η_k`/`ω_k`/`μ_k` monotonicity and whether `ω_k` actually reaches `user_omega`.

---

## Phase 3 — Verification against references

1. Cross-check a bound-constrained AL problem against the scipy/Algencan-style reference you already use in `tests/`. Confirm final `‖P[∇L]‖_∞ ≤ grad_atol` **and** `‖c‖_∞ ≤ constraint_tol` both hold at the reported solution.
2. Add a regression test that specifically sets `grad_atol = 1e-8` (tighter than the old `1e-6` clamp floor) on a problem with a known solution, and asserts the achieved projected-Lagrangian residual is `≤ 1e-8`. This test would have failed under the old clamp and guards against reintroducing it.
3. Add a large-`n` bound-constrained unconstrained-inner problem (e.g. n = 500) and confirm ASA converges in comparable iterations to small-n — this guards against the dimension-dependent-rtol regression fixed in 1.1.

---

## Order-of-operations checklist

- [ ] Phase 0 baselines saved
- [ ] 1.1 ASA line ~227 → `abs_max`; line ~442 diagnostic → `abs_max`; μ-test norms untouched
- [ ] 1.3 `IterData::new` → `abs_max`; ASA lines ~350, ~660 → `abs_max`
- [ ] 1.2 QN/CG `new()` reference norm → `abs_max`
- [ ] 1.4 re-baseline QN/CG/ASA tests (tighten tol inputs, not accuracy asserts)
- [ ] 1.5 gate green
- [ ] 2.1 struct fields + consts + constructor init
- [ ] 2.2 replace `set_inner_tolerances`
- [ ] 2.3 add `update_schedule_and_lagrangian` + split penalty helpers
- [ ] 2.4 wire into `minimize`
- [ ] 2.5 fix trace at ~1114
- [ ] 2.6 gate green
- [ ] Phase 3 reference cross-checks + two new regression tests

## Invariants to assert while editing (cheap guards)
- `λ = penalties * shifts` holds after both `update_multipliers_only` and `increase_penalty`.
- `omega_k` and `eta_k` are non-increasing across outer iterations *within a penalty-constant run*, and both `≥` their user floors always.
- After a penalty increase, `mu_k` strictly increased and `omega_k/eta_k` were reset relative to the new `mu_k`.
- Inner solver's effective stopping norm is L∞ (Phase 1) so that handing it `omega_k` means what the schedule intends.
