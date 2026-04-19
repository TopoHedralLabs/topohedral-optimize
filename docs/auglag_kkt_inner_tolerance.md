# Augmented Lagrangian KKT Residual and Inner Tolerance Schedule

This note summarizes the KKT residual for the current augmented Lagrangian setup and
sketches a replacement inner tolerance schedule that fits the existing code structure in
`src/constrained/augmented_lagrangian.rs`.

## Problem setup

Assume the constrained problem is

$$
\min_x f(x)
$$

subject to

$$
h(x) = 0, \qquad g(x) \le 0.
$$

For the current implementation, the augmented Lagrangian pieces are

$$
\nabla P_{\mathrm{eq}}(x)
= \sum_i p_i \left(h_i(x) + \theta_i\right)\nabla h_i(x),
$$

and

$$
\nabla P_{\mathrm{ieq}}(x)
= \sum_i q_i \max\left(g_i(x) + \phi_i, 0\right)\nabla g_i(x).
$$

The corresponding multiplier estimates are

$$
\lambda_i = p_i \theta_i, \qquad \mu_i = q_i \phi_i.
$$

## KKT residual in this context

The original nonlinear programming KKT conditions are

$$
\nabla f(x) + J_h(x)^T \lambda + J_g(x)^T \mu = 0,
$$

$$
h(x) = 0, \qquad g(x) \le 0,
$$

$$
\mu \ge 0,
$$

$$
\mu_i g_i(x) = 0 \quad \forall i.
$$

A practical infinity-norm residual is

$$
r_{\mathrm{stat}}
= \left\|\nabla f(x) + J_h(x)^T \lambda + J_g(x)^T \mu\right\|_\infty,
$$

$$
r_{\mathrm{prim}}
= \max\left(\|h(x)\|_\infty,\ \|g_+(x)\|_\infty\right),
\qquad
g_+(x)_i = \max(g_i(x), 0),
$$

$$
r_{\mathrm{dual}}
= \|\min(\mu, 0)\|_\infty,
$$

$$
r_{\mathrm{comp}}
= \|\mu \odot g(x)\|_\infty,
$$

and then

$$
r_{\mathrm{kkt}}
= \max\left(
r_{\mathrm{stat}},
r_{\mathrm{prim}},
r_{\mathrm{dual}},
r_{\mathrm{comp}}
\right).
$$

In the current code, inequality shifts are projected with `max(0, old_shift_i + g_i)`, so
the estimated multipliers satisfy `mu >= 0` by construction. In that case, the useful
reduced residual is

$$
r_{\mathrm{kkt}}
= \max\left(
\left\|\nabla f(x) + J_h(x)^T \lambda + J_g(x)^T \mu\right\|_\infty,
\|h(x)\|_\infty,
\|g_+(x)\|_\infty,
\|\mu \odot g(x)\|_\infty
\right).
$$

## Why the current inner schedule is awkward

The current inner tolerance is based on

$$
r_k = \max\left(K_{\max, k}, \frac{1}{\rho_k}\right),
$$

with `rho_k = max_penalty_k`, followed by

$$
\texttt{grad\_rtol}_k = \mathrm{clamp}(0.01 r_k, 10^{-5}, 10^{-3}).
$$

That schedule has two problems:

1. It drives the inner solver with a relative reduction target, not an absolute stationarity
   target for the subproblem.
2. It uses `Kmax`, which for inequalities is currently closer to an activity measure than to
   pure primal infeasibility.

Near the solution, `||grad_f||` and the penalty gradient can nearly cancel, so the augmented
Lagrangian gradient can look small even when the primal and complementarity residuals are not
yet at the desired KKT level.

## Recommended replacement: use an absolute inner tolerance

Keep the outer loop logic mostly as-is, but drive the inner solve with an absolute target

$$
\|\nabla_x \mathcal{L}_A(x; \lambda_k, \mu_k, \rho_k)\| \le \eta_k,
$$

instead of relying mainly on the inner solver's relative reduction.

### Step 1: define a pure primal infeasibility measure

For the outer schedule, use

$$
r_{\mathrm{prim}, k}
= \max\left(
\|h(x_k)\|_\infty,\ \|g_+(x_k)\|_\infty
\right).
$$

For inequalities this should use `max(g_i, 0)`, not `max(g_i, -phi_i)`.

### Step 2: combine feasibility and penalty scale

Define the outer forcing quantity

$$
s_k = \max\left(r_{\mathrm{prim}, k}, \frac{1}{\rho_k}\right).
$$

This preserves the useful idea in your current code:

- when constraints are still bad, `r_prim` dominates,
- when feasibility is already good, `1 / rho_k` keeps tightening the inner solve as the
  penalty grows.

### Step 3: choose a concrete forcing schedule

A simple practical choice is

$$
\eta_k
= \mathrm{clamp}\left(\eta_{\min}, \eta_{\max}, c_\eta s_k^{3/2}\right).
$$

Good starting values are

$$
\eta_{\max} = 10^{-2}, \qquad
\eta_{\min} = 10^{-10}, \qquad
c_\eta = 0.1.
$$

This behaves like:

- if `s_k = 1`, then `eta_k = 1e-2`,
- if `s_k = 1e-1`, then `eta_k \approx 3e-3`,
- if `s_k = 1e-2`, then `eta_k = 1e-4`,
- if `s_k = 1e-3`, then `eta_k \approx 3e-6`.

That is usually a better progression than a fixed relative target capped at `1e-3`.

## How this fits the current code

### 1. Add a pure primal violation helper

In `AugmentedLagrangianFcn`, add a helper separate from `compute_max_constraint_violation()`:

```rust
fn compute_max_primal_violation(&self) -> f64 {
    let mut max_violation = 0.0;

    if let Some(eq_constraint_data) = &self.eq_constraint_data {
        max_violation = max_violation.max(eq_constraint_data.values.abs_max().unwrap());
    }

    if let Some(ieq_constraint_data) = &self.ieq_constraint_data {
        let max_ieq = ieq_constraint_data
            .values
            .iter()
            .map(|&g_i| g_i.max(0.0))
            .reduce(f64::max)
            .unwrap_or(0.0);
        max_violation = max_violation.max(max_ieq);
    }

    max_violation
}
```

Use this for the inner forcing schedule. Keep the existing activity-aware quantity if you still
want it for penalty-update logic or diagnostics.

### 2. Replace the inner tolerance logic in `set_uncon_options`

Conceptually:

```rust
const ETA_MAX: f64 = 1e-2;
const ETA_MIN: f64 = 1e-10;
const ETA_SCALE: f64 = 1e-1;
const INNER_RTOL_FLOOR: f64 = 1e-12;

fn set_uncon_options(&mut self, _max_violation_k: f64) {
    let mut counting_fcn = self.fcn.lock().unwrap();
    let auglag_fcn = counting_fcn.inner_mut();

    let rho_k = auglag_fcn.compute_max_penalty();
    let r_prim_k = auglag_fcn.compute_max_primal_violation();
    let s_k = r_prim_k.max(1.0 / rho_k.max(1.0));
    let eta_k = (ETA_SCALE * s_k.powf(1.5)).clamp(ETA_MIN, ETA_MAX);

    self.opts.uncon_method.uncon_opts_mut().grad_atol = eta_k;
    self.opts.uncon_method.uncon_opts_mut().grad_rtol = INNER_RTOL_FLOOR;

    self.opts.uncon_method.uncon_opts_mut().max_iter = if eta_k > 1e-3 {
        50
    } else if eta_k > 1e-5 {
        100
    } else {
        200
    };
}
```

### 3. Why set `grad_atol`, not `grad_rtol`

Your inner CG solver stops when either

$$
\frac{\|\nabla f_k\|}{\|\nabla f_0\|} < \texttt{grad\_rtol}
$$

or

$$
\|\nabla f_k\| < \texttt{grad\_atol}.
$$

For an inexact augmented Lagrangian solve, the quantity that matters is much closer to an
absolute stationarity target for the current subproblem, so `grad_atol = eta_k` is the more
natural control knob.

If `grad_rtol` remains loose, the inner solve can stop after a modest relative reduction even
though `||grad L_A||` is still much too large in absolute terms. Setting `grad_rtol` to a tiny
value effectively makes `grad_atol = eta_k` the active stopping rule.

## Optional upgrade: track an actual outer KKT residual

If you want the outer convergence check to align more closely with the NLP, add a diagnostic
helper that computes

$$
r_{\mathrm{kkt}, k}
= \max\left(
\left\|\nabla f(x_k) + J_h(x_k)^T \lambda_k + J_g(x_k)^T \mu_k\right\|_\infty,
\|h(x_k)\|_\infty,
\|g_+(x_k)\|_\infty,
\|\mu_k \odot g(x_k)\|_\infty
\right).
$$

Then log it next to your current gradient-split diagnostics. Even if you do not yet use it as
the formal stop test, it will tell you whether the outer loop is progressing in the quantity
you actually care about.

## Suggested first pass

If you want a minimal change set:

1. Add `compute_max_primal_violation()`.
2. Replace the current `grad_rtol_k` schedule with

$$
\eta_k = \mathrm{clamp}(10^{-10}, 10^{-2}, 0.1\, s_k^{3/2}),
\qquad
s_k = \max\left(r_{\mathrm{prim}, k}, \frac{1}{\rho_k}\right).
$$

3. Set inner `grad_atol = eta_k`.
4. Set inner `grad_rtol = 1e-12`.

That keeps your existing outer algorithm intact while making the inner stopping rule much
closer to the intended inexact AL behavior.

## Suggested outer `is_converged()` based on a KKT residual

If you want the outer stopping rule to align with the original constrained problem, the clean
option is to stop on a KKT-style residual instead of the current normalized augmented
Lagrangian gradient plus `Kmax`.

### Residual components

Using the multiplier estimates

$$
\lambda_i = p_i \theta_i, \qquad \mu_i = q_i \phi_i,
$$

define the stationarity residual

$$
r_{\mathrm{stat}}
= \left\|\nabla f(x) + J_h(x)^T \lambda + J_g(x)^T \mu \right\|_\infty.
$$

Define the primal feasibility residual

$$
r_{\mathrm{prim}}
= \max\left(\|h(x)\|_\infty,\ \|g_+(x)\|_\infty\right),
\qquad
g_+(x)_i = \max(g_i(x), 0).
$$

Define the complementarity residual

$$
r_{\mathrm{comp}}
= \|\mu \odot g(x)\|_\infty.
$$

Since the current inequality shift update projects with `max(0, old_shift_i + g_i)`, the dual
feasibility condition `\mu \ge 0` is enforced by construction. So the reduced KKT residual is

$$
r_{\mathrm{kkt}}
= \max\left(r_{\mathrm{stat}}, r_{\mathrm{prim}}, r_{\mathrm{comp}}\right).
$$

If you want a scaled version for logging, a useful normalized stationarity quantity is

$$
r_{\mathrm{stat,scaled}}
= \frac{
\left\|\nabla f(x) + J_h(x)^T \lambda + J_g(x)^T \mu \right\|_\infty
}{
\max\left(1,\ \|\nabla f(x)\|_\infty,\ \|J_h(x)^T \lambda\|_\infty,\ \|J_g(x)^T \mu\|_\infty\right)
}.
$$

Then a practical stopping test is

$$
r_{\mathrm{prim}} \le c_{\mathrm{tol}}
\quad \text{and} \quad
r_{\mathrm{comp}} \le c_{\mathrm{tol}}
\quad \text{and either} \quad
r_{\mathrm{stat}} \le a_{\mathrm{tol}}
\quad \text{or} \quad
r_{\mathrm{stat,scaled}} \le r_{\mathrm{tol}}.
$$

This mirrors your current `Atol` / `Rtol` interface while using quantities that are tied to the
original NLP.

### Suggested helper struct

You could introduce a diagnostic container like:

```rust
#[derive(Debug, Copy, Clone)]
struct KktResidual {
    stat_inf: f64,
    stat_scaled: f64,
    prim_inf: f64,
    comp_inf: f64,
    kkt_inf: f64,
}
```

### Suggested helper implementation

This sketch is written to fit the current `AugmentedLagrangianFcn` structure and naming. It is
not meant to compile unchanged, but it is close to what I would actually implement.

```rust
fn compute_kkt_residual(&self) -> KktResidual {
    let mut stat_vec = self.cached_value.fcn_grad.clone();
    let mut eq_term_inf: f64 = 0.0;
    let mut ieq_term_inf: f64 = 0.0;
    let mut prim_eq_inf: f64 = 0.0;
    let mut prim_ieq_inf: f64 = 0.0;
    let mut comp_inf: f64 = 0.0;

    if let Some(eq_constraint_data) = &self.eq_constraint_data {
        prim_eq_inf = eq_constraint_data.values.abs_max().unwrap_or(0.0);

        for i in 0..eq_constraint_data.values.len() {
            let p_i = eq_constraint_data.penalties[i];
            let theta_i = eq_constraint_data.shifts[i];
            let lambda_i = p_i * theta_i;
            let grad_h_i = eq_constraint_data.gradients.col(i).to_dmatrix();
            let term_i = lambda_i * grad_h_i;
            eq_term_inf = eq_term_inf.max(term_i.abs_max().unwrap_or(0.0));
            stat_vec += term_i;
        }
    }

    if let Some(ieq_constraint_data) = &self.ieq_constraint_data {
        for i in 0..ieq_constraint_data.values.len() {
            let q_i = ieq_constraint_data.penalties[i];
            let phi_i = ieq_constraint_data.shifts[i];
            let g_i = ieq_constraint_data.values[i];
            let mu_i = q_i * phi_i;

            prim_ieq_inf = prim_ieq_inf.max(g_i.max(0.0));
            comp_inf = comp_inf.max((mu_i * g_i).abs());

            let grad_g_i = ieq_constraint_data.gradients.col(i).to_dmatrix();
            let term_i = mu_i * grad_g_i;
            ieq_term_inf = ieq_term_inf.max(term_i.abs_max().unwrap_or(0.0));
            stat_vec += term_i;
        }
    }

    let stat_inf = stat_vec.abs_max().unwrap_or(0.0);
    let grad_f_inf = self.cached_value.fcn_grad.abs_max().unwrap_or(0.0);
    let prim_inf = prim_eq_inf.max(prim_ieq_inf);

    let scale = 1.0_f64.max(grad_f_inf).max(eq_term_inf).max(ieq_term_inf);
    let stat_scaled = stat_inf / scale;
    let kkt_inf = stat_inf.max(prim_inf).max(comp_inf);

    KktResidual {
        stat_inf,
        stat_scaled,
        prim_inf,
        comp_inf,
        kkt_inf,
    }
}
```

### Important note about gradients and cached values

This helper assumes:

1. `self.cached_value.fcn_grad` is up to date at the current `x`.
2. `eq_constraint_data.values`, `ieq_constraint_data.values`,
   `eq_constraint_data.gradients`, and `ieq_constraint_data.gradients` are also current at the
   same `x`.

In practice I would make sure `is_converged()` is only called after a fresh `self.fcn.grad(&x)`
and after the constraint values have also been evaluated at that same `x`.

## Rust sketch for `is_converged()`

The following shape keeps your existing `ConvergedReason::{Rtol, Atol}` and current option
layout:

```rust
#[trace_fn]
fn is_converged(&self) -> Option<ConvergedReason> {
    let mut counting_fcn = self.fcn.lock().unwrap();
    let auglag_fcn = counting_fcn.inner_mut();
    let kkt = auglag_fcn.compute_kkt_residual();

    let rtol = self.opts.constrained_opts.grad_rtol;
    let atol = self.opts.constrained_opts.grad_atol;
    let ctol = self.opts.constrained_opts.constraint_tol;

    let rtol_converged = kkt.stat_scaled < rtol;
    let atol_converged = kkt.stat_inf < atol;
    let prim_converged = kkt.prim_inf < ctol;
    let comp_converged = kkt.comp_inf < ctol;

    trace!(target: "aug", "KKT stat_inf = {:1.4e}", kkt.stat_inf);
    trace!(target: "aug", "KKT stat_scaled = {:1.4e}", kkt.stat_scaled);
    trace!(target: "aug", "KKT prim_inf = {:1.4e}", kkt.prim_inf);
    trace!(target: "aug", "KKT comp_inf = {:1.4e}", kkt.comp_inf);
    trace!(target: "aug", "KKT kkt_inf = {:1.4e}", kkt.kkt_inf);
    trace!(target: "aug", "rtol_converged = {rtol_converged}");
    trace!(target: "aug", "atol_converged = {atol_converged}");
    trace!(target: "aug", "prim_converged = {prim_converged}");
    trace!(target: "aug", "comp_converged = {comp_converged}");

    if prim_converged && comp_converged && rtol_converged {
        return Some(ConvergedReason::Rtol);
    }

    if prim_converged && comp_converged && atol_converged {
        return Some(ConvergedReason::Atol);
    }

    None
}
```

### Minimal-change variant

If you want to be a little less strict on the first pass, you could omit `comp_converged` and
use

```rust
if prim_converged && rtol_converged {
    return Some(ConvergedReason::Rtol);
}

if prim_converged && atol_converged {
    return Some(ConvergedReason::Atol);
}
```

That still improves the current logic because:

1. stationarity is measured against the original KKT system, not just the AL gradient split,
2. primal feasibility is measured with `g_+`, not an activity-like surrogate.

## Recommendation

My preference would be:

1. Use the full `stat + prim + comp` residual for diagnostics immediately.
2. Use `prim + stat` for stopping first if you want a low-risk transition.
3. Add the complementarity gate once the rest of the AL behavior is stable.

That gives you a much more interpretable convergence test without forcing a large refactor all
at once.

## Appendix: Compact `is_converged()` sketch

This appendix is a shorter version of the outer convergence check, included separately so it is
easy to find at the end of the file.

```rust
#[trace_fn]
fn is_converged(&self) -> Option<ConvergedReason> {
    let mut counting_fcn = self.fcn.lock().unwrap();
    let auglag_fcn = counting_fcn.inner_mut();
    let kkt = auglag_fcn.compute_kkt_residual();

    let rtol = self.opts.constrained_opts.grad_rtol;
    let atol = self.opts.constrained_opts.grad_atol;
    let ctol = self.opts.constrained_opts.constraint_tol;

    let prim_converged = kkt.prim_inf < ctol;
    let comp_converged = kkt.comp_inf < ctol;
    let rtol_converged = kkt.stat_scaled < rtol;
    let atol_converged = kkt.stat_inf < atol;

    if prim_converged && comp_converged && rtol_converged {
        return Some(ConvergedReason::Rtol);
    }

    if prim_converged && comp_converged && atol_converged {
        return Some(ConvergedReason::Atol);
    }

    None
}
```

The intended residuals are

$$
r_{\mathrm{stat}}
= \left\|\nabla f(x) + J_h(x)^T \lambda + J_g(x)^T \mu \right\|_\infty,
$$

$$
r_{\mathrm{prim}}
= \max\left(\|h(x)\|_\infty,\ \|g_+(x)\|_\infty\right),
$$

$$
r_{\mathrm{comp}}
= \|\mu \odot g(x)\|_\infty.
$$
