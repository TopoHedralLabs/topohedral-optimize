# Inner Tolerance Schedule for Augmented Lagrangian Solves

## Goal

Use loose inner solves when the current iterate is still far from feasible, then tighten the inner solve tolerance as feasibility improves and as penalties grow. A fixed inner relative tolerance is usually too blunt for augmented Lagrangian methods.

In the current implementation, this matters especially because the inner conjugate-gradient solver resets its relative gradient reference at the start of each outer iteration. That means a fixed `grad_rtol` does not correspond to a fixed absolute stationarity requirement for the augmented Lagrangian.

## Recommended Schedule

Let:

- `K_k` = current maximum constraint violation
- `P_k` = `max(max_i p_i, max_i q_i)` = largest current penalty
- `||g0_k||` = norm of the augmented-Lagrangian gradient at the start of the inner solve

Define

```text
r_k = max(K_k, 1 / max(P_k, 1))
eta_k = clamp(1e-10, 1e-2, 0.1 * r_k^2)
```

Interpretation:

- When constraints are badly violated, `K_k` is large, so `eta_k` stays loose.
- As feasibility improves, `eta_k` shrinks quickly.
- If penalties grow while feasibility stalls, the `1 / P_k` term still forces gradual tightening.

## How to Set the Inner Solver Tolerances

Use `eta_k` as the absolute target for the inner augmented-Lagrangian gradient norm.

If the inner solver supports both absolute and relative tolerances:

```text
grad_atol_k = eta_k
grad_rtol_k = clamp(1e-8, 5e-2, eta_k / max(||g0_k||, 1.0))
```

This makes the relative tolerance reflect a meaningful absolute stationarity target instead of being a fixed percentage of the current AL gradient.

## Suggested Iteration Budget

The maximum inner iteration count should also tighten with the tolerance:

```text
if eta_k > 1e-3:
    max_iter = 20
elif eta_k > 1e-6:
    max_iter = 50
else:
    max_iter = 100
```

This avoids oversolving early subproblems while giving later stiff AL subproblems enough room to converge.

## Extra Safeguards

Two simple safeguards are worth using:

### 1. Make the schedule monotone

Do not allow the inner tolerance to loosen if the outer iteration temporarily becomes worse:

```text
eta_k = min(eta_k, eta_{k-1})
```

### 2. Tighten after a penalty increase

If penalties were increased this outer iteration, tighten the next inner target more aggressively:

```text
eta_k = min(eta_k, 0.1 * eta_{k-1})
```

This is useful because once penalties increase, the inner subproblem becomes stiffer and a previously acceptable loose solve may no longer be good enough.

## Simplest Version

If you want a simpler first pass, use only the current constraint violation:

```text
eta_k = clamp(1e-10, 1e-2, 0.1 * K_k^2)
```

Then set:

```text
grad_atol_k = eta_k
grad_rtol_k = clamp(1e-8, 5e-2, eta_k / max(||g0_k||, 1.0))
```

That already gives a much better schedule than a fixed `1e-2` relative tolerance.

## Practical Rationale

The outer augmented Lagrangian method only needs crude inner minimization while feasibility is poor. Near the solution, however, the penalty terms dominate local conditioning and the subproblems need to be solved much more accurately. A schedule based on both feasibility and penalty size captures that behavior well.

In practice:

- early outer iterations should be cheap
- late outer iterations should be accurate
- penalty increases should trigger tighter inner solves
- the inner stopping rule should correspond to an absolute stationarity goal, not just a relative reduction from an arbitrary starting AL gradient

## Pseudocode

```rust
let grad0 = current_aug_lag_grad_norm;
let pmax = max_penalty;
let rk = kmax.max(1.0 / pmax.max(1.0));
let mut eta = (0.1 * rk * rk).clamp(1e-10, 1e-2);

if let Some(prev_eta) = prev_eta {
    eta = eta.min(prev_eta);
}

if penalties_increased {
    if let Some(prev_eta) = prev_eta {
        eta = eta.min(0.1 * prev_eta);
    }
}

opts.grad_atol = eta;
opts.grad_rtol = (eta / grad0.max(1.0)).clamp(1e-8, 5e-2);
opts.max_iter = if eta > 1e-3 {
    20
} else if eta > 1e-6 {
    50
} else {
    100
};
```

## Suggested First Implementation

For this codebase, a good first implementation would be:

1. Compute `K_k` after each outer iteration.
2. Track the current maximum penalty.
3. Replace the fixed inner tolerance with the `eta_k` schedule above.
4. Set both `grad_atol` and `grad_rtol`.
5. Increase inner `max_iter` as `eta_k` shrinks.

That should make the outer AL loop much less likely to stagnate near the constrained optimum.
