# User Guide

This guide gives an overview of the public API. The generated [API Docs](api/)
remain the reference for every option field and method.

## Imports and function traits

The commonly used types are re-exported from the crate root:

| Area | Main exports |
|---|---|
| Function interfaces | `DifferentiableFn`, `RealFn1`, `RealFn`, `RealVectorFn` |
| Data | `Vector`, `Matrix`, `IterData`, `ScalarReturns`, `VectorReturns` |
| Stopping | `BaseOptions`, `ConvergedReason` |
| Scalar | `ScalarMethod`, `BoundedOptions`, `BrentOptions`, `GoldenOptions` |
| Line search | `LineSearchMethod`, `ThuenteOptions`, `NocedalOptions` |
| Vector optimization | `UnconstrainedMethod`, `BoundConstrainedMethod`, `ConstrainedMethod` |
| Constraints | `BoundsConstraints`, `BoundStatus`, `NoConstraints` |

`RealFn1` represents a differentiable `f64 -> f64` function. `RealFn` represents
a differentiable `Vector -> f64` objective. Vector constraints implement
`RealVectorFn`, with a vector value and Jacobian matrix.

## Choosing a method

| Problem | Entry point | Available methods |
|---|---|---|
| Scalar, finite interval | `scalar_minimze` | `ScalarMethod::Bounded` |
| Scalar, local bracket | `scalar_minimze` | `ScalarMethod::Brent`, `Golden` |
| Step along a vector direction | `lsearch` | `Thuente`, `Nocedal` |
| Unconstrained vector objective | `unconstrained_minimize` | conjugate gradient, quasi-Newton |
| Vector objective with box bounds | `bound_constrained_minimize` | ASA, L-BFGS-B |
| General vector constraints | `constrained_minimize` | augmented Lagrangian |

## Scalar minimization

For a known finite interval, create `BoundedOptions` with `BoundedOptions::new`:

```rust
let options = topohedral_optimize::BoundedOptions::new(-2.0, 3.0)?;
let result = topohedral_optimize::scalar_minimze(
    &mut objective,
    topohedral_optimize::ScalarMethod::Bounded(options),
)?;
```

Brent and golden-section search use a `Bracket`. A bracket is an ordered triple
`(xa, xb, xc)` where `xb` has a lower function value than both endpoints. Use
`bracket` to grow a bracket from two initial points, or construct the documented
`Bracket` variant for an existing triple.

## Unconstrained minimization

`UnconstrainedMethod` selects the algorithm and owns its algorithm-specific
options. Conjugate gradient supports `Steepest`, `FletcherReeves`, and
`PolakRibiere` directions. Quasi-Newton supports `BFGS` and `DFP` update choices;
the BFGS path is the recommended first choice in the current implementation.

Both methods use a line search and stop when the infinity norm of the gradient
meets either `grad_rtol` relative to its initial value or `grad_atol`, or when
`max_iter` is reached.

## Bound constraints

Bounds are sparse: only variables with a bound need to be added.

```rust
let mut bounds = topohedral_optimize::BoundsConstraints::new(2);
bounds.add_bounds(0, Some(-1.0), Some(1.0));
bounds.add_bounds(1, Some(0.0), None);
```

Pass the bounds and an initial vector to `bound_constrained_minimize`. ASA uses
an unconstrained method to solve free faces; L-BFGS-B maintains a limited-memory
curvature model while projecting trial points into the feasible box.

## Equality and inequality constraints

`constrained_minimize` accepts optional bounds, equality constraints, and
inequality constraints. Constraint functions use `RealVectorFn`; their
`derivative` method returns the Jacobian. The current constrained method is an
augmented Lagrangian, whose `InnerMethod` chooses either an unconstrained or a
bound-constrained inner solve.

## Diagnostics and errors

Errors are returned rather than hidden: `ScalarError`, `LineSearchError`,
`UnconstrainedError`, `BoundConstrainedError`, and `ConstrainedError` distinguish
invalid brackets, failed line searches, and iteration limits. Enable the
`enable_trace` feature and configure `topohedral-tracing` when iteration-level
diagnostics are needed.
