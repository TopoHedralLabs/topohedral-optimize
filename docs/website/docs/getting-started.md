# Getting Started

## Installation

Add the crate and its registry dependency to `Cargo.toml`:

```toml
[dependencies]
topohedral-optimize = "0.0.0"
```

The version above reflects the current development package; use the released
version when the crate is published.

## A differentiable objective

Implement `DifferentiableFn` for a type whose input is a column `DVector<f64>`.
The optimization entry points accept a mutable reference because evaluations may
carry state or counters.

```rust
use topohedral_linalg::{DVector, VecType};
use topohedral_optimize::{
    BaseOptions, ConjugateGradientDirection, ConjugateGradientOptions,
    DifferentiableFn, LineSearchMethod, NocedalOptions, UnconstrainedMethod,
    UnconstrainedOptions, unconstrained_minimize,
};

#[derive(Debug)]
struct Rosenbrock;

impl DifferentiableFn for Rosenbrock {
    type Input = DVector<f64>;
    type Output = f64;
    type Derivative = DVector<f64>;

    fn dimension_domain(&self) -> usize { 2 }

    fn eval(&mut self, x: &Self::Input) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }

    fn derivative(&mut self, x: &Self::Input) -> Self::Derivative {
        DVector::from_slice_vec(
            &[-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
             200.0 * (x[1] - x[0] * x[0])],
            2,
            VecType::Col,
        )
    }
}

let mut objective = Rosenbrock;
let x0 = DVector::from_slice_vec(&[-1.2, 1.0], 2, VecType::Col);
let base = BaseOptions { grad_rtol: 1e-6, grad_atol: 1e-8, max_iter: 1_000 };
let options = ConjugateGradientOptions {
    uncon_opts: base,
    ls_method: LineSearchMethod::Nocedal(NocedalOptions::default()),
    direction: ConjugateGradientDirection::PolakRibiere,
    restart: 20,
};
let result = unconstrained_minimize(
    &mut objective,
    x0,
    UnconstrainedMethod::ConjugateGradient(options),
)?;
assert!(result.fmin.is_finite());
# Ok::<(), topohedral_optimize::UnconstrainedError>(())
```

## Feature flags

| Flag | Effect |
|---|---|
| `enable_trace` | Enables structured tracing through `topohedral-tracing`. |

## Result values

Every successful minimization returns a `ScalarReturns` or `VectorReturns` value.
The result contains the minimizer (`xmin`), objective value (`fmin`), convergence
reason, iteration count, and function/derivative evaluation counts.
