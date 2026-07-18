# Generally-Constrained Hypersphere

This example adapts the hypersphere inequality tests for the augmented-Lagrangian solver. We
minimize a shifted quadratic in two dimensions:

$$
f(\mathbf{x}) = (x_1 - 2)^2 + x_2^2
$$

subject to the unit-disk inequality

$$
g(\mathbf{x}) = x_1^2 + x_2^2 - 1 \leq 0.
$$

The unconstrained minimum $(2, 0)$ is outside the disk. The constrained minimum is therefore the
nearest point on its boundary, $(1, 0)$.

Define the scalar-valued objective:

```rust
use topohedral_linalg::{DVector, VecType};
use topohedral_optimize::{
    constrained_minimize, AugmentedLagrangianInnerMethod, AugmentedLagrangianOptions,
    BaseOptions, ConstrainedMethod, ConstrainedOptions, DifferentiableFn, LineSearchMethod,
    LineSearchOptions, Matrix, QuasiNewtonOptions, QuasiNewtonUpdateMethod, ThuenteOptions,
    UnconstrainedMethod, UnconstrainedOptions, Vector,
};

fn colvec(values: &[f64]) -> Vector {
    DVector::<f64>::from_slice_vec(values, values.len(), VecType::Col)
}

#[derive(Debug)]
struct ShiftedQuadratic;

impl DifferentiableFn for ShiftedQuadratic {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;

    fn dimension_domain(&self) -> usize {
        2
    }

    fn dimension_range(&self) -> usize {
        1
    }

    fn eval(&mut self, x: &Vector) -> f64 {
        (x[0] - 2.0).powi(2) + x[1].powi(2)
    }

    fn derivative(&mut self, x: &Vector) -> Vector {
        colvec(&[2.0 * (x[0] - 2.0), 2.0 * x[1]])
    }
}
```

The inequality function returns a one-element `Vector`. Its derivative is a $2 \times 1$ matrix
whose only column is $\nabla g$:

```rust
#[derive(Debug)]
struct UnitDisk;

impl DifferentiableFn for UnitDisk {
    type Input = Vector;
    type Output = Vector;
    type Derivative = Matrix;

    fn dimension_domain(&self) -> usize {
        2
    }

    fn dimension_range(&self) -> usize {
        1
    }

    fn eval(&mut self, x: &Vector) -> Vector {
        colvec(&[x[0].powi(2) + x[1].powi(2) - 1.0])
    }

    fn derivative(&mut self, x: &Vector) -> Matrix {
        let mut jacobian = Matrix::zeros(2, 1);
        jacobian[(0, 0)] = 2.0 * x[0];
        jacobian[(1, 0)] = 2.0 * x[1];
        jacobian
    }
}
```

There are no variable bounds, so the augmented-Lagrangian subproblems can use an unconstrained
quasi-Newton method:

```rust
let line_search = LineSearchMethod::Thuente(ThuenteOptions {
    ls_opts: LineSearchOptions {
        c1: 1e-4,
        c2: 0.9,
        step_min: 1e-8,
        step_max: 1e5,
    },
    maxiter: 100,
});

let inner_method = UnconstrainedMethod::QuasiNewton(QuasiNewtonOptions {
    uncon_opts: UnconstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
    },
    ls_method: line_search,
    method: QuasiNewtonUpdateMethod::BFGS,
    restart: 10,
});

let method = ConstrainedMethod::AugmentedLagrangian(
    AugmentedLagrangianOptions::new(
        ConstrainedOptions {
            base_opts: BaseOptions {
                grad_rtol: 1e-6,
                grad_atol: 1e-8,
                max_iter: 1_000,
            },
            constraint_tol: 1e-8,
        },
        AugmentedLagrangianInnerMethod::Unconstrained(inner_method),
    ),
);

let mut objective = ShiftedQuadratic;
let mut inequality = UnitDisk;
let x0 = colvec(&[0.5, 0.5]);

let result = constrained_minimize(
    &mut objective,
    None,
    None,
    Some(&mut inequality),
    x0,
    method,
)
.unwrap();
```

`result.xmin` is approximately $(1, 0)$ and `result.fmin` is approximately $1$. Evaluating the
constraint at the result gives approximately zero, so the inequality is active.
