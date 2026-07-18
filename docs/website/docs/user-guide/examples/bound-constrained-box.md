# Bound-Constrained Box

This example is based on the box-constrained linear-objective test for BFGS-B. We minimize

$$
f(\mathbf{x}) = -x_1 - 2x_2 - 3x_3
$$

subject to $0 \leq x_i \leq 1$. Because each coefficient is negative, decreasing the objective
pushes every variable to its upper bound, so the solution is $(1, 1, 1)$.

First define the objective and its constant gradient:

```rust
use topohedral_linalg::{DVector, VecType, VectorOps};
use topohedral_optimize::{
    bound_constrained_minimize, BaseOptions, BfgsbOptions, BoundConstrainedMethod,
    BoundConstrainedOptions, BoundsConstraints, DifferentiableFn, LineSearchMethod,
    LineSearchOptions, NocedalOptions, Vector,
};

fn colvec(values: &[f64]) -> Vector {
    DVector::<f64>::from_slice_vec(values, values.len(), VecType::Col)
}

#[derive(Debug)]
struct Linear {
    coefficients: Vector,
}

impl DifferentiableFn for Linear {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;

    fn dimension_domain(&self) -> usize {
        self.coefficients.len()
    }

    fn dimension_range(&self) -> usize {
        1
    }

    fn eval(&mut self, x: &Vector) -> f64 {
        self.coefficients.dot(x)
    }

    fn derivative(&mut self, _x: &Vector) -> Vector {
        self.coefficients.clone()
    }
}
```

Add both bounds for each variable, configure BFGS-B, and solve:

```rust
let n = 3;
let mut objective = Linear {
    coefficients: colvec(&[-1.0, -2.0, -3.0]),
};

let mut bounds = BoundsConstraints::new(n);
for i in 0..n {
    bounds.add_bounds(i, Some(0.0), Some(1.0));
}

let x0 = DVector::<f64>::from_value_vec(0.5, n, VecType::Col);
let method = BoundConstrainedMethod::Bfgsb(BfgsbOptions {
    bound_opts: BoundConstrainedOptions {
        base_opts: BaseOptions {
            grad_rtol: 0.0,
            grad_atol: 1e-6,
            max_iter: 100,
        },
        constraint_tol: 1e-15,
    },
    ls_method: LineSearchMethod::Nocedal(NocedalOptions {
        ls_opts: LineSearchOptions {
            c1: 1e-4,
            c2: 0.9,
            step_min: 1e-20,
            step_max: 1e20,
        },
        maxiter: 50,
        zoom_maxiter: 50,
    }),
});

let result = bound_constrained_minimize(
    &mut objective,
    bounds,
    x0,
    method,
)
.unwrap();
```

`result.xmin` is approximately $(1, 1, 1)$ and `result.fmin` is approximately $-6$. All three
variables are active at their upper bounds.
