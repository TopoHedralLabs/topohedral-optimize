# One-Dimensional Line Search

This example comes from the one-dimensional line-search tests. Define

$$
\phi(\alpha) = (\alpha - 10)(\alpha - 100),
\qquad
\phi'(\alpha) = 2\alpha - 110.
$$

At $\alpha = 0$, the derivative is negative, so increasing $\alpha$ is a descent direction.

```rust
use topohedral_optimize::{
    lsearch1d, DifferentiableFn, LineSearchMethod, LineSearchOptions, NocedalOptions,
    ThuenteOptions,
};

#[derive(Debug)]
struct Quadratic1D {
    root1: f64,
    root2: f64,
}

impl DifferentiableFn for Quadratic1D {
    type Input = f64;
    type Output = f64;
    type Derivative = f64;

    fn dimension_domain(&self) -> usize {
        1
    }

    fn dimension_range(&self) -> usize {
        1
    }

    fn eval(&mut self, alpha: &f64) -> f64 {
        (alpha - self.root1) * (alpha - self.root2)
    }

    fn derivative(&mut self, alpha: &f64) -> f64 {
        2.0 * alpha - (self.root1 + self.root2)
    }
}
```

Run the More-Thuente method with an initial step of 10:

```rust
let mut phi = Quadratic1D {
    root1: 10.0,
    root2: 100.0,
};

let thuente_result = lsearch1d(
    &mut phi,
    10.0,
    LineSearchMethod::Thuente(ThuenteOptions {
        ls_opts: LineSearchOptions {
            step_max: 500.0,
            ..Default::default()
        },
        maxiter: 100,
    }),
)
.unwrap();
```

The Nocedal method uses the same common options and adds a limit for its zoom phase:

```rust
let nocedal_result = lsearch1d(
    &mut phi,
    10.0,
    LineSearchMethod::Nocedal(NocedalOptions {
        ls_opts: LineSearchOptions {
            step_max: 500.0,
            ..Default::default()
        },
        maxiter: 100,
        zoom_maxiter: 100,
    }),
)
.unwrap();
```

For this initial step, both methods accept $\alpha = 10$ and return
`phi_alpha = 0`. The exact minimizer of this quadratic is $\alpha = 55$; the earlier step is
accepted because a line search looks for sufficient decrease and curvature, not necessarily the
exact one-dimensional minimum.
