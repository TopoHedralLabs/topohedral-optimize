# Rosenbrock Function

The Rosenbrock function is a classic 2D optimization test problem:

$$
f(\mathbf{x}) = (a - x_{1})^{2} + b(y - x^{2})^{2}
, \quad
\nabla f(\mathbf{x})
=
\begin{bmatrix}
-2 (a - x) - 4 b x (y - x^{2}) \\
2 b (y - x^{2})
\end{bmatrix}
$$

The parameters $(a, b) = (1, 100)$ give a minimum at $(1, 1)$. We may implement this
function as follows:

```rust
struct Rosenbrock {
    a: f64,
    b: f64,
}
impl Rosenbrock {
    fn new() -> Self {
        Self { a: 1.0, b: 100.0 }
    }
}
impl topohedral_optimize::DifferentiableFn for Rosenbrock {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;
    fn dimension_domain(&self) -> usize {
        2
    }

    fn dimension_range(&self) -> usize {
        1
    }

    fn eval(
        &mut self,
        xvec: &Vector,
    ) -> f64 {
        let x = xvec[0];
        let y = xvec[1];
        (self.a - x).powi(2) + self.b * (y - x.powi(2)).powi(2)
    }

    fn derivative(
        &mut self,
        xvec: &Vector,
    ) -> Vector {
        let a = self.a;
        let b = self.b;
        let x = xvec[0];
        let y = xvec[1];
        let mut out = DVector::<f64>::zeros_vec(2, VecType::Col);
        out[0] = -2.0 * (a - x) - 4.0 * b * x * (y - x.powi(2));
        out[1] = 2.0 * b * (y - x.powi(2));
        out
    }
}
```

We will solve this problem using two different methods:

- Method 1: Conjugate gradient using a Polak-Ribiere direction update formula and a
  More-Thuente line search.
- Method 2: Quasi-Newton using a BFGS Hessian update formula and a Nocedal line search.

```rust
// method 1
let mut rosenbrock = Rosenbrock::new();
let x0 = Vector::from_slice_vec(&[-5.0, -10.0], VecType::Col);
let ret = unconstrained_minimize(rosenbrock, x0, UnconstrainedMethod::ConjugateGradient(ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
    },
    ls_method: LineSearchMethod::Thuente(ThuenteOptions {
        ls_opts: LineSearchOptions {
            c1: 1.0e-4,
            c2: 0.4,
            step_min: 1e-8,
            step_max: 1e5,
        },
        maxiter: 10,
    }),
    direction: Direction::PolakRibiere,
    restart: 10,
}));
```

```rust
// method 2
let mut rosenbrock = Rosenbrock::new();
let x0 = Vector::from_slice_vec(&[-5.0, -10.0], VecType::Col);
let ret = unconstrained_minimize(rosenbrock, x0, UnconstrainedMethod::QuasiNewton(QuasiNewtonOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
    },
    ls_method: LineSearchMethod::Nocedal(NocedalOptions {
        ls_opts: LineSearchOptions {
            c1: 1.0e-4,
            c2: 0.9,
            step_min: 1e-8,
            step_max: 1e5,
        },
        maxiter: 10,
        zoom_maxiter: 10,
    }),
    method: UpdateMethod::BFGS,
    restart: 10,
}));
```
