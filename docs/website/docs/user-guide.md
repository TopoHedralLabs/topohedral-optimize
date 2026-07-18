# User Guide

This guide gives an overview of the public API. The generated [API Docs](api/)
remain the reference for every option field and method. The public API has the following
areas of functionality:

- Scalar minimization (exact minimization in 1D)
- Scalar line search (sufficient-decrease step-finder in 1D)
- Multi-dimensional, unconstrained optimization.
- Multi-dimensional, bound-constrained optimization.
- Multi-dimensional, generally-constrained optimization.

A constraint is considered "general" if is not a bound constraint, so both linear and non-linear
constraints fall into this category.

## A Tour of the API 

- The first thing to notice is that each area of functionality has a single entry point:

    - `scalar_minimize(fcn, method)`: Minimize a scalar-scalar function.
    - `unconstrained_minimize(fcn, method)`: Mimimize an unconstrained vector-scalar function.
    - `bound_constrained_minimize(fcn, bounds, x0, method)`: Minimize a bound-constrained
       vector-scalar function.
    - `constrained_minimze(fcn, bounds, ieq_con, eq_con, method)`: Minimize a generally-constrained
       vector-scalar function.

    with the exception of the line search functionality which has two entry-points:

    - `lsearch(fcn, iter_data, dir, alpha_init, method)`: Find a sufficient-decrease step
       for a vector-scalar function at a particular point in the iteration and in direction `dir`.
    - `lsearch1d(fcn, alpha_init, method)`: Find a sufficient decrease step for a scalar-scalar
       function.

- Each of these minimize functions takes the function you wish to minimize and a set of
  parameters specific to the area. For example, the scalar optimizers take a function and
  a bracket or an interval. The bounded optimizers all take a `BoundConstraints` struct.
  The generally-constrained optimizers will take a `BoundConstraints`, an object implementing
  equality constraints and an object implementing
- Every category of solver: so scalar minimizers, unconstrained minimizers, bound constrained minimizers
  and generally-constrained miminizers, has its own method enum and each value of the enum wraps
  around an options struct specific to that method.

    - `ScalarMethod`
        - `ScalarMethod::Bounded(BoundedOptions)`
        - `ScalarMethod::Brent(BrentOptions)`
        - `ScalarMethod::Golden(GoldenOptions)`
    - `UnconstraintedMethod`
        - `UnconstraintedMethod::ConjugateGradient(ConjugateGradientOptions)`
        - `UnconstraintedMethod::QuasiNewton(QuasiNewtonOptions)`
    - `BoundConstrainedMethod`
        - `BoundConstrainedMethod::Asa(AsaOptions)`
        - `BoundConstrainedMethod::Bfgsb(BfgsbOptions)`
    - `ConstrainedMethod`
        - `ConstrainedMethod::AugmentedLagrangian(AugmentedLagrangianOptions)`


- The options structs are constructed hierarchically by category. So a bound-constrained
  solver such as `Asa` will have its own options struct `AsaOptions` which will contain the
  settings specific to itself but it will also contain an instance of `BoundConstrainedOptions`
  which will contain settings common to all the bound constrained minimizers. Finally,
  `BoundConstrainedOptions` will itself containt an instance of `BaseOpts`, which contains
  settings which are common to all the descent-based minimizers in the crate.

## Specifying a Function

We shall use the following terminology when describing functions:

- A **scalar-scalar** function is: $f(x): \mathbb{R} \rightarrow \mathbb{R}$
- A **vector-scalar** function: $f(\mathbf{x}): \mathbb{R}^{n} \rightarrow \mathbb{R}$:
- A **vector-vector** function is:  $f(\mathbf{x}): \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}$

The first two are the objects which this crate provides methods to minimize. The last is used
primarily to represent sets of constriant functions.

The trait which users must implement to make their type compatible with the optimization algorithms
is `DifferentiableFn`. It has three types which must be specified:

- `Input`: This is the input type, for example `f64` for a scalar-scalar function.
- `Output`: This is the output type, so the result of the function `eval()` for example `Vector` for a
            scalar-vector function or a vector-vector funcition.
- `Derivative`: This is the type of the function `derivative()`, which for scalar-scalar functions is `f64`.
                For a vector-vector function this will be a `Matrix`.

In addition to providing these three types you must implement the following methods:

- `eval(&mut self, x: &Self::Input)`: evaluates the function at point `x`.
- `derivative(&mut self, x: &Self::Input)`: evaluates the gradient at point `x`.
- `dimension_domain(&self)`: tells you the expected dimension of then input.
- `dimension_range(&self)`: tells you the expected dimension of the output.

For the purpose of annotating types we created the following trait aliases:

- `RealFn1` for the case where `Input=f64`, `Output=f64` and `Derivative=f64`
- `RealFn` for the case where `Input=Vector`, `Output=f64` and `Derivative=Vector`
- `RealVectorFn` for the case where `Input=Vector`, `Output=Vector` and `Derivative=Matrix`


### Example 1: Quartic Functions

Lets's say we want a quartic function centered around some point $\mathbf{c}$

$$
f(\mathbf{x})
=
\sum_{i = 1}^{n}
(x_{i} - c_{i})^{4},
\quad
\frac{\partial f}{\partial x_{i}} = 4 (x_{i} - c_{i})^{3}
$$

```rust
struct Quartic {
    center: Vector,
}

impl DifferentiableFn for Quartic {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;

    fn dimension_domain(&self) -> usize {
        self.center.len()
    }

    fn dimension_range(&self) -> usize {
        1
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        let tmp = x.clone() - self.center.clone();
        let mut out = 0.0;
        for i in 0..5 {
            out += tmp[i].powi(4);
        }
        out
    }

    fn derivative(
        &mut self,
        x_in: &Vector,
    ) -> Vector {
        let tmp = x_in.clone() - self.center.clone();
        let mut out = DVector::<f64>::zeros_vec(5, VecType::Col);
        for i in 0..x_in.len() {
            out[i] = 4.0 * tmp[i].powi(3);
        }
        out
    }
}
```

### Example 2: InequalityConstraints

Let's say we which to constrain 3D optimization problem with respect to constraints (these may or
may not make sense as constraints but are used here for the purpose of illustration):

$$
\begin{align*}
g_{1}(\mathbf{x}) & = x_{1} + x_{2} + x_{3} - 5 & \leq  0 \\
g_{2}(\mathbf{x}) & = \exp(x_{1} x_{3}) & \leq 0
\end{align*}
$$

In topohedral-optimize we would encode this as a single, vector-valued function with the following
derivative:

$$
\mathbf{g}(\mathbf{x})
=
\begin{bmatrix}
g_{1}(\mathbf{x}) \\
g_{2}(\mathbf{x})
\end{bmatrix}
,\quad
\mathbf{J} =
\nabla \mathbf{g}
=
[\nabla g_{1}, \nabla g_{2}]
=
\begin{bmatrix}
    1 & x_{3}\exp(x_{1} x_{3}) \\
    1 &  0 \\
    1 & x_{1}\exp(x_{1} x_{3})
\end{bmatrix}
$$

We folow the convention that the Jacobin matrix stores gradients in the columns, with one column
per function. Therefore, the Jacobian should always have `dimension_domain` rows and `dimension_range`
columns.

```rust
struct ConstraintsExample{}

impl DifferentiableFn for ConstraintsExample{
    type Input = Vector;
    type Output = Vector;
    type Derivative = Matrix;

    fn dimension_domain(&self) -> usize {
        3
    }

    fn dimension_range(&self) -> usize {
        2
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> Vector {
        let mut out = Vector::zeros_cvec(2, Col);
        out[0] = x[0] + x[1] + x[2] - 5.0;
        out[1] = (x[0] * x[2]).exp()
    }

    fn derivative(
        &mut self,
        x_in: &Vector,
    ) -> Vector {
        let mut out Matrix::zeros(3, 2);
        // nabla g1
        out[(0, 0)] = 1;
        out[(1, 0)] = 1;
        out[(2, 0)] = 1;
        // nabla g2
        let tmp = (x[0] * x[2]).exp();
        out[(0, 1)] = x[2] * tmp;
        out[(1, 1)] = 0;
        out[(2, 1)] = x[0] * tmp;
    }
}
```


## Scalar minimization

Scalar minimization solves the following problem for $f(x): \mathbb{R} \rightarrow \mathbb{R}$

$$
x_{\text{min}} = \argmin_{x \in [a, b]} f(x)
$$

The entry-point for this area of functionality is the function:
```rust
fn scalar_minimize(fcn: RealFn1, method: ScalarMethod);
```
where `method` is is one of:

- `ScalarMethod::Brent`: Unbounded, hybrid of bisection, secant and inverse quadratic interpolation.
- `ScalarMethod::Golden`: Unbounded, bisection method.
- `ScalarMethod::Bounded`: Similar to Brant but bounded on a given interval.

Each member of this enum contains the option struct corresponding to that method, in this way
one cas specify both the method and its options in one place.
With the methods `Brent` and `Golden`  the user can either specify a initial bracket where they beleive
the minimum to exist or have the minimizer find one based on two inital points.
A bracket is an ordered triple $(a, b, c)$ such that $f(a) > f(b) < f(c)$.
For the `Bounded` method the user specifies an interval where to search. As opposed to `Brent` and
`Golden`, where the initial bracketing phase can take you quite far from the initial points you
specify, `Bounded` is guaranteed to give you a result in the interval you specify, even if the
minimum occurs at either of the extrema of the interval.

### Example 1: Parabola

For the purposes of this
example, let us define a little helper to wrap around a rust lambda:

```rust
struct ScalarFunction<G: Fn(f64) -> f64> {
    f: G,
}
impl<G: Fn(f64) -> f64> ScalarFunction<G> {
    fn new(f: G) -> Self {
        Self { f }
    }
}
```

Let's begin with a simple problem: to find the minimum of a parabola. Using `Brent` one can do one
of the following:

```rust
let mut f = ScalarFunction::new(|f64|(x - 1.0).powi(2));

// method 1: let us compute the bracket starting from  interval [0, 1].
let res1 = scalar_minimize(
                    &mut f,
                    ScalarMethod::Brent(
                        BrentOptions::new(Bracket::AUTO)
                    )
                ).unwrap();

// method 2: let us compute the bracket starting from interval [0, 5].
let res2 = scalar_minimize(
                    &mut f,
                    ScalarMethod::Brent(
                        BrentOptions::new(Bracket::Points((0.0, 0.5)))
                    )
                ).unwrap();

// method 3: User gives us a bracket where they know the minimum exists
let res3 = scalar_minimize(
                    &mut f,
                    ScalarMethod::Brent(
                        BrentOptions::new(Bracket::Triple((0.0, 0.5, 1.0)))
                    )
                ).unwrap();
```
Methods 1 and 2 implicitly call `bracket`, method 3 does not. Minimizing using `Bounded` would
look like:

```rust
let res3 = scalar_minimize(
                        &mut f,
                        ScalarMethod::Bounded(
                            BoundedOptions::new(0.0, 1.0)
                        )
                    ).unwrap();
```


## Unconstrained minimization

Unconstrained mimimimzation solve solves the following problem for
$f(\mathbf{x}): \mathbb{R}^{n} \rightarrow \mathbb{R}$:

$$
\mathbf{x}_{\text{min}} = \argmin_{\mathbf{x} \in \mathbb{R}^{n}} f(\mathbf{x})
$$

The entry point for this area of functionality is the function:

```rust
fn unconstrained_minimize(fcn: RealFn, x0: &Vector, method: UnconstrainedMethod)
```

where `x0` is some initial point at which to start the optimization. The enum
`UnconstrainedMethod` selects the algorithm and owns its algorithm-specific
options. Current options are:

- `UnconstraindMethod::ConjugateGradient`
- `UnconstrainedMethod::QuasiNewton`


### Example 1 Rosenbrock

A classic 2D test problem of optimization:

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

where the parameters $(a, b) = (1, 100)$ give a minimum at $(1 1)$. We may implement this
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

We will solve this problem using 2 different methods:

- Method 1: Conjugate-Gradient using a Polak-Ribiere direction update formula and using a
 thuente-More line-search
- Method 2: Quasi-Newton using a BFGS hessian update forula and a Nocedal line-search


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