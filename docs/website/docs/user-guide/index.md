# User Guide

This guide gives an overview of the public API. The generated [API Docs](../api/)
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
    - `unconstrained_minimize(fcn, x0, method)`: Minimize an unconstrained vector-scalar function.
    - `bound_constrained_minimize(fcn, bounds, x0, method)`: Minimize a bound-constrained
       vector-scalar function.
    - `constrained_minimize(fcn, bounds, eq_constraints, ieq_constraints, x0, method)`: Minimize
       a generally-constrained vector-scalar function.

    with the exception of the line search functionality which has two entry-points:

    - `lsearch(fcn, iter_data, dir, alpha_init, method)`: Find a sufficient-decrease step
       for a vector-scalar function at a particular point in the iteration and in direction `dir`.
    - `lsearch1d(fcn, alpha_init, method)`: Find a sufficient decrease step for a scalar-scalar
       function.

- Each of these minimize functions takes the function you wish to minimize and a set of
  parameters specific to the area. For example, the scalar optimizers take a function and
  a bracket or an interval. The bounded optimizers all take a `BoundsConstraints` struct.
  The generally-constrained optimizers accept optional bounds and optional vector-valued
  equality and inequality constraint functions.
- Every category of solver: so scalar minimizers, unconstrained minimizers, bound constrained minimizers
  and generally-constrained miminizers, has its own method enum and each value of the enum wraps
  around an options struct specific to that method.

    - `ScalarMethod`
        - `ScalarMethod::Bounded(BoundedOptions)`
        - `ScalarMethod::Brent(BrentOptions)`
        - `ScalarMethod::Golden(GoldenOptions)`
    - `UnconstrainedMethod`
        - `UnconstrainedMethod::ConjugateGradient(ConjugateGradientOptions)`
        - `UnconstrainedMethod::QuasiNewton(QuasiNewtonOptions)`
    - `BoundConstrainedMethod`
        - `BoundConstrainedMethod::Asa(AsaOptions)`
        - `BoundConstrainedMethod::Bfgsb(BfgsbOptions)`
    - `ConstrainedMethod`
        - `ConstrainedMethod::AugmentedLagrangian(AugmentedLagrangianOptions)`


- The options structs are constructed hierarchically by category. So a bound-constrained
  solver such as `Asa` will have its own options struct `AsaOptions` which will contain the
  settings specific to itself but it will also contain an instance of `BoundConstrainedOptions`
  which will contain settings common to all the bound constrained minimizers. Finally,
  `BoundConstrainedOptions` will itself contain an instance of `BaseOptions`, which contains
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

See the following complete examples:

- [Quartic Functions](examples/quartic-functions.md)
- [Inequality Constraints](examples/inequality-constraints.md)

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

See the complete [parabola example](examples/scalar-parabola.md), which demonstrates Brent
minimization with automatic and explicit brackets, as well as bounded minimization.

## Unconstrained Minimization

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

See the complete [Rosenbrock example](examples/unconstrained-rosenbrock.md), which demonstrates both
methods.

## Bound-Constrained Minimization

Bound-constrained minimization solves

$$
\mathbf{x}_{\text{min}}
=
\argmin_{\mathbf{x} \in \mathbb{R}^{n}} f(\mathbf{x})
\quad \text{subject to} \quad
l_i \leq x_i \leq u_i.
$$

Either side of a variable's interval may be omitted. Variables for which no bounds are added
remain free. The entry point is:

```rust
fn bound_constrained_minimize<F: RealFn + ?Sized>(
    fcn: &mut F,
    bounds: BoundsConstraints,
    x0: Vector,
    method: BoundConstrainedMethod,
) -> Result<VectorReturns, BoundConstrainedError>
```

### Specifying Bounds

Create `BoundsConstraints` with the dimension of the objective's domain, then add the bounds for
each constrained variable:

```rust
let mut bounds = BoundsConstraints::new(n);

// 0 <= x[0] <= 1
bounds.add_bounds(0, Some(0.0), Some(1.0));

// x[2] <= 5; x[2] has no lower bound
bounds.add_bounds(2, None, Some(5.0));
```

The variable index must be smaller than `n`, and bounds for a given index must be added in a
single call. Both bound-constrained algorithms project `x0` into the feasible box before starting,
so the first objective evaluation is made at a feasible point.

### Choosing a Method

`BoundConstrainedMethod` has two variants:

- `BoundConstrainedMethod::Asa(AsaOptions)` uses an active-set algorithm. It identifies variables
  at their bounds and invokes a configured `UnconstrainedMethod` on the remaining free variables.
  Construct its options with `AsaOptions::new(bound_opts, unconstrained_method)`; the remaining
  ASA-specific fields are initialized to their algorithm defaults.
- `BoundConstrainedMethod::Bfgsb(BfgsbOptions)` uses BFGS-B with a configurable
  `LineSearchMethod`. Construct `BfgsbOptions` from its `bound_opts` and `ls_method` fields.

Both methods use `BoundConstrainedOptions`, which contains:

- `base_opts: BaseOptions` for `grad_rtol`, `grad_atol`, and `max_iter`;
- `constraint_tol`, the shared bound-constrained tolerance. BFGS-B also uses this value as its
  relative function-reduction tolerance.

The return value is `VectorReturns`. Its `xmin` is feasible with respect to the supplied bounds,
and it also reports `fmin`, the convergence reason, iteration count, and function and gradient
evaluation counts.

## Generally-Constrained Minimization

Generally-constrained minimization supports bounds, vector-valued equality constraints, and
vector-valued inequality constraints:

$$
\begin{aligned}
\mathbf{x}_{\text{min}}
&= \argmin_{\mathbf{x} \in \mathbb{R}^{n}} f(\mathbf{x}) \\
\text{subject to} \quad
\mathbf{h}(\mathbf{x}) &= \mathbf{0}, \\
\mathbf{g}(\mathbf{x}) &\leq \mathbf{0}, \\
l_i \leq x_i &\leq u_i.
\end{aligned}
$$

The entry point is:

```rust
fn constrained_minimize<F: RealFn + ?Sized>(
    fcn: &mut F,
    bounds: Option<BoundsConstraints>,
    eq_constraints: Option<&mut dyn RealVectorFn>,
    ieq_constraints: Option<&mut dyn RealVectorFn>,
    x0: Vector,
    method: ConstrainedMethod,
) -> Result<VectorReturns, ConstrainedError>
```

Pass `None` for any category of constraint that is not present. Equality and inequality
constraints implement `RealVectorFn`, returning all constraint values in one `Vector`. Their
derivative is an $n \times m$ `Matrix`, where column $j$ is the gradient of constraint $j$, as
described in [Specifying a Function](#specifying-a-function). An inequality is feasible when its
returned value is non-positive.

The currently available method is
`ConstrainedMethod::AugmentedLagrangian(AugmentedLagrangianOptions)`. It updates penalties and
Lagrange multipliers in an outer loop and solves each resulting subproblem with one of:

- `AugmentedLagrangianInnerMethod::Unconstrained(UnconstrainedMethod)` when there are no native
  variable bounds;
- `AugmentedLagrangianInnerMethod::BoundConstrained(BoundConstrainedMethod)` when `bounds` is
  `Some(...)`.

To enforce native bounds, pair `Some(bounds)` with the bound-constrained inner method. Conversely,
the bound-constrained inner method requires bounds to be present.

Construct the method as follows:

```rust
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
        AugmentedLagrangianInnerMethod::Unconstrained(unconstrained_method),
    ),
);

let result = constrained_minimize(
    &mut objective,
    None,
    Some(&mut equalities),
    Some(&mut inequalities),
    x0,
    method,
)?;
```

`constraint_tol` controls primal feasibility, while the gradient tolerances in `base_opts`
control stationarity. `AugmentedLagrangianOptions::new` supplies defaults for `initial_penalty`,
`constraint_improvement_factor`, and `penalty_growth_factor`; these public fields may be adjusted
after construction.

## Line-Search Methods

A line search chooses a step length $\alpha$ along a direction $\mathbf{d}$ by applying a scalar
search method to

$$
\phi(\alpha) = f(\mathbf{x} + \alpha \mathbf{d}),
\qquad
\phi'(\alpha) = \nabla f(\mathbf{x} + \alpha \mathbf{d})^{T}\mathbf{d}.
$$

The initial direction should be a descent direction, so
$\phi'(0) = \nabla f(\mathbf{x})^{T}\mathbf{d} < 0$. Both available methods seek a step satisfying
the strong Wolfe conditions:

$$
\begin{aligned}
\phi(\alpha)
&\leq \phi(0) + c_1 \alpha \phi'(0), \\
|\phi'(\alpha)|
&\leq c_2 |\phi'(0)|.
\end{aligned}
$$

### Searching Along a Vector Direction

Use `lsearch` when the objective implements `RealFn`:

```rust
fn lsearch<F: RealFn + ?Sized>(
    fcn: &mut F,
    iter_data: &IterData,
    dir: &Vector,
    alpha_init: f64,
    method: LineSearchMethod,
) -> Result<IterData, LineSearchError>
```

`iter_data` contains the current point, function value, gradient, and gradient norm. It can be
created with `IterData::new(&mut fcn, &x)`. On success, `lsearch` returns a new `IterData` at
$\mathbf{x} + \alpha\mathbf{d}$, including the function value and freshly evaluated gradient.

### Searching a Scalar Function

Use `lsearch1d` when the line-search function $\phi$ already implements `RealFn1`:

```rust
let result = lsearch1d(&mut phi, alpha_init, method)?;
let alpha = result.alpha;
let phi_alpha = result.phi_alpha;
```

`lsearch1d` treats zero as the starting point: it evaluates `phi(0)` and `phi'(0)`, then searches
from `alpha_init`. It returns the accepted step and the function value at that step.

### Configuring a Line Search

`LineSearchMethod` has two variants:

- `LineSearchMethod::Thuente(ThuenteOptions)` selects the More-Thuente method and has a `maxiter`
  limit.
- `LineSearchMethod::Nocedal(NocedalOptions)` selects a bracket-and-zoom method and has both
  `maxiter` and `zoom_maxiter` limits.

Both option structs contain `ls_opts: LineSearchOptions`. Its fields are the Wolfe constants `c1`
and `c2`, and the permitted step interval `step_min` to `step_max`. `LineSearchOptions::default()`
uses `c1 = 1e-4`, `c2 = 0.9`, `step_min = 0`, and `step_max = 50`. Set the method-specific
iteration limits explicitly when constructing `ThuenteOptions` or `NocedalOptions`.
