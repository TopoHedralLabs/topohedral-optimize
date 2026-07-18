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

See the complete [Rosenbrock example](examples/unconstrained-rosenbrock.md), which demonstrates both
methods.
