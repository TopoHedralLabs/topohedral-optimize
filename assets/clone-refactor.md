Blast radius: **moderate-to-high** if you remove `Clone` from `RealFn` itself. I would not do it just to make `ConstraintCollection` heterogeneous.

Right now `RealFn: Clone + Debug` is acting as a global promise: any optimizer can duplicate a function handle whenever it needs to pass it into line search, nested solvers, or bookkeeping. A lot of your code relies on that implicitly.

Main affected areas:

- [src/common.rs](/Users/johnferguson/projects/TopoHedralLabs/topohedral/topohedral-optimize/src/common.rs:41): `RealFn` definition. Removing `Clone` makes `dyn RealFn` possible, but every place that currently assumes `F: RealFn` also means `F: Clone` loses that guarantee.

- [src/line_search/mod.rs](/Users/johnferguson/projects/TopoHedralLabs/topohedral/topohedral-optimize/src/line_search/mod.rs:46): `search` clones the function to build a `LineSearchFcn`, then keeps the original to compute the final gradient. This would need redesign, probably passing `&mut F`, returning richer data from the line-search function, or requiring `F: Clone` locally.

- [src/unconstrained/quasi_newton.rs](/Users/johnferguson/projects/TopoHedralLabs/topohedral/topohedral-optimize/src/unconstrained/quasi_newton.rs:169) and [src/unconstrained/conjugate_gradient.rs](/Users/johnferguson/projects/TopoHedralLabs/topohedral/topohedral-optimize/src/unconstrained/conjugate_gradient.rs:180): solver loops clone `self.fcn` for `IterData::new` and line search.

- [src/bound_constrained/bfgsb.rs](/Users/johnferguson/projects/TopoHedralLabs/topohedral/topohedral-optimize/src/bound_constrained/bfgsb.rs:444): BFGS-B clones the objective into line search.

- [src/bound_constrained/asa.rs](/Users/johnferguson/projects/TopoHedralLabs/topohedral/topohedral-optimize/src/bound_constrained/asa.rs:460): ASA is more involved; it clones the function into a restricted subproblem, then calls an inner unconstrained minimizer.

- [src/constrained/augmented_lagrangian.rs](/Users/johnferguson/projects/TopoHedralLabs/topohedral/topohedral-optimize/src/constrained/augmented_lagrangian.rs:773): augmented Lagrangian uses cheap `Arc<Mutex<_>>` cloning for nested solvers, but also does a real clone of the internal augmented function to make a temporary classical Lagrangian in convergence checks.

So mechanically, you could remove `Clone` from `RealFn` and then add `+ Clone` back onto many generic impls/functions:

```rust
impl<F: RealFn + Clone> Minimizer for QuasiNewton<F> { ... }
pub fn search<F: RealFn + Clone>(...) { ... }
```

But that mostly preserves the current constraint under a different spelling. It would make `RealFn` object-safe, but the solvers would still not accept non-clone objectives.

For your immediate goal, I’d keep `RealFn: Clone + Debug` and add a separate object-safe adapter for heterogeneous collections:

```rust
pub trait DynRealFn: Debug
{
    fn dimension(&self) -> usize;
    fn eval(&mut self, x: &Vector) -> f64;
    fn grad(&mut self, x: &Vector) -> Vector;
    fn clone_box(&self) -> Box<dyn DynRealFn>;
}

impl<F> DynRealFn for F
where
    F: RealFn + 'static,
{
    fn dimension(&self) -> usize { RealFn::dimension(self) }
    fn eval(&mut self, x: &Vector) -> f64 { RealFn::eval(self, x) }
    fn grad(&mut self, x: &Vector) -> Vector { RealFn::grad(self, x) }

    fn clone_box(&self) -> Box<dyn DynRealFn>
    {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn DynRealFn>
{
    fn clone(&self) -> Self
    {
        self.clone_box()
    }
}
```

Then:

```rust
#[derive(Debug, Clone)]
pub struct ConstraintCollection
{
    constraints: Vec<Box<dyn DynRealFn>>,
}
```

That gives you heterogeneous constraints without disturbing the rest of the optimizer stack. The larger refactor would be separating “callable real function” from “cloneable real function,” but that is an architectural cleanup, not a small `constraints.rs` change.