# Welcome to TopoHedral-Optimize

`topohedral-optimize` provides differentiable optimization routines for scalar and
vector-valued problems. The public API covers:

- one-dimensional minimization with bounded, Brent, and golden-section methods;
- vector line searches using More-Thuente and Nocedal-style algorithms;
- unconstrained conjugate-gradient and quasi-Newton minimization;
- simple bound-constrained minimization with ASA and L-BFGS-B;
- equality and inequality constrained minimization with an augmented Lagrangian.

The algorithms operate on `f64` values and `topohedral_linalg::DVector<f64>`
vectors. Functions provide both their value and derivative through the crate’s
`DifferentiableFn` trait.

Use the [Getting Started](getting-started.md) page for installation and a minimal
example, then see the [User Guide](user-guide.md) for choosing an algorithm and
configuring stopping criteria.
