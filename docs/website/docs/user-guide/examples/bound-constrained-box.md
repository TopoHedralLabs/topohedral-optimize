# Bound-Constrained Quadratic

This example minimizes a shifted quadratic while restricting both variables.
The unconstrained minimizer is $(3,-2)$, but the bounds require
$x_0 \in [0,1]$ and $x_1 \ge -1$, so the constrained solution lies at
$(1,-1)$.

`BoundConstraints::add_bounds` validates indices and values and returns a
`Result`. Bound entries may be added in any order; traversal is deterministic
by variable index.

```rust
--8<-- "examples/bound_constrained.rs"
```

Run it with:

```console
cargo run --example bound_constrained
```
