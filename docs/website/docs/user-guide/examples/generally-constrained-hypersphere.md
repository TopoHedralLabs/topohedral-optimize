# Generally-Constrained Unit Disk

This example minimizes

$$
f(x_0,x_1) = (x_0-2)^2 + x_1^2
$$

subject to the unit-disk inequality

$$
g(x_0,x_1)=x_0^2+x_1^2-1 \le 0.
$$

The unconstrained solution lies outside the disk, so the inequality is active
at the constrained solution. Constraint functions return a vector, and their
derivative is a matrix with one gradient column per constraint.

The augmented-Lagrangian method uses a quasi-Newton optimizer for its inner
unconstrained problems:

```rust
--8<-- "examples/constrained.rs"
```

Run it with:

```console
cargo run --example constrained
```
