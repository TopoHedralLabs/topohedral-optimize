# Unconstrained Rosenbrock Function

The Rosenbrock function is a standard two-dimensional optimization problem:

$$
f(x, y) = (1-x)^2 + 100(y-x^2)^2.
$$

Its narrow curved valley makes it a useful demonstration of quasi-Newton
minimization and line search. This complete example defines the objective,
builds the method from private, validated option types, and prints the solution.
The source below is compiled as part of the crate's normal test suite.

```rust
--8<-- "examples/unconstrained.rs"
```

Run it with:

```console
cargo run --example unconstrained
```
