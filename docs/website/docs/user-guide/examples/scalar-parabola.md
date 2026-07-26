# Scalar Minimization: Parabola

This complete example minimizes $(x-2)^2+1$ over the interval $[-5,5]$.
The objective type intentionally does not implement `Debug`; objective traits
place no formatting requirement on user types.

```rust
--8<-- "examples/scalar.rs"
```

For an unbounded search, choose `ScalarMethod::Brent` or
`ScalarMethod::Golden` and supply `Bracket::Auto`, `Bracket::Points`, or a
known `Bracket::Triple`.

Run the example with:

```console
cargo run --example scalar
```
