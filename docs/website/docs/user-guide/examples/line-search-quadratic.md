# One-Dimensional Line Search

A line search chooses a positive step satisfying sufficient-decrease and
curvature conditions. Here More–Thuente searches the quadratic
$\phi(\alpha)=(\alpha-3)^2$ from $\alpha=0$.

```rust
--8<-- "examples/line_search.rs"
```

Run it with:

```console
cargo run --example line_search
```
