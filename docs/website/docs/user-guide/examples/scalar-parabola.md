# Scalar Minimization: Parabola

For the purposes of this example, let us define a little helper to wrap around a Rust lambda:

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

Let's begin with a simple problem: finding the minimum of a parabola. Using `Brent`, one can do one
of the following:

```rust
let mut f = ScalarFunction::new(|f64|(x - 1.0).powi(2));

// method 1: let us compute the bracket starting from interval [0, 1].
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

Methods 1 and 2 implicitly call `bracket`; method 3 does not. Minimizing using `Bounded` would
look like:

```rust
let res3 = scalar_minimize(
                        &mut f,
                        ScalarMethod::Bounded(
                            BoundedOptions::new(0.0, 1.0)
                        )
                    ).unwrap();
```
