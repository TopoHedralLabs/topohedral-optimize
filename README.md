# topohedral-optimize

Differentiable scalar and multidimensional optimization algorithms for Rust.

The crate includes scalar minimizers, More–Thuente and Nocedal line searches,
conjugate-gradient and quasi-Newton methods, ASA and BFGS-B box-constrained
methods, and augmented-Lagrangian constrained minimization.

## Quick start

```rust
use topohedral_optimize::{
    scalar_minimize, BoundedOptions, DifferentiableFn, ScalarMethod,
};

struct Parabola;

impl DifferentiableFn for Parabola {
    type Input = f64;
    type Output = f64;
    type Derivative = f64;

    fn eval(&mut self, x: &f64) -> f64 {
        (x - 2.0).powi(2) + 1.0
    }

    fn derivative(&mut self, x: &f64) -> f64 {
        2.0 * (x - 2.0)
    }

    fn dimension_domain(&self) -> usize {
        1
    }

    fn dimension_range(&self) -> usize {
        1
    }
}

fn main() -> Result<(), topohedral_optimize::ScalarError> {
    let options = BoundedOptions::new(-5.0, 5.0)?;
    let result = scalar_minimize(&mut Parabola, ScalarMethod::Bounded(options))?;

    assert!((result.xmin - 2.0).abs() < 1e-6);
    Ok(())
}
```

Configuration structs provide `new`, `Default` where a meaningful default
exists, read-only accessors, `with_*` builders, and `validate`. Optimizer entry
points validate complete configurations automatically.

Enable the `serde` feature to serialize supported options, constraints, and
results. Enable `trace` for detailed algorithm tracing.

## Documentation

- [API documentation](https://topohedrallabs.github.io/topohedral-optimize/api/topohedral_optimize/)
- [User guide](https://topohedrallabs.github.io/topohedral-optimize/)
- [Changelog](CHANGELOG.md)

## License

Licensed under the [MIT License](LICENSE).
