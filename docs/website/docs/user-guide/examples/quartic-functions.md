# Quartic Functions

Let's say we want a quartic function centered around some point $\mathbf{c}$:

$$
f(\mathbf{x})
=
\sum_{i = 1}^{n}
(x_{i} - c_{i})^{4},
\quad
\frac{\partial f}{\partial x_{i}} = 4 (x_{i} - c_{i})^{3}
$$

```rust
struct Quartic {
    center: Vector,
}

impl DifferentiableFn for Quartic {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;

    fn dimension_domain(&self) -> usize {
        self.center.len()
    }

    fn dimension_range(&self) -> usize {
        1
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        let tmp = x.clone() - self.center.clone();
        let mut out = 0.0;
        for i in 0..5 {
            out += tmp[i].powi(4);
        }
        out
    }

    fn derivative(
        &mut self,
        x_in: &Vector,
    ) -> Vector {
        let tmp = x_in.clone() - self.center.clone();
        let mut out = DVector::<f64>::zeros_vec(5, VecType::Col);
        for i in 0..x_in.len() {
            out[i] = 4.0 * tmp[i].powi(3);
        }
        out
    }
}
```
