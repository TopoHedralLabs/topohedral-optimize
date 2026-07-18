# Inequality Constraints

Let's say we wish to constrain a 3D optimization problem with respect to constraints (these may or
may not make sense as constraints but are used here for the purpose of illustration):

$$
\begin{align*}
g_{1}(\mathbf{x}) & = x_{1} + x_{2} + x_{3} - 5 & \leq  0 \\
g_{2}(\mathbf{x}) & = \exp(x_{1} x_{3}) & \leq 0
\end{align*}
$$

In topohedral-optimize we would encode this as a single, vector-valued function with the following
derivative:

$$
\mathbf{g}(\mathbf{x})
=
\begin{bmatrix}
g_{1}(\mathbf{x}) \\
g_{2}(\mathbf{x})
\end{bmatrix}
,\quad
\mathbf{J} =
\nabla \mathbf{g}
=
[\nabla g_{1}, \nabla g_{2}]
=
\begin{bmatrix}
    1 & x_{3}\exp(x_{1} x_{3}) \\
    1 &  0 \\
    1 & x_{1}\exp(x_{1} x_{3})
\end{bmatrix}
$$

We follow the convention that the Jacobian matrix stores gradients in the columns, with one column
per function. Therefore, the Jacobian should always have `dimension_domain` rows and `dimension_range`
columns.

```rust
struct ConstraintsExample{}

impl DifferentiableFn for ConstraintsExample{
    type Input = Vector;
    type Output = Vector;
    type Derivative = Matrix;

    fn dimension_domain(&self) -> usize {
        3
    }

    fn dimension_range(&self) -> usize {
        2
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> Vector {
        let mut out = Vector::zeros_cvec(2, Col);
        out[0] = x[0] + x[1] + x[2] - 5.0;
        out[1] = (x[0] * x[2]).exp()
    }

    fn derivative(
        &mut self,
        x_in: &Vector,
    ) -> Vector {
        let mut out Matrix::zeros(3, 2);
        // nabla g1
        out[(0, 0)] = 1;
        out[(1, 0)] = 1;
        out[(2, 0)] = 1;
        // nabla g2
        let tmp = (x[0] * x[2]).exp();
        out[(0, 1)] = x[2] * tmp;
        out[(1, 1)] = 0;
        out[(2, 1)] = x[0] * tmp;
    }
}
```
