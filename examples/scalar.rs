use topohedral_optimize::{
    scalar_minimize, BoundedOptions, DifferentiableFn, ScalarError, ScalarMethod,
};

struct Parabola;

impl DifferentiableFn for Parabola {
    type Input = f64;
    type Output = f64;
    type Derivative = f64;

    fn eval(
        &mut self,
        x: &f64,
    ) -> f64 {
        (x - 2.0).powi(2) + 1.0
    }

    fn derivative(
        &mut self,
        x: &f64,
    ) -> f64 {
        2.0 * (x - 2.0)
    }

    fn dimension_domain(&self) -> usize {
        1
    }

    fn dimension_range(&self) -> usize {
        1
    }
}

fn main() -> Result<(), ScalarError> {
    let options = BoundedOptions::new(-5.0, 5.0)?
        .with_x_abs_tolerance(1e-10)
        .with_max_iter(200);
    let result = scalar_minimize(&mut Parabola, ScalarMethod::Bounded(options))?;

    println!("x = {:.8}, f(x) = {:.8}", result.xmin, result.fmin);
    Ok(())
}
