use topohedral_linalg::{DVector, VecType};
use topohedral_optimize::{
    unconstrained_minimize, BaseOptions, DifferentiableFn, LineSearchMethod, QuasiNewtonOptions,
    QuasiNewtonUpdateMethod, ThuenteOptions, UnconstrainedError, UnconstrainedMethod, Vector,
};

struct Rosenbrock;

impl DifferentiableFn for Rosenbrock {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;

    fn eval(
        &mut self,
        point: &Vector,
    ) -> f64 {
        let x = point[0];
        let y = point[1];
        (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2)
    }

    fn derivative(
        &mut self,
        point: &Vector,
    ) -> Vector {
        let x = point[0];
        let y = point[1];
        DVector::from_slice_vec(
            &[
                -2.0 * (1.0 - x) - 400.0 * x * (y - x.powi(2)),
                200.0 * (y - x.powi(2)),
            ],
            2,
            VecType::Col,
        )
    }

    fn dimension_domain(&self) -> usize {
        2
    }

    fn dimension_range(&self) -> usize {
        1
    }
}

fn main() -> Result<(), UnconstrainedError> {
    let start = DVector::from_slice_vec(&[-1.2, 1.0], 2, VecType::Col);
    let options = QuasiNewtonOptions::new(
        BaseOptions::default(),
        LineSearchMethod::Thuente(ThuenteOptions::default()),
        QuasiNewtonUpdateMethod::Bfgs,
    );
    let result = unconstrained_minimize(
        &mut Rosenbrock,
        start,
        UnconstrainedMethod::QuasiNewton(options),
    )?;

    println!("x = {:?}, f(x) = {:.8}", result.xmin, result.fmin);
    Ok(())
}
