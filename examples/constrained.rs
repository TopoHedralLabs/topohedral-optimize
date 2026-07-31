use topohedral_linalg::{DVector, VecType};
use topohedral_optimize::{
    constrained_minimize, AugmentedLagrangianInnerMethod, AugmentedLagrangianOptions, BaseOptions,
    ConstrainedMethod, ConstrainedOptions, DifferentiableFn, LineSearchMethod, Matrix,
    QuasiNewtonOptions, QuasiNewtonUpdateMethod, ThuenteOptions, UnconstrainedMethod, Vector,
};

fn vector(values: &[f64]) -> Vector {
    DVector::from_slice_vec(values, values.len(), VecType::Col)
}

struct ShiftedQuadratic;

impl DifferentiableFn for ShiftedQuadratic {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;

    fn eval(
        &mut self,
        point: &Vector,
    ) -> f64 {
        (point[0] - 2.0).powi(2) + point[1].powi(2)
    }

    fn derivative(
        &mut self,
        point: &Vector,
    ) -> Vector {
        vector(&[2.0 * (point[0] - 2.0), 2.0 * point[1]])
    }

    fn dimension_domain(&self) -> usize {
        2
    }

    fn dimension_range(&self) -> usize {
        1
    }
}

struct UnitDisk;

impl DifferentiableFn for UnitDisk {
    type Input = Vector;
    type Output = Vector;
    type Derivative = Matrix;

    fn eval(
        &mut self,
        point: &Vector,
    ) -> Vector {
        vector(&[point[0].powi(2) + point[1].powi(2) - 1.0])
    }

    fn derivative(
        &mut self,
        point: &Vector,
    ) -> Matrix {
        let mut jacobian = Matrix::zeros(2, 1);
        jacobian[(0, 0)] = 2.0 * point[0];
        jacobian[(1, 0)] = 2.0 * point[1];
        jacobian
    }

    fn dimension_domain(&self) -> usize {
        2
    }

    fn dimension_range(&self) -> usize {
        1
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let inner = UnconstrainedMethod::QuasiNewton(QuasiNewtonOptions::new(
        BaseOptions::default(),
        LineSearchMethod::Thuente(ThuenteOptions::default()),
        QuasiNewtonUpdateMethod::Bfgs,
    ));
    let method = ConstrainedMethod::AugmentedLagrangian(AugmentedLagrangianOptions::new(
        ConstrainedOptions::default(),
        AugmentedLagrangianInnerMethod::Unconstrained(inner),
    ));

    let mut inequality = UnitDisk;
    let result = constrained_minimize(
        &mut ShiftedQuadratic,
        None,
        None,
        Some(&mut inequality),
        vector(&[0.5, 0.5]),
        method,
    )?;

    println!("x = {:?}, f(x) = {:.8}", result.xmin, result.fmin);
    Ok(())
}
