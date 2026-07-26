use topohedral_linalg::{DVector, VecType};
use topohedral_optimize::{
    bound_constrained_minimize, BaseOptions, BfgsbOptions, BoundConstrainedMethod,
    BoundConstrainedOptions, BoundConstraints, DifferentiableFn, LineSearchMethod, NocedalOptions,
    Vector,
};

struct ShiftedQuadratic;

impl DifferentiableFn for ShiftedQuadratic {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;

    fn eval(
        &mut self,
        point: &Vector,
    ) -> f64 {
        (point[0] - 3.0).powi(2) + (point[1] + 2.0).powi(2)
    }

    fn derivative(
        &mut self,
        point: &Vector,
    ) -> Vector {
        DVector::from_slice_vec(
            &[2.0 * (point[0] - 3.0), 2.0 * (point[1] + 2.0)],
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut bounds = BoundConstraints::new(2);
    bounds.add_bounds(0, Some(0.0), Some(1.0))?;
    bounds.add_bounds(1, Some(-1.0), None)?;

    let method = BoundConstrainedMethod::Bfgsb(BfgsbOptions::new(
        BoundConstrainedOptions::new(BaseOptions::default()),
        LineSearchMethod::Nocedal(NocedalOptions::default()),
    ));
    let start = DVector::from_slice_vec(&[0.5, 0.0], 2, VecType::Col);
    let result = bound_constrained_minimize(&mut ShiftedQuadratic, bounds, start, method)?;

    println!("x = {:?}, f(x) = {:.8}", result.xmin, result.fmin);
    Ok(())
}
