use topohedral_optimize::{
    line_search_1d, DifferentiableFn, LineSearchError, LineSearchMethod, ThuenteOptions,
};

struct Quadratic;

impl DifferentiableFn for Quadratic {
    type Input = f64;
    type Output = f64;
    type Derivative = f64;

    fn eval(
        &mut self,
        alpha: &f64,
    ) -> f64 {
        (alpha - 3.0).powi(2)
    }

    fn derivative(
        &mut self,
        alpha: &f64,
    ) -> f64 {
        2.0 * (alpha - 3.0)
    }

    fn dimension_domain(&self) -> usize {
        1
    }

    fn dimension_range(&self) -> usize {
        1
    }
}

fn main() -> Result<(), LineSearchError> {
    let result = line_search_1d(
        &mut Quadratic,
        1.0,
        LineSearchMethod::Thuente(ThuenteOptions::default()),
    )?;

    println!(
        "alpha = {:.8}, phi(alpha) = {:.8}",
        result.alpha, result.phi_alpha
    );
    Ok(())
}
