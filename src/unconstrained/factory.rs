//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

use topohedral_tracing::trace_fn;

//{{{ crate imports
use super::common::UnconstrainedMinimizer;
use super::conjugate_gradient::ConjugateGradient;
use super::conjugate_gradient::Options as ConjugateGradientOptions;
use super::quasi_newton::Options as QuasiNewtonOptions;
use super::quasi_newton::QuasiNewton;
use crate::common::RealFn;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_linalg::dvector::DVector;
//}}}
//--------------------------------------------------------------------------------------------------

#[derive(Copy, Clone)]
pub enum Method
{
    ConjugateGradient(ConjugateGradientOptions),
    QuasiNewton(QuasiNewtonOptions),
}

#[trace_fn]
pub fn create<'a, F: RealFn + 'a>(
    fcn: F,
    x0: DVector<f64>,
    method: Method,
) -> Box<dyn UnconstrainedMinimizer + 'a>
{
    match method
    {
        Method::ConjugateGradient(opts) => Box::new(ConjugateGradient::new(fcn, x0, opts)),
        Method::QuasiNewton(opts) => Box::new(QuasiNewton::new(fcn, x0, opts)),
    }
}
