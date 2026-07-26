//! Algorithms for unconstrained multidimensional optimization.
//!
//! Conjugate-gradient and quasi-Newton methods are available through one entry point.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports

use crate::common::{Evaluator, RealFn, Vector};
use crate::{ValidationError, VectorReturns};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_linalg::VectorOps;
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ mod: submodules
mod common;
mod conjugate_gradient;
mod factory;
mod quasi_newton;
//}}}
//{{{ pub use: common exports
pub use common::{Error as UnconstrainedError, Options as UnconstrainedOptions};
//}}}
//{{{ pub use: conjugate_gradient exports
pub use conjugate_gradient::{Direction, Options as ConjugateGradientOptions};
//}}}
//{{{ pub use: quasi_newton exports
pub use quasi_newton::{Options as QuasiNewtonOptions, UpdateMethod};
//}}}
//{{{ pub use: factory exports
pub use factory::Method as UnconstrainedMethod;
//}}}
//{{{ fn: minimize
#[trace_fn]
/// Minimizes a differentiable function without explicit constraints.
///
/// # Errors
///
/// Returns [`UnconstrainedError`] if the method configuration or initial-point
/// dimension is invalid, a line search fails, or the iteration limit is
/// reached.
pub fn minimize<F: RealFn + ?Sized>(
    fcn: &mut F,
    x0: Vector,
    method: UnconstrainedMethod,
) -> Result<VectorReturns, UnconstrainedError> {
    let mut evaluator = Evaluator::new(fcn);
    let mut ret = minimize_impl(&mut evaluator, x0, method)?;
    ret.num_fun_evals = evaluator.num_func_evals;
    ret.num_grad_evals = evaluator.num_grad_evals;
    Ok(ret)
}

pub(crate) fn minimize_impl<F: RealFn + ?Sized>(
    fcn: &mut F,
    x0: Vector,
    method: UnconstrainedMethod,
) -> Result<VectorReturns, UnconstrainedError> {
    method.validate()?;
    let expected = fcn.dimension_domain();
    if x0.len() != expected {
        return Err(ValidationError::DimensionMismatch {
            parameter: "x0",
            expected,
            actual: x0.len(),
        }
        .into());
    }
    let mut minimizer = factory::create(fcn, x0, method);
    minimizer.minimize()
}
//}}}
