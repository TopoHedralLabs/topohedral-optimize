//! Algorithms for minimizing functions subject to simple variable bounds.
//!
//! The module exposes active-set and BFGS-B methods through one entry point.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::{
    DifferentiableFn, Evaluator, RealFn, Vector, VectorReturns as BoundConstrainedReturns,
};
use crate::constraints::BoundConstraints;
use crate::ValidationError;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_linalg::VectorOps;
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ mod: submodules
mod asa;
mod bfgsb;
mod common;
mod factory;
mod utils;
//}}}
//{{{ pub use: asa exports
pub use asa::Options as AsaOptions;
//}}}
//{{{ pub use: bfgsb exports
pub use bfgsb::Options as BfgsbOptions;
//}}}
//{{{ pub use: common exports
pub use common::{Error as BoundConstrainedError, Options as BoundConstrainedOptions};
//}}}
//{{{ pub use: factory exports
pub use factory::Method as BoundConstrainedMethod;
//}}}

#[trace_fn]
/// Minimizes a differentiable function subject to bound constraints.
///
/// # Errors
///
/// Returns [`BoundConstrainedError`] if the method configuration or dimensions
/// are invalid, an inner solve or line search fails, or the iteration limit is
/// reached.
pub fn minimize<F: RealFn + ?Sized>(
    fcn: &mut F,
    bounds: BoundConstraints,
    x0: Vector,
    method: BoundConstrainedMethod,
) -> Result<BoundConstrainedReturns, BoundConstrainedError> {
    let mut evaluator = Evaluator::new(fcn);
    let mut ret = minimize_impl(&mut evaluator, bounds, x0, method)?;
    ret.num_fun_evals = evaluator.num_func_evals;
    ret.num_grad_evals = evaluator.num_grad_evals;
    Ok(ret)
}

pub(crate) fn minimize_impl<F: RealFn + ?Sized>(
    fcn: &mut F,
    bounds: BoundConstraints,
    x0: Vector,
    method: BoundConstrainedMethod,
) -> Result<BoundConstrainedReturns, BoundConstrainedError> {
    method.validate()?;
    let expected = fcn.dimension_domain();
    for (parameter, actual) in [("x0", x0.len()), ("bounds", bounds.dimension_domain())] {
        if actual != expected {
            return Err(ValidationError::DimensionMismatch {
                parameter,
                expected,
                actual,
            }
            .into());
        }
    }
    let mut minimizer = factory::create(fcn, bounds, x0, method);
    minimizer.minimize()
}
