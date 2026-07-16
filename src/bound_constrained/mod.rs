//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::{Evaluator, RealFn, Vector, VectorReturns as BoundConstrainedReturns};
use crate::constraints::BoundsConstraints;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
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
pub fn minimize<F: RealFn + ?Sized>(
    fcn: &mut F,
    bounds: BoundsConstraints,
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
    bounds: BoundsConstraints,
    x0: Vector,
    method: BoundConstrainedMethod,
) -> Result<BoundConstrainedReturns, BoundConstrainedError> {
    let mut minimizer = factory::create(fcn, bounds, x0, method);
    minimizer.minimize()
}
