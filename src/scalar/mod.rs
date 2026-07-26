//! One-dimensional scalar minimization algorithms.
//!
//! Supports bounded minimization and bracketed unbounded searches.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::RealFn1;
use crate::common::ScalarReturns;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::trace_fn;
//}}}
//--------------------------------------------------------------------------------------------------

mod bounded;
mod brent;
mod common;
mod factory;
mod golden;

//{{{ pub use: crate common exports
//}}}
//{{{ pub use: bounded exports
pub use bounded::Options as BoundedOptions;
//}}}
//{{{ pub use: brent exports
pub use brent::Options as BrentOptions;
//}}}
//{{{ pub use: common exports
pub use common::{bracket, Bracket, BracketOptions, BracketResult, Error as ScalarError};
//}}}
//{{{ pub use: golden exports
pub use golden::Options as GoldenOptions;
//}}}
//{{{ pub use factory exports
pub use factory::Method as ScalarMethod;
//}}}

#[trace_fn]
/// Minimizes a differentiable scalar function using the selected method.
///
/// # Errors
///
/// Returns [`ScalarError`] if the method configuration or bracket is invalid,
/// the objective evaluates to NaN, or the algorithm reaches its iteration
/// limit.
pub fn minimize<F: RealFn1 + ?Sized>(
    fcn: &mut F,
    method: ScalarMethod,
) -> Result<ScalarReturns, ScalarError> {
    method.validate()?;
    let minimizer = factory::create(fcn, method);
    minimizer?.minimize()
}
