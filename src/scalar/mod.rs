//! Short Description of module
//!
//! Longer description of module
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
pub fn minimize<F: RealFn1 + ?Sized>(
    fcn: &mut F,
    method: ScalarMethod,
) -> Result<ScalarReturns, ScalarError> {
    let minimizer = factory::create(fcn, method);
    minimizer?.minimize()
}
