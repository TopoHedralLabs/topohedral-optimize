//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::{RealFn1, ScalarReturns};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

mod bounded;
mod brent;
mod common;
mod factory;
mod golden;

//{{{ pub use: bounded exports
pub use bounded::{Bounded, Options as BoundedOptions};
//}}}
//{{{ pub use: brent exports
pub use brent::{Brent, Options as BrentOptions};
//}}}
//{{{ pub use: common exports
pub use common::{
    bracket, resolve_bracket, Bracket, BracketOptions, BracketResult, Error as ScalarError,
};
//}}}
//{{{ pub use: golden exports
pub use golden::{Golden, Options as GoldenOptions};
use topohedral_tracing::trace_fn;
//}}}
//{{{ pub use factory exports
pub use factory::{create, Method as ScalarMethod};
//}}}

#[trace_fn]
pub fn minimize<F: RealFn1>(
    fcn: F,
    method: ScalarMethod,
) -> Result<ScalarReturns, ScalarError>
{
    let minimizer = factory::create(fcn, method);
    minimizer?.minimize()
}
