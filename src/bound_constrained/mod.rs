//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::constraints::BoundsConstraints;
use crate::{RealFn, Vector};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ mod: submodules
mod asa;
mod common;
mod factory;
mod utils;
//}}}
//{{{ pub use: asa exports
pub use asa::{ActiveSetAlgorithm, Options as AsaOptions};
//}}}
//{{{ pub use: common exports
pub use common::{Error as BoundConstrainedError, Options as BoundConstrainedOptions};
//}}}
//{{{ pub use: factory exports
pub use factory::{create, Method as BoundConstrainedMethod};
//}}}

#[trace_fn]
pub fn minimize<F1: RealFn>(
    fcn: F1,
    bounds: BoundsConstraints,
    x0: Vector,
    method: BoundConstrainedMethod,
)
{
}
