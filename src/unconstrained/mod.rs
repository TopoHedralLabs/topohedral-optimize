//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

mod common;
mod conjugate_gradient;

pub use common::{
    Error as UnconstrainedError, Options as UnonstrainedOptions, Returns as UnconstrainedReturns,
    UnconstrainedMinimizer,
};
pub use conjugate_gradient::{ConjugateGradient, Direction, Options as ConjugateGradientOptions};
