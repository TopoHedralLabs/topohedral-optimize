//! Shared options and errors for unconstrained optimization.
//!
//! Unconstrained methods use the common gradient-based stopping settings.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::line_search::LineSearchError;
//}}}
//{{{ dep imports
use thiserror::Error;
//}}}
//{{{ std imports
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ type: Options
/// Common options used by unconstrained methods.
pub type Options = crate::common::BaseOptions;
//}}}
//{{{ enum: Error
#[derive(Error, Debug)]
/// Errors reported by unconstrained optimization.
pub enum Error {
    /// A line search failed.
    #[error("Linear search failed with error {0}")]
    LineSearch(#[from] LineSearchError),
    #[error("Maximum iterations of {0} reached")]
    /// The iteration limit was reached.
    MaxIterations(usize),
}
//}}}
