//! Shared options and errors for unconstrained optimization.
//!
//! Unconstrained methods use the common gradient-based stopping settings.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::line_search::LineSearchError;
use crate::ValidationError;
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
#[non_exhaustive]
pub enum Error {
    /// A line search failed.
    #[error("line search failed: {0}")]
    LineSearch(#[from] LineSearchError),
    #[error("maximum number of iterations ({0}) reached")]
    /// The iteration limit was reached.
    MaxIterations(usize),
    /// An argument or optimizer option was invalid.
    #[error(transparent)]
    Validation(#[from] ValidationError),
}
//}}}
