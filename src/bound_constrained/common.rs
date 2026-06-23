//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::BaseOptions;
use crate::line_search::LineSearchError;
use crate::unconstrained::UnconstrainedError;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use thiserror::Error;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct: Options
#[derive(Copy, Clone)]
pub struct Options
{
    pub base_opts: BaseOptions,
    pub constraint_tol: f64,
}
//}}}
//{{{ enum: Error
#[derive(Error, Debug)]
pub enum Error
{
    #[error("Unconstrianed minimization failed with error {0}")]
    UnconstrainedError(#[from] UnconstrainedError),
    #[error("Maximum iterations of {0} reached")]
    MaxIterations(usize),
    #[error("Line Search Failed with error {0}")]
    LineSearch(#[from] LineSearchError),
}
//}}}
