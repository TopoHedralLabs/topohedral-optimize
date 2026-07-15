//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::bound_constrained::BoundConstrainedError;
use crate::common::BaseOptions;
use crate::unconstrained::UnconstrainedError;
//}}}
//{{{ dep imports
use thiserror::Error;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct: Options
#[derive(Copy, Clone)]
pub struct Options {
    pub base_opts: BaseOptions,
    pub constraint_tol: f64,
}
//}}}

//{{{ enum: Error
#[derive(Error, Debug)]
pub enum Error {
    #[error("Unconstrianed minimization failed with error {0}")]
    UnconstrainedError(#[from] UnconstrainedError),
    #[error("Bound constrained minimization failed with error {0}")]
    BoundConstrainedError(#[from] BoundConstrainedError),
    #[error("Maximum iterations of {0} reached")]
    MaxIterations(usize),
    #[error("Cannot use bounds without inner bounded solver")]
    BoundsWithoutBoundsSolver,
}
//}}}
