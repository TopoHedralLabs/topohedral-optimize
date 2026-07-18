//! Shared options and errors for constrained optimization.
//!
//! These settings control feasibility and the inner solver's stopping criteria.
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
/// Common options for constrained methods.
pub struct Options {
    /// General optimizer stopping options.
    pub base_opts: BaseOptions,
    /// Feasibility tolerance for constraints.
    pub constraint_tol: f64,
}
//}}}

//{{{ enum: Error
#[derive(Error, Debug)]
/// Errors reported by constrained optimization.
pub enum Error {
    /// The unconstrained inner solve failed.
    #[error("Unconstrianed minimization failed with error {0}")]
    UnconstrainedError(#[from] UnconstrainedError),
    /// The bound-constrained inner solve failed.
    #[error("Bound constrained minimization failed with error {0}")]
    BoundConstrainedError(#[from] BoundConstrainedError),
    /// The outer iteration limit was reached.
    #[error("Maximum iterations of {0} reached")]
    MaxIterations(usize),
    /// Bounds were requested without a compatible inner solver.
    #[error("Cannot use bounds without inner bounded solver")]
    BoundsWithoutBoundsSolver,
}
//}}}
