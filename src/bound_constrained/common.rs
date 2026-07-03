//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::BaseOptions;
use crate::constraints::BoundStatus;
use crate::line_search::LineSearchError;
use crate::unconstrained::UnconstrainedError;
use crate::Vector;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use thiserror::Error;
use topohedral_linalg::VecType;
use topohedral_tracing::trace_fn;
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

#[trace_fn]
pub fn lift(
    bound_statuses: &[BoundStatus],
    x_reduced: &Vector,
) -> Vector
{
    let n = bound_statuses.len();
    let mut x_full = Vector::zeros_vec(n, VecType::Col);
    let mut local_index = 0;

    for (global_index, bound_status) in bound_statuses.iter().enumerate()
    {
        match bound_status
        {
            BoundStatus::Free =>
            {
                x_full[global_index] = x_reduced[local_index];
                local_index += 1;
            }
            BoundStatus::AtLower(value) | BoundStatus::AtUpper(value) =>
            {
                x_full[global_index] = *value
            }
        }
    }
    x_full
}

#[trace_fn]
pub fn restrict(
    bound_statuses: &[BoundStatus],
    x_full: &Vector,
) -> Vector
{
    let n = bound_statuses
        .iter()
        .filter(|status| **status == BoundStatus::Free)
        .count();

    let mut x_reduced = Vector::zeros_vec(n, VecType::Col);
    let mut local_index = 0;

    for (global_index, bound_status) in bound_statuses.iter().enumerate()
    {
        match bound_status
        {
            BoundStatus::Free =>
            {
                x_reduced[local_index] = x_full[global_index];
                local_index += 1;
            }
            _ =>
            {
                continue;
            }
        }
    }
    x_reduced
}
