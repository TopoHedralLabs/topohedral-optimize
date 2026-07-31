//! Shared options and errors for bound-constrained optimization.
//!
//! Also contains helpers for lifting and restricting bound-constrained vectors.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::BaseOptions;
use crate::constraints::BoundStatus;
use crate::line_search::LineSearchError;
use crate::unconstrained::UnconstrainedError;
use crate::{ValidationError, Vector};
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
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
/// Common options for bound-constrained methods.
pub struct Options {
    /// General optimizer stopping options.
    pub(crate) base_opts: BaseOptions,
    /// Feasibility tolerance for bound constraints.
    pub(crate) constraint_tol: f64,
}
//}}}
impl Options {
    /// Creates bound-constrained options with a feasibility tolerance of
    /// `1e-8`.
    pub const fn new(base: BaseOptions) -> Self {
        Self {
            base_opts: base,
            constraint_tol: 1e-8,
        }
    }

    /// Returns the shared optimizer stopping options.
    pub const fn base(&self) -> &BaseOptions {
        &self.base_opts
    }

    /// Returns the constraint feasibility tolerance.
    pub const fn constraint_tolerance(&self) -> f64 {
        self.constraint_tol
    }

    /// Returns options with different shared stopping settings.
    pub const fn with_base(
        mut self,
        base: BaseOptions,
    ) -> Self {
        self.base_opts = base;
        self
    }

    /// Returns options with a different constraint tolerance.
    pub const fn with_constraint_tolerance(
        mut self,
        tolerance: f64,
    ) -> Self {
        self.constraint_tol = tolerance;
        self
    }

    /// Validates this configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError`] if a nested option is invalid or the
    /// constraint tolerance is non-positive or non-finite.
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.base_opts.validate()?;
        crate::common::validate_positive_finite("constraint_tolerance", self.constraint_tol)
    }
}

impl Default for Options {
    fn default() -> Self {
        Self::new(BaseOptions::default())
    }
}
//{{{ enum: Error
#[derive(Error, Debug)]
/// Errors reported by bound-constrained optimization.
#[non_exhaustive]
pub enum Error {
    /// An underlying unconstrained solve failed.
    #[error("unconstrained minimization failed: {0}")]
    UnconstrainedError(#[from] UnconstrainedError),
    /// The iteration limit was reached.
    #[error("maximum number of iterations ({0}) reached")]
    MaxIterations(usize),
    /// The line search failed.
    #[error("line search failed: {0}")]
    LineSearch(#[from] LineSearchError),
    /// An argument or optimizer option was invalid.
    #[error(transparent)]
    Validation(#[from] ValidationError),
}
//}}}

#[trace_fn]
pub fn lift(
    bound_statuses: &[BoundStatus],
    x_reduced: &Vector,
) -> Vector {
    let n = bound_statuses.len();
    let mut x_full = Vector::zeros_vec(n, VecType::Col);
    let mut local_index = 0;

    for (global_index, bound_status) in bound_statuses.iter().enumerate() {
        match bound_status {
            BoundStatus::Free => {
                x_full[global_index] = x_reduced[local_index];
                local_index += 1;
            }
            BoundStatus::AtLower(value) | BoundStatus::AtUpper(value) => {
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
) -> Vector {
    let n = bound_statuses
        .iter()
        .filter(|status| **status == BoundStatus::Free)
        .count();

    let mut x_reduced = Vector::zeros_vec(n, VecType::Col);
    let mut local_index = 0;

    for (global_index, bound_status) in bound_statuses.iter().enumerate() {
        match bound_status {
            BoundStatus::Free => {
                x_reduced[local_index] = x_full[global_index];
                local_index += 1;
            }
            _ => {
                continue;
            }
        }
    }
    x_reduced
}
