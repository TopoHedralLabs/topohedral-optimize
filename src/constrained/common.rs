//! Shared options and errors for constrained optimization.
//!
//! These settings control feasibility and the inner solver's stopping criteria.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::bound_constrained::BoundConstrainedError;
use crate::common::BaseOptions;
use crate::unconstrained::UnconstrainedError;
use crate::ValidationError;
//}}}
//{{{ dep imports
use thiserror::Error;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct: Options
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
/// Common options for constrained methods.
pub struct Options {
    /// General optimizer stopping options.
    pub(crate) base_opts: BaseOptions,
    /// Feasibility tolerance for constraints.
    pub(crate) constraint_tol: f64,
}
//}}}
impl Options {
    /// Creates constrained options with a feasibility tolerance of `1e-8`.
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

    /// Returns the feasibility tolerance.
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

    /// Returns options with a different feasibility tolerance.
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
/// Errors reported by constrained optimization.
#[non_exhaustive]
pub enum Error {
    /// The unconstrained inner solve failed.
    #[error("unconstrained minimization failed: {0}")]
    UnconstrainedError(#[from] UnconstrainedError),
    /// The bound-constrained inner solve failed.
    #[error("bound-constrained minimization failed: {0}")]
    BoundConstrainedError(#[from] BoundConstrainedError),
    /// The outer iteration limit was reached.
    #[error("maximum number of iterations ({0}) reached")]
    MaxIterations(usize),
    /// Bounds were requested without a compatible inner solver.
    #[error("bounds require a bound-constrained inner solver")]
    BoundsWithoutBoundsSolver,
    /// An argument or optimizer option was invalid.
    #[error(transparent)]
    Validation(#[from] ValidationError),
}
//}}}
