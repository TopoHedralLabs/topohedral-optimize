//! Shared function traits, optimizer results, and iteration bookkeeping.
//!
//! These types form the common interface used by all optimization families.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
use std::fmt::{self, Display, Formatter};
//}}}
//{{{ dep imports
use thiserror::Error;
use topohedral_linalg::DMatrix;
use topohedral_linalg::DVector;
use topohedral_linalg::VectorOps;
use topohedral_tracing::trace_fn;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ type: core aliases
/// Column vector of `f64` values used by the optimization routines.
pub type Vector = DVector<f64>;
/// Dense matrix of `f64` values used by the optimization routines.
pub type Matrix = DMatrix<f64>;
//}}}
//{{{ trait: DifferentiableFn
/// A differentiable function with associated input, output, and derivative types.
pub trait DifferentiableFn {
    /// Function input type.
    type Input;
    /// Function output type.
    type Output;
    /// Derivative type.
    type Derivative;

    /// Evaluates the function at `x`.
    fn eval(
        &mut self,
        x: &Self::Input,
    ) -> Self::Output;

    /// Evaluates the derivative at `x`.
    fn derivative(
        &mut self,
        x: &Self::Input,
    ) -> Self::Derivative;

    /// Returns the dimension of the input space.
    fn dimension_domain(&self) -> usize;

    /// Returns the dimension of the output space.
    fn dimension_range(&self) -> usize;
}
//}}}
//{{{ enum: ValidationError
/// Describes an invalid public API argument or optimizer configuration value.
///
/// Optimizer entry points validate their complete configuration before doing
/// any work, so configuration mistakes are returned as ordinary errors rather
/// than surfacing later as arithmetic failures or incidental panics.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum ValidationError {
    /// A floating-point parameter did not satisfy its documented requirement.
    #[error("invalid value {value} for {parameter}: {requirement}")]
    InvalidFloat {
        /// Parameter name.
        parameter: &'static str,
        /// Rejected value.
        value: f64,
        /// Human-readable requirement.
        requirement: &'static str,
    },
    /// An integer parameter did not satisfy its documented requirement.
    #[error("invalid value {value} for {parameter}: {requirement}")]
    InvalidInteger {
        /// Parameter name.
        parameter: &'static str,
        /// Rejected value.
        value: u64,
        /// Human-readable requirement.
        requirement: &'static str,
    },
    /// Two dimensions that must agree were different.
    #[error("{parameter} has dimension {actual}, expected {expected}")]
    DimensionMismatch {
        /// Argument or result whose dimension was invalid.
        parameter: &'static str,
        /// Required dimension.
        expected: usize,
        /// Supplied dimension.
        actual: usize,
    },
    /// A bound was assigned to an out-of-range variable.
    #[error("bound index {index} is out of range for {num_variables} variables")]
    BoundIndexOutOfRange {
        /// Rejected variable index.
        index: usize,
        /// Number of variables in the bound set.
        num_variables: usize,
    },
    /// A variable already had a bound entry.
    #[error("bounds for variable {index} have already been added")]
    DuplicateBound {
        /// Duplicate variable index.
        index: usize,
    },
    /// A bound entry specified neither a lower nor an upper bound.
    #[error("variable {index} must have a lower bound, an upper bound, or both")]
    EmptyBound {
        /// Variable index.
        index: usize,
    },
    /// A lower bound exceeded its corresponding upper bound.
    #[error("lower bound {lower} exceeds upper bound {upper} for variable {index}")]
    InvalidBounds {
        /// Variable index.
        index: usize,
        /// Rejected lower bound.
        lower: f64,
        /// Rejected upper bound.
        upper: f64,
    },
}
//}}}
//{{{ impl: DifferentiableFn for &mut F
impl<F: DifferentiableFn + ?Sized> DifferentiableFn for &mut F {
    type Input = F::Input;
    type Output = F::Output;
    type Derivative = F::Derivative;

    fn eval(
        &mut self,
        x: &Self::Input,
    ) -> Self::Output {
        (**self).eval(x)
    }

    fn derivative(
        &mut self,
        x: &Self::Input,
    ) -> Self::Derivative {
        (**self).derivative(x)
    }

    fn dimension_domain(&self) -> usize {
        (**self).dimension_domain()
    }

    fn dimension_range(&self) -> usize {
        (**self).dimension_range()
    }
}
//}}}
//{{{ trait: RealFn1
/// Stable-Rust equivalent of a trait alias for a differentiable `f64 -> f64` function.
pub trait RealFn1: DifferentiableFn<Input = f64, Output = f64, Derivative = f64> {}

impl<F> RealFn1 for F where F: DifferentiableFn<Input = f64, Output = f64, Derivative = f64> {}
//}}}
//{{{ trait: RealFn
/// Stable-Rust equivalent of a trait alias for a scalar-valued function on [`Vector`].
///
/// Unlike earlier versions of this crate, this trait does not require
/// [`Debug`](std::fmt::Debug), so callers may use closures and other lightweight
/// objective wrappers without adding an unrelated formatting implementation.
pub trait RealFn: DifferentiableFn<Input = Vector, Output = f64, Derivative = Vector> {}

impl<F> RealFn for F where
    F: DifferentiableFn<Input = Vector, Output = f64, Derivative = Vector> + ?Sized
{
}
//}}}
//{{{ enum: ConvergedReason
/// Criterion that caused an optimization run to converge.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConvergedReason {
    /// Relative gradient tolerance was met.
    Rtol,
    /// Absolute gradient tolerance was met.
    Atol,
}
//}}}
//{{{ struct: Options
/// Common stopping options for vector optimization methods.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BaseOptions {
    /// Relative gradient tolerance.
    pub(crate) grad_rtol: f64,
    /// Absolute gradient tolerance.
    pub(crate) grad_atol: f64,
    /// Maximum number of iterations.
    pub(crate) max_iter: u64,
}
//}}}
//{{{ impl: BaseOptions
impl BaseOptions {
    /// Creates common stopping options.
    ///
    /// Values are checked by [`Self::validate`] and by every optimizer entry
    /// point. The `const` constructor makes reusable method configurations
    /// possible without exposing the representation.
    pub const fn new(
        grad_rtol: f64,
        grad_atol: f64,
        max_iter: u64,
    ) -> Self {
        Self {
            grad_rtol,
            grad_atol,
            max_iter,
        }
    }

    /// Returns the relative gradient tolerance.
    pub const fn grad_rtol(&self) -> f64 {
        self.grad_rtol
    }

    /// Returns the absolute gradient tolerance.
    pub const fn grad_atol(&self) -> f64 {
        self.grad_atol
    }

    /// Returns the maximum number of iterations.
    pub const fn max_iter(&self) -> u64 {
        self.max_iter
    }

    /// Returns options with a different relative gradient tolerance.
    pub const fn with_grad_rtol(
        mut self,
        grad_rtol: f64,
    ) -> Self {
        self.grad_rtol = grad_rtol;
        self
    }

    /// Returns options with a different absolute gradient tolerance.
    pub const fn with_grad_atol(
        mut self,
        grad_atol: f64,
    ) -> Self {
        self.grad_atol = grad_atol;
        self
    }

    /// Returns options with a different iteration limit.
    pub const fn with_max_iter(
        mut self,
        max_iter: u64,
    ) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Validates the stopping options.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError`] when a tolerance is negative or non-finite,
    /// or when `max_iter` is zero.
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_nonnegative_finite("grad_rtol", self.grad_rtol)?;
        validate_nonnegative_finite("grad_atol", self.grad_atol)?;
        validate_nonzero("max_iter", self.max_iter)?;
        Ok(())
    }
}

impl Default for BaseOptions {
    fn default() -> Self {
        Self::new(1e-6, 1e-8, 1_000)
    }
}
//}}}
//{{{ struct Returns
/// Result and convergence information returned by an optimizer.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Returns<T> {
    /// Minimizing point.
    pub xmin: T,
    /// Function value at the minimizing point.
    pub fmin: f64,
    /// Convergence criterion that was met.
    pub reason: ConvergedReason,
    /// Number of optimizer iterations.
    pub num_iterations: usize,
    /// Number of function evaluations.
    pub num_fun_evals: usize,
    /// Number of derivative evaluations.
    pub num_grad_evals: usize,
}
//}}}
//{{{ type: Returns aliases
/// Scalar optimization result.
pub type ScalarReturns = Returns<f64>;
/// Vector optimization result.
pub type VectorReturns = Returns<Vector>;
//}}}
//{{{ trait: Minimizer
pub(crate) trait Minimizer {
    type Error;
    type Returns;

    fn minimize(&mut self) -> Result<Self::Returns, Self::Error>;
}
//}}}
//{{{ struct: IterData
/// Function and gradient data for one optimization iterate.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone)]
pub struct IterData {
    /// Current point.
    pub x: Vector,
    /// Function value at the current point.
    pub fx: f64,
    /// Current gradient.
    pub grad_fx: Vector,
    /// Euclidean norm of the current gradient.
    pub norm_grad_fx: f64,
}
//}}}
//{{{ impl: IterData
impl IterData {
    /// Evaluates a function and derivative at `x`.
    #[trace_fn]
    pub fn new<F: RealFn + ?Sized>(
        fcn: &mut F,
        x: &Vector,
    ) -> Self {
        let fx = fcn.eval(x);
        let grad_fx = fcn.derivative(x);
        let norm_grad_fx = grad_fx.norm();
        IterData {
            x: x.clone(),
            fx,
            grad_fx,
            norm_grad_fx,
        }
    }

    /// Copies another iterate's values into this record.
    #[trace_fn]
    pub fn copy_from(
        &mut self,
        iter_data: &Self,
    ) {
        self.x.copy_from(&iter_data.x);
        self.fx = iter_data.fx;
        self.grad_fx.copy_from(&iter_data.grad_fx);
        self.norm_grad_fx = iter_data.norm_grad_fx;
    }
}
//}}}
//{{{ impl: Display for IterData
impl Display for IterData {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        let fx = self.fx;
        let norm_grad_fx = self.norm_grad_fx;
        let out = format!("fx={fx:.4e}, norm_grad_fx={norm_grad_fx:.4e}");
        f.pad(&out)
    }
}
//}}}
//{{{ struct: Evaluator
#[derive(Debug)]
pub(crate) struct Evaluator<'a, F: RealFn + ?Sized> {
    fcn: &'a mut F,
    pub num_func_evals: usize,
    pub num_grad_evals: usize,
}
//}}}
//{{{ impl: DifferentiableFn for Evaluator
impl<F: RealFn + ?Sized> DifferentiableFn for Evaluator<'_, F> {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;

    #[trace_fn]
    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        self.num_func_evals += 1;
        self.fcn.eval(x)
    }

    #[trace_fn]
    fn derivative(
        &mut self,
        x: &Vector,
    ) -> Vector {
        self.num_grad_evals += 1;
        self.fcn.derivative(x)
    }

    #[trace_fn]
    fn dimension_domain(&self) -> usize {
        self.fcn.dimension_domain()
    }

    fn dimension_range(&self) -> usize {
        self.fcn.dimension_range()
    }
}
//}}}
//{{{ impl: Evaluator
impl<'a, F: RealFn + ?Sized> Evaluator<'a, F> {
    #[trace_fn]
    pub fn new(fcn: &'a mut F) -> Self {
        Self {
            fcn,
            num_func_evals: 0,
            num_grad_evals: 0,
        }
    }
}
//}}}
//{{{ trait: RealVectorFn
/// Stable-Rust equivalent of a trait alias for a vector-valued function on [`Vector`].
pub trait RealVectorFn:
    DifferentiableFn<Input = Vector, Output = Vector, Derivative = Matrix>
{
}

impl<F> RealVectorFn for F where
    F: DifferentiableFn<Input = Vector, Output = Vector, Derivative = Matrix> + ?Sized
{
}
//}}}

pub(crate) fn validate_nonnegative_finite(
    parameter: &'static str,
    value: f64,
) -> Result<(), ValidationError> {
    if !value.is_finite() || value < 0.0 {
        return Err(ValidationError::InvalidFloat {
            parameter,
            value,
            requirement: "must be finite and non-negative",
        });
    }
    Ok(())
}

pub(crate) fn validate_positive_finite(
    parameter: &'static str,
    value: f64,
) -> Result<(), ValidationError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(ValidationError::InvalidFloat {
            parameter,
            value,
            requirement: "must be finite and greater than zero",
        });
    }
    Ok(())
}

pub(crate) fn validate_nonzero(
    parameter: &'static str,
    value: u64,
) -> Result<(), ValidationError> {
    if value == 0 {
        return Err(ValidationError::InvalidInteger {
            parameter,
            value,
            requirement: "must be greater than zero",
        });
    }
    Ok(())
}
