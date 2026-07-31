//! Differentiable optimization algorithms for scalar and multidimensional problems.
//!
//! `topohedral-optimize` provides a common differentiable-function interface
//! and four families of algorithms:
//!
//! - scalar minimization with bounded, Brent, and golden-section methods;
//! - More–Thuente and Nocedal line searches;
//! - conjugate-gradient and quasi-Newton unconstrained minimization;
//! - ASA and BFGS-B bound-constrained minimization; and
//! - augmented-Lagrangian minimization with equality and inequality constraints.
//!
//! Optimizers borrow the objective mutably, which allows an objective to retain
//! caches and counters. Configuration values are checked before an algorithm
//! starts and invalid values are returned as structured [`ValidationError`]s.
//!
//! # Quick start
//!
//! Minimize \((x - 2)^2 + 1\) on a closed interval:
//!
//! ```
//! use topohedral_optimize::{
//!     scalar_minimize, BoundedOptions, DifferentiableFn, ScalarMethod,
//! };
//!
//! struct Parabola;
//!
//! impl DifferentiableFn for Parabola {
//!     type Input = f64;
//!     type Output = f64;
//!     type Derivative = f64;
//!
//!     fn eval(&mut self, x: &f64) -> f64 {
//!         (x - 2.0).powi(2) + 1.0
//!     }
//!
//!     fn derivative(&mut self, x: &f64) -> f64 {
//!         2.0 * (x - 2.0)
//!     }
//!
//!     fn dimension_domain(&self) -> usize {
//!         1
//!     }
//!
//!     fn dimension_range(&self) -> usize {
//!         1
//!     }
//! }
//!
//! let options = BoundedOptions::new(-5.0, 5.0)?;
//! let result = scalar_minimize(&mut Parabola, ScalarMethod::Bounded(options))?;
//!
//! assert!((result.xmin - 2.0).abs() < 1e-6);
//! assert!((result.fmin - 1.0).abs() < 1e-10);
//! # Ok::<(), topohedral_optimize::ScalarError>(())
//! ```
//!
//! See the crate's `examples` directory for vector-valued problem setups.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------
//{{{ mod: submodules
mod bound_constrained;
mod common;
mod constrained;
mod constraints;
mod line_search;
mod quadratic_model;
mod scalar;
mod unconstrained;
//}}}
//{{{ pub use: bound_constrained exports
pub use bound_constrained::{
    minimize as bound_constrained_minimize, AsaOptions, BfgsbOptions, BoundConstrainedError,
    BoundConstrainedMethod, BoundConstrainedOptions,
};
//}}}
//{{{ pub use: common exports
pub use common::{
    BaseOptions, ConvergedReason, DifferentiableFn, IterData, Matrix, RealFn, RealFn1,
    RealVectorFn, ScalarReturns, ValidationError, Vector, VectorReturns,
};
//}}}
//{{{ pub use: constrained
pub use constrained::{
    minimize as constrained_minimize, AugmentedLagrangianInnerMethod, AugmentedLagrangianOptions,
    ConstrainedError, ConstrainedMethod, ConstrainedOptions,
};
//}}}
//{{{ pub use constraints
pub use constraints::{
    BoundConstraints, BoundSide, BoundSignature, BoundStatus, CauchyPathPoint, NoConstraints,
};
/// Deprecated misspelling of [`BoundConstraints`].
#[deprecated(since = "0.0.0", note = "renamed to `BoundConstraints`")]
pub type BoundsConstraints = BoundConstraints;
//}}}
//{{{ pub use quadratic_model
pub use quadratic_model::{QuadraticModel, UpdateType};
//}}}
//{{{ pub use line_search
pub use line_search::{
    line_search, line_search_1d, LineSearchError, LineSearchMethod, LineSearchOptions,
    NocedalOptions, ThuenteOptions,
};
//}}}
//{{{ pub use scalar
pub use scalar::{
    bracket, minimize as scalar_minimize, BoundedOptions, Bracket, BracketOptions, BracketResult,
    BrentOptions, GoldenOptions, ScalarError, ScalarMethod,
};
//}}}
//{{{ pub use unconstrained
pub use unconstrained::{
    minimize as unconstrained_minimize, ConjugateGradientOptions,
    Direction as ConjugateGradientDirection, QuasiNewtonOptions, UnconstrainedError,
    UnconstrainedMethod, UnconstrainedOptions, UpdateMethod as QuasiNewtonUpdateMethod,
};

/// Deprecated misspelling of [`scalar_minimize`].
#[deprecated(since = "0.0.0", note = "renamed to `scalar_minimize`")]
pub use scalar::minimize as scalar_minimze;

/// Deprecated name for [`line_search`].
#[deprecated(since = "0.0.0", note = "renamed to `line_search`")]
pub use line_search::line_search as lsearch;

/// Deprecated name for [`line_search_1d`].
#[deprecated(since = "0.0.0", note = "renamed to `line_search_1d`")]
pub use line_search::line_search_1d as lsearch1d;
//}}}
