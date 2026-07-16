//! Toppohedral-optimize is an optimisation library written entirely in rust. Part of the topohedral
//! collection of libraries.
//!
//! The library supports:
//! - Approximate line-search algorithms, configured with [`LineSearchMethod`]:
//!     - More-Thuente
//!     - Nocedal
//! - multidimensional, unconstrained optimisation, configured with [`UnconstrainedMethod`], and has
//!   the following algorithms:
//!     - Conjugate Gradient with a selection of direction implementations
//!     - Quasi-Newton with a selection of Heassian-update implementations
//! - multidimensional, constrained optimisation, configured with [`ConstrainedMethod`], with
//!   the following algorithms:
//!     - Augmented Lagrangian Method.
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
    RealVectorFn, ScalarReturns, Vector, VectorReturns,
};
//}}}
//{{{ pub use: constrained
pub use constrained::{
    minimize as constrained_minimize, AugmentedLagrangianInnerMethod, AugmentedLagrangianOptions,
    ConstrainedError, ConstrainedMethod, ConstriainedOptions,
};
//}}}
//{{{ pub use constraints
pub use constraints::{
    BoundSide, BoundSignature, BoundStatus, BoundsConstraints, CauchyPathPoint, NoConstraints,
};
//}}}
//{{{ pub use quadratic_model
pub use quadratic_model::{QuadraticModel, UpdateType};
//}}}
//{{{ pub use line_search
pub use line_search::{
    search as lsearch, search1d as lsearch1d, LineSearchError, LineSearchMethod, LineSearchOptions,
    NocedalOptions, ThuenteOptions,
};
//}}}
//{{{ pub use scalar
pub use scalar::{
    bracket, minimize as scalar_minimze, BoundedOptions, Bracket, BracketOptions, BracketResult,
    BrentOptions, GoldenOptions, ScalarError, ScalarMethod,
};
//}}}
//{{{ pub use unconstrained
pub use unconstrained::{
    minimize as unconstrained_minimize, ConjugateGradientOptions,
    Direction as ConjugateGradientDirection, QuasiNewtonOptions, UnconstrainedError,
    UnconstrainedMethod, UnconstrainedOptions, UpdateMethod as QuasiNewtonUpdateMethod,
};
//}}}
