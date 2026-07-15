//! Toppohedral-optimize is an optimisation library written entirely in rust. Part of the topohedral
//! collection of libraries.
//!
//! The library supports:
//! - Approximate Line-Search algorithms, contained in the [`line_search`] module:
//!     - More-Thuente
//!     - Nocedal
//! - multidimensional, unconstrained optimisation, contained the [`unconstrained`] module, and has
//!   the following algorithms:
//!     - Conjugate Gradient with a selection of direction implementations
//!     - Quasi-Newton with a selection of Heassian-update implementations
//! - multidimensionsional, constrained optimisation, contained in the [`constrained`] module with
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
    arc_real_vector_fn, rc_real_fn, rc_real_vector_fn, ArcRealVectorFn, BaseOptions,
    ConvergedReason, IterData, Matrix, RcRealFn, RcRealVectorFn, RealFn, RealFn1, RealVectorFn,
    ScalarReturns, Vector, VectorReturns,
};
//}}}
//{{{ pub use: constrained
pub use constrained::{
    minimize as constrained_minimize, AugmentedLagrangianInnerMethod, AugmentedLagrangianOptions,
    ConstrainedError, ConstrainedMethod, ConstriainedOptions,
};
//}}}
//{{{ pub use constraints
pub use constraints::{BoundsConstraints, NoConstraints};
//}}}
//{{{ pub use line_search
pub use line_search::{
    search as lsearch, search1d as lsearch1d, LineSearchError, LineSearchMethod, LineSearchOptions,
};
//}}}
//{{{ pub use scalar
pub use scalar::{
    bracket, minimize as scalar_minimze, BoundedOptions, BracketOptions, BracketResult,
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
