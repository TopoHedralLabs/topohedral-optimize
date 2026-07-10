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
mod common;
//}}}
//{{{ pub use: common exports
pub use common::{
    arc_real_vector_fn, rc_real_fn, rc_real_vector_fn, ArcRealVectorFn, BaseOptions,
    ConvergedReason, IterData, Matrix, Minimizer, RcRealFn, RcRealVectorFn, RealFn, RealFn1,
    RealVectorFn, Returns, Vector,
};
//}}}
//{{{ pub mod: public modules
pub mod bound_constrained;
pub mod constrained;
pub mod constraints;
pub mod line_search;
pub mod quadratic_model;
pub mod scalar;
pub mod unconstrained;
//}}}
