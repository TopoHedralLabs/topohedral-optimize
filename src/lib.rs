//! Toppohedral-optimize is an optimisation library written entirely in rust. Part of the topohedral
//! collection of libraries.
//!
//! The library supports:
//! - Approximate Line-Search algorithms, contained in the [`line_search`] module:
//!     - More-Thuente
//!     - Nocedal
//!     - Backtracking + polynomial interpolation
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
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

//{{{ mod: submodules
mod common;
//}}}
//{{{ pub use: common exports
pub use common::{
    arc_real_vector_fn, rc_real_fn, rc_real_vector_fn, ArcRealVectorFn, ConvergedReason, IterData,
    Matrix, RcRealFn, RcRealVectorFn, RealFn, RealFn1, RealVectorFn, Returns, Vector,
};
//}}}
//{{{ pub mod: public modules
pub mod constrained;
pub mod line_search;
pub mod unconstrained;
//}}}
