//! Short Description of module
//!
//! Longer description of module
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

mod common;
pub use common::{
    arc_real_vector_fn, rc_real_fn, rc_real_vector_fn, ArcRealVectorFn, RcRealFn, RcRealVectorFn,
    RealFn, RealFn1, RealVectorFn, Vector,
};
pub mod constrained;
pub mod line_search;
pub mod unconstrained;
