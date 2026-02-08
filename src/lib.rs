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
pub use common::{rc_real_fn, RcRealFn, RealFn, RealFn1};
pub mod line_search;
pub mod unconstrained;
