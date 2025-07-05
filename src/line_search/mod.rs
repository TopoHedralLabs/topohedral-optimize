//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports 
use crate::{RealFn, RealFn1};
//}}}
//{{{ std imports 
use std::ops::{Add, Mul};
//}}}
//{{{ dep imports 
use topohedral_linalg::VectorOps;
//}}}
//--------------------------------------------------------------------------------------------------

mod common;
mod nocedal;
mod utils;
mod factory;
mod interp;

pub use common::{LineSearchFcn, LineSearcher, Options, Returns, Error};
pub use factory::{Method, create};
pub use interp::Interp;
pub use interp::Options as InterpOptions;