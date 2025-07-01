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
mod interp;

pub use common::*;