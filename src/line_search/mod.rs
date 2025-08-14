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
mod factory;
mod interp;
mod nocedal;
mod utils;

pub use common::{
    Error as LineSearchError, LineSearchFcn, LineSearcher, Options as LineSearchOptions,
    Returns as LineSearchReturns,
};
pub use factory::{create, Method as LineSearchMethod};
pub use interp::Interp;
pub use interp::Options as InterpOptions;
