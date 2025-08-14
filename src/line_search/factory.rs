//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

use topohedral_linalg::VectorOps;

//{{{ crate imports 
use crate::RealFn1;
use super::common::*;
use super::interp;
//}}}
//{{{ std imports 
use std::ops::{Mul, Add};
//}}}
//{{{ dep imports 
//}}}
//--------------------------------------------------------------------------------------------------


#[derive(Copy, Clone)]
pub enum Method{
    Interp(interp::Options)
} 

pub fn create<'a, F: RealFn1 + 'a>(fcn: F, method: Method) 
-> Box<dyn LineSearcher<Function = F> + 'a>
{
    match method {
        Method::Interp(opts) => {
            Box::new(interp::Interp{
                opts: opts, 
                f: fcn 
            })
        }
    }
}