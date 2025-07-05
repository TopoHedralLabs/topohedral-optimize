//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

use topohedral_linalg::VectorOps;

//{{{ crate imports 
use crate::RealFn;
use super::common::*;
use super::interp;
//}}}
//{{{ std imports 
use std::ops::{Mul, Add};
//}}}
//{{{ dep imports 
//}}}
//--------------------------------------------------------------------------------------------------

pub enum Method{
    Interp(interp::Options)
} 

pub fn create<'a, F: RealFn + 'a>(x: F::Vector, dir: F::Vector, obj_fcn: F, method: Method) 
-> Box<dyn LineSearcher + 'a>
where 
    F::Vector: VectorOps<ScalarType = f64>,
    F::Vector: Add<F::Vector, Output = F::Vector>,
    f64: Mul<F::Vector, Output = F::Vector>,
{

    let search_fcn = LineSearchFcn { f: obj_fcn, x: x, dir: dir };

    match method {
        Method::Interp(opts) => {
            Box::new(interp::Interp{
                opts: opts, 
                f: search_fcn
            })
        }
    }
}