//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------


//{{{ crate imports 
use crate::RealFn1;
use super::common::*;
use super::interp;
//}}}
//{{{ std imports 
//}}}
//{{{ dep imports 
//}}}
//--------------------------------------------------------------------------------------------------


#[derive(Copy, Clone)]
pub enum Method{
    Interp(interp::Options)
} 

pub fn create<'a, F: RealFn1 + 'a>(fcn: F, method: Method) 
-> Box<dyn LineSearch<Function = F> + 'a>
{
    match method {
        Method::Interp(opts) => {
            Box::new(interp::Interp{
                opts, 
                f: fcn 
            })
        }
    }
}