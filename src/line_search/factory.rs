//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::*;
use super::interp;
use super::nocedal;
use super::thuente;
use crate::RealFn1;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

#[derive(Copy, Clone)]
pub enum Method
{
    Interp(interp::Options),
    Thuente(thuente::Options),
    Nocedal(nocedal::Options),
}

pub fn create<'a, F: RealFn1 + 'a>(
    fcn: F,
    method: Method,
) -> Box<dyn LineSearch<Function = F> + 'a>
{
    match method
    {
        Method::Interp(opts) => Box::new(interp::Interp { opts, f: fcn }),
        Method::Thuente(opts) => Box::new(thuente::Thuente::new(fcn, opts)),
        Method::Nocedal(opts) => Box::new(nocedal::Nocedal::new(fcn, opts)),
    }
}
