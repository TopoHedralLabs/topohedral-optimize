//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::*;
use super::nocedal;
use super::thuente;
use crate::RealFn1;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum: Method
#[derive(Clone)]
pub enum Method {
    Thuente(thuente::Options),
    Nocedal(nocedal::Options),
}
//}}}
//{{{ fun: create
#[trace_fn]
pub fn create<'a, F: RealFn1 + 'a>(
    fcn: F,
    method: Method,
) -> Box<dyn LineSearch<Function = F> + 'a> {
    match method {
        Method::Thuente(opts) => Box::new(thuente::Thuente::new(fcn, opts)),
        Method::Nocedal(opts) => Box::new(nocedal::Nocedal::new(fcn, opts)),
    }
}
//}}}
