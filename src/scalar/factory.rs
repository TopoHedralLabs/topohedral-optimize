//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::Error;
use super::{Bounded, BoundedOptions, Brent, BrentOptions, Golden, GoldenOptions};
use crate::common::{Minimizer, ScalarReturns};
use crate::RealFn1;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::trace_fn;
//}}}
//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub enum Method {
    Bounded(BoundedOptions),
    Brent(BrentOptions),
    Golden(GoldenOptions),
}

#[trace_fn]
pub fn create<'a, F: RealFn1 + 'a>(
    fcn: F,
    method: Method,
) -> Result<Box<dyn Minimizer<Error = Error, Returns = ScalarReturns> + 'a>, Error> {
    match method {
        Method::Bounded(opts) => Ok(Box::new(Bounded::new(fcn, opts))),
        Method::Brent(opts) => Ok(Box::new(Brent::new(fcn, opts))),
        Method::Golden(opts) => Ok(Box::new(Golden::new(fcn, opts))),
    }
}
