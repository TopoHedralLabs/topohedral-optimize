//! Factory for scalar minimizer implementations.
//!
//! The factory selects bounded, Brent, or golden-section minimization.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::Error;
use super::{
    bounded::Bounded, bounded::Options as BoundedOptions, brent::Brent,
    brent::Options as BrentOptions, golden::Golden, golden::Options as GoldenOptions,
};
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
/// Selects a scalar minimization algorithm.
pub enum Method {
    /// Bounded minimization.
    Bounded(BoundedOptions),
    /// Brent's method.
    Brent(BrentOptions),
    /// Golden-section search.
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
