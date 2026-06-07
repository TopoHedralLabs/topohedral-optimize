//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::asa::{ActiveSetAlgorithm, Options as AsaOptions};
use super::common::BoundConstrainedMinimizer;
use crate::constraints::BoundsConstraints;
use crate::{RealFn, Vector};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::trace_fn;
//}}}
//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub enum Method
{
    Asa(AsaOptions),
}

#[trace_fn]
pub fn create<'a, F: RealFn + 'a>(
    mut fcn: F,
    mut x0: Vector,
    bounds: BoundsConstraints,
    method: Method,
) -> Box<dyn BoundConstrainedMinimizer + 'a>
{
    match method
    {
        Method::Asa(opts) => Box::new(ActiveSetAlgorithm::new(fcn, x0, bounds, opts)),
    }
}
