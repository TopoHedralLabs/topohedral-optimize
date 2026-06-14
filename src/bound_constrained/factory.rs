//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::asa::{ActiveSetAlgorithm, Options as AsaOptions};
use super::common::Error;
use crate::bound_constrained::BoundConstrainedOptions;
use crate::constraints::BoundsConstraints;
use crate::{Minimizer, RealFn, Vector};
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

impl Method
{
    pub fn bound_opts(&self) -> &BoundConstrainedOptions
    {
        match self
        {
            Method::Asa(asa_opts) => &asa_opts.bound_opts,
        }
    }

    pub fn bound_opts_mut(&mut self) -> &mut BoundConstrainedOptions
    {
        match self
        {
            Method::Asa(asa_opts) => &mut asa_opts.bound_opts,
        }
    }
}

#[trace_fn]
pub fn create<'a, F: RealFn + 'a>(
    fcn: F,
    bounds: BoundsConstraints,
    x0: Vector,
    method: Method,
) -> Box<dyn Minimizer<Error = Error> + 'a>
{
    match method
    {
        Method::Asa(opts) => Box::new(ActiveSetAlgorithm::new(fcn, bounds, x0, opts)),
    }
}
