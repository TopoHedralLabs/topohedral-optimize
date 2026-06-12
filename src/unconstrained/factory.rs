//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

use topohedral_tracing::trace_fn;

//{{{ crate imports
use super::common::{Error, Options};
use super::conjugate_gradient::ConjugateGradient;
use super::conjugate_gradient::Options as ConjugateGradientOptions;
use super::quasi_newton::Options as QuasiNewtonOptions;
use super::quasi_newton::QuasiNewton;
use crate::{Minimizer, RealFn, Vector};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum: Method
#[derive(Clone)]
pub enum Method
{
    ConjugateGradient(ConjugateGradientOptions),
    QuasiNewton(QuasiNewtonOptions),
}
//}}}
//{{{ impl: Method
impl Method
{
    #[trace_fn]
    pub fn uncon_opts_mut(&mut self) -> &mut Options
    {
        match self
        {
            Method::ConjugateGradient(cg_opts) => &mut cg_opts.uncon_opts,
            Method::QuasiNewton(qn_opts) => &mut qn_opts.uncon_opts,
        }
    }

    #[trace_fn]
    pub fn uncon_opts(&self) -> &Options
    {
        match self
        {
            Method::ConjugateGradient(cg_opts) => &cg_opts.uncon_opts,
            Method::QuasiNewton(qn_opts) => &qn_opts.uncon_opts,
        }
    }
}
//}}}
//{{{ fun: create
#[trace_fn]
pub fn create<'a, F: RealFn + 'a>(
    fcn: F,
    x0: Vector,
    method: Method,
) -> Box<dyn Minimizer<Error = Error> + 'a>
{
    match method
    {
        Method::ConjugateGradient(opts) => Box::new(ConjugateGradient::new(fcn, x0, opts)),
        Method::QuasiNewton(opts) => Box::new(QuasiNewton::new(fcn, x0, opts)),
    }
}
//}}}
