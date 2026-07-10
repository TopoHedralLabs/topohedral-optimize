//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{
    constrained::{
        augmented_lagrangian::AugmentedLagrangian, common::Error, AugmentedLagrangianOptions,
        ConstriainedOptions,
    },
    constraints::BoundsConstraints,
    Minimizer, RealFn, RealVectorFn, Returns, Vector,
};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::trace_fn;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum: Method
#[derive(Clone)]
pub enum Method
{
    AugmentedLagrangian(AugmentedLagrangianOptions),
}
//}}}
//{{{ impl Method
impl Method
{
    pub fn con_opts(&self) -> &ConstriainedOptions
    {
        match self
        {
            Method::AugmentedLagrangian(opts) => &opts.constrained_opts,
        }
    }

    pub fn con_opts_mut(&mut self) -> &mut ConstriainedOptions
    {
        match self
        {
            Method::AugmentedLagrangian(opts) => &mut opts.constrained_opts,
        }
    }
}
//}}}

//{{{ fun: create
#[trace_fn]
pub fn create<'a, F1: RealFn + 'a, F2: RealVectorFn + 'a, F3: RealVectorFn + 'a>(
    fcn: F1,
    bounds: Option<BoundsConstraints>,
    eq_constraints: Option<F2>,
    ieq_constraints: Option<F3>,
    x0: Vector,
    method: Method,
) -> Result<Box<dyn Minimizer<Error = Error, Returns = Returns> + 'a>, Error>
{
    match method
    {
        Method::AugmentedLagrangian(opts) => Ok(Box::new(AugmentedLagrangian::new(
            fcn,
            bounds,
            eq_constraints,
            ieq_constraints,
            x0,
            opts,
        ))),
    }
}

//}}}
