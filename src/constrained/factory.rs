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
    unconstrained::UnconstrainedMethod,
    Minimizer, RealFn, RealVectorFn, Vector,
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
//{{{ impl: Method
impl Method
{
    #[trace_fn]
    pub fn uncon_method_mut(&mut self) -> &mut UnconstrainedMethod
    {
        match self
        {
            Method::AugmentedLagrangian(aut_opts) => aut_opts.uncon_method_mut(),
        }
    }

    #[trace_fn]
    pub fn uncon_method(&self) -> &UnconstrainedMethod
    {
        match self
        {
            Method::AugmentedLagrangian(aut_opts) => aut_opts.uncon_method(),
        }
    }

    #[trace_fn]
    pub fn con_opts_mut(&mut self) -> &mut ConstriainedOptions
    {
        match self
        {
            Method::AugmentedLagrangian(aut_opts) => &mut aut_opts.constrained_opts,
        }
    }

    #[trace_fn]
    pub fn con_opts(&self) -> &ConstriainedOptions
    {
        match self
        {
            Method::AugmentedLagrangian(aut_opts) => &aut_opts.constrained_opts,
        }
    }
}
//}}}
//{{{ fun: create
#[trace_fn]
pub fn create<'a, F1: RealFn + 'a, F2: RealVectorFn + 'a, F3: RealVectorFn + 'a>(
    fcn: F1,
    eq_constraints: Option<F2>,
    ieq_constraints: Option<F3>,
    x0: Vector,
    method: Method,
) -> Box<dyn Minimizer<Error = Error> + 'a>
{
    match method
    {
        Method::AugmentedLagrangian(opts) => Box::new(AugmentedLagrangian::new(
            fcn,
            eq_constraints,
            ieq_constraints,
            x0,
            opts,
        )),
    }
}

//}}}
