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
//{{{ fun: create
#[trace_fn]
pub fn create<'a, F1: RealFn + 'a, F2: RealVectorFn + 'a, F3: RealVectorFn + 'a>(
    fcn: F1,
    bounds: Option<BoundsConstraints>,
    eq_constraints: Option<F2>,
    ieq_constraints: Option<F3>,
    x0: Vector,
    method: Method,
) -> Result<Box<dyn Minimizer<Error = Error> + 'a>, Error>
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
