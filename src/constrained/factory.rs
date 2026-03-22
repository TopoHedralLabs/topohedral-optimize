//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

use topohedral_tracing::trace_fn;

//{{{ crate imports
use crate::{
    constrained::{
        augmented_lagrangian::AugmentedLagrangian, common::ConstrainedMinimizer,
        AugmentedLagrangianOptions,
    },
    RealFn, RealVectorFn, Vector,
};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum: Method
#[derive(Copy, Clone)]
pub enum Method
{
    AugmentedLagrangian(AugmentedLagrangianOptions),
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
) -> Box<dyn ConstrainedMinimizer + 'a>
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
