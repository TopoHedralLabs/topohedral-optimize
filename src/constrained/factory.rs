//! Factory for constrained optimizer implementations.
//!
//! It assembles the selected outer method and its optional constraint functions.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::Minimizer;
use crate::{
    constrained::{
        augmented_lagrangian::AugmentedLagrangian, common::Error, AugmentedLagrangianOptions,
        ConstriainedOptions,
    },
    constraints::BoundsConstraints,
    RealFn, RealVectorFn, Vector, VectorReturns,
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
/// Selects a constrained optimization algorithm and its options.
pub enum Method {
    /// Augmented-Lagrangian method.
    AugmentedLagrangian(AugmentedLagrangianOptions),
}
//}}}
//{{{ impl Method
impl Method {
    /// Returns the shared constrained options.
    pub fn con_opts(&self) -> &ConstriainedOptions {
        match self {
            Method::AugmentedLagrangian(opts) => &opts.constrained_opts,
        }
    }

    /// Returns mutable access to the shared constrained options.
    pub fn con_opts_mut(&mut self) -> &mut ConstriainedOptions {
        match self {
            Method::AugmentedLagrangian(opts) => &mut opts.constrained_opts,
        }
    }
}
//}}}

//{{{ fun: create
#[trace_fn]
/// Constructs the selected constrained optimizer.
pub fn create<'a, F1: RealFn + 'a, F2: RealVectorFn + 'a, F3: RealVectorFn + 'a>(
    fcn: F1,
    bounds: Option<BoundsConstraints>,
    eq_constraints: Option<F2>,
    ieq_constraints: Option<F3>,
    x0: Vector,
    method: Method,
) -> Result<Box<dyn Minimizer<Error = Error, Returns = VectorReturns> + 'a>, Error> {
    match method {
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
