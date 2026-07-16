//! Module implementing a set of constrained optimisation algorithms
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{
    common::{Evaluator, RealFn, RealVectorFn, Vector, VectorReturns},
    constraints::BoundsConstraints,
};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ mod: submodules
mod augmented_lagrangian;
mod common;
mod factory;
//}}}

//{{{ pub use: common exports
pub use common::{Error as ConstrainedError, Options as ConstriainedOptions};
//}}}
//{{{ pub use: augmented_lagrangian exports
pub use augmented_lagrangian::{
    InnerMethod as AugmentedLagrangianInnerMethod, Options as AugmentedLagrangianOptions,
};
//}}}
//{{{ pub use: factory export
pub use factory::Method as ConstrainedMethod;
//}}}

//{{{ fn: minimize
#[trace_fn]
/// Minimizes a function subject to optional bounds and vector constraints.
pub fn minimize<F: RealFn + ?Sized>(
    fcn: &mut F,
    bounds: Option<BoundsConstraints>,
    eq_constraints: Option<&mut dyn RealVectorFn>,
    ieq_constraints: Option<&mut dyn RealVectorFn>,
    x0: Vector,
    method: ConstrainedMethod,
) -> Result<VectorReturns, ConstrainedError> {
    let mut evaluator = Evaluator::new(fcn);
    let mut ret = minimize_impl(
        &mut evaluator,
        bounds,
        eq_constraints,
        ieq_constraints,
        x0,
        method,
    )?;
    ret.num_fun_evals = evaluator.num_func_evals;
    ret.num_grad_evals = evaluator.num_grad_evals;
    Ok(ret)
}

pub(crate) fn minimize_impl<F: RealFn + ?Sized>(
    fcn: &mut F,
    bounds: Option<BoundsConstraints>,
    eq_constraints: Option<&mut dyn RealVectorFn>,
    ieq_constraints: Option<&mut dyn RealVectorFn>,
    x0: Vector,
    method: ConstrainedMethod,
) -> Result<VectorReturns, ConstrainedError> {
    let minimizer = factory::create(fcn, bounds, eq_constraints, ieq_constraints, x0, method);
    minimizer?.minimize()
}
//}}}
