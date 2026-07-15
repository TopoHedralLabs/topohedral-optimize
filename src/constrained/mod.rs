//! Module implementing a set of constrained optimisation algorithms
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{
    common::{arc_real_fn, CountingRealFn, RealFn, RealVectorFn, VectorReturns, Vector},
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
    AugmentedLagrangian, AugmentedLagrangianFcn, InnerMethod as AugmentedLagrangianInnerMethod,
    Options as AugmentedLagrangianOptions,
};
//}}}
//{{{ pub use: factory export
pub use factory::Method as ConstrainedMethod;
//}}}

//{{{ fn: minimize
#[trace_fn]
pub fn minimize<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn>(
    fcn: F1,
    bounds: Option<BoundsConstraints>,
    eq_constraints: Option<F2>,
    ieq_constraints: Option<F3>,
    x0: Vector,
    method: ConstrainedMethod,
) -> Result<VectorReturns, ConstrainedError> {
    if method.con_opts().base_opts.make_counting {
        let counting_fcn = arc_real_fn(CountingRealFn::new(fcn));
        let minimizer = factory::create(
            counting_fcn.clone(),
            bounds,
            eq_constraints,
            ieq_constraints,
            x0,
            method,
        );
        let mut ret = minimizer?.minimize()?;
        let counting_fcn_lock = counting_fcn.lock().unwrap();
        ret.num_fun_evals = counting_fcn_lock.num_func_evals;
        ret.num_grad_evals = counting_fcn_lock.num_grad_evals;
        Ok(ret)
    } else {
        let minimizer = factory::create(fcn, bounds, eq_constraints, ieq_constraints, x0, method);
        minimizer?.minimize()
    }
}
//}}}
