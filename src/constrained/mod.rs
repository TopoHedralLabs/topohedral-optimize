//! Module implementing a set of constrained optimisation algorithms
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::{arc_real_fn, CountingRealFn, RealFn, RealVectorFn, Vector};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ mod: submodules
mod augmented_lagrangian;
mod common;
mod constraints;
mod factory;
//}}}

//{{{ pub use: common exports
pub use common::{
    ConstrainedMinimizer, ConvergedReason as ConstrainedConvergedReason, Error as ConstrainedError,
    Options as ConstriainedOptions, Returns as ConstrainedReturns,
};
//}}}
//{{{ pub use: augmented_lagrangian exports
pub use augmented_lagrangian::{
    AugmentedLagrangian, AugmentedLagrangianFcn, Options as AugmentedLagrangianOptions,
};
//}}}
//{{{ pub use: factory export
pub use factory::Method as ConstrainedMethod;

//}}}

//{{{ fn: minimize
pub fn minimize<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn>(
    fcn: F1,
    eq_constraints: Option<F2>,
    ieq_constraints: Option<F3>,
    x0: Vector,
    method: ConstrainedMethod,
) -> Result<ConstrainedReturns, ConstrainedError>
{
    if method.con_opts().make_counting
    {
        let counting_fcn = arc_real_fn(CountingRealFn::new(fcn));
        let mut minimizer = factory::create(
            counting_fcn.clone(),
            eq_constraints,
            ieq_constraints,
            x0,
            method,
        );
        let mut ret = minimizer.minimize()?;
        let counting_fcn_lock = counting_fcn.lock().unwrap();
        ret.num_fun_evals = counting_fcn_lock.num_func_evals;
        ret.num_grad_evals = counting_fcn_lock.num_grad_evals;
        Ok(ret)
    }
    else
    {
        let mut minimizer = factory::create(fcn, eq_constraints, ieq_constraints, x0, method);
        minimizer.minimize()
    }
}
//}}}
