//! Module implementing a set of constrained optimisation algorithms
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::{arc_real_fn, CountingRealFn, RealFn, RealVectorFn, Vector};
use crate::unconstrained;
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
pub use crate::common::{
    ConvergedReason as ConstrainedConvergedReason, Minimizer as ConstrainedMinimizer,
    Returns as ConstrainedReturns,
};
pub use common::{Error as ConstrainedError, Options as ConstriainedOptions};
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
#[trace_fn]
pub fn minimize<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn>(
    fcn: F1,
    eq_constraints: Option<F2>,
    ieq_constraints: Option<F3>,
    x0: Vector,
    method: ConstrainedMethod,
) -> Result<ConstrainedReturns, ConstrainedError>
{
    let has_constraints = eq_constraints.is_some() || ieq_constraints.is_some();
    if has_constraints
    {
        if method.con_opts().base_opts.make_counting
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
    else
    {
        let atol = method.con_opts().base_opts.grad_atol;
        let rtol = method.con_opts().base_opts.grad_rtol;
        let max_iter = method.con_opts().base_opts.max_iter;
        let make_counting = method.con_opts().base_opts.make_counting;
        let mut uncon_method = method.uncon_method().clone();
        uncon_method.uncon_opts_mut().grad_atol = atol;
        uncon_method.uncon_opts_mut().grad_rtol = rtol;
        uncon_method.uncon_opts_mut().max_iter = max_iter;
        uncon_method.uncon_opts_mut().make_counting = make_counting;
        Ok(unconstrained::minimize(fcn, x0, uncon_method)?)
    }
}
//}}}
