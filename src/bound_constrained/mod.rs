//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::{
    arc_real_fn, CountingRealFn, RealFn, RealVectorFn, Returns as BoundConstrainedReturns, Vector,
};
use crate::constraints::BoundsConstraints;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ mod: submodules
mod asa;
mod common;
mod factory;
mod utils;
//}}}
//{{{ pub use: asa exports
pub use asa::{ActiveSetAlgorithm, Options as AsaOptions};
//}}}
//{{{ pub use: common exports
pub use common::{Error as BoundConstrainedError, Options as BoundConstrainedOptions};
//}}}
//{{{ pub use: factory exports
pub use factory::{create, Method as BoundConstrainedMethod};
//}}}

#[trace_fn]
pub fn minimize<F1: RealFn>(
    fcn: F1,
    bounds: BoundsConstraints,
    x0: Vector,
    method: BoundConstrainedMethod,
) -> Result<BoundConstrainedReturns, BoundConstrainedError>
{
    if method.bound_opts().base_opts.make_counting
    {
        let counting_fcn = arc_real_fn(CountingRealFn::new(fcn));
        let mut minimizer = create(counting_fcn.clone(), bounds, x0, method);
        let mut ret = minimizer.minimize()?;
        let counting_fcn_lock = counting_fcn.lock().unwrap();
        ret.num_fun_evals = counting_fcn_lock.num_func_evals;
        ret.num_grad_evals = counting_fcn_lock.num_grad_evals;
        Ok(ret)
    }
    else
    {
        let mut minimizer = create(fcn, bounds, x0, method);
        minimizer.minimize()
    }
}
