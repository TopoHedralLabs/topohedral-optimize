//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports

use crate::common::{arc_real_fn, CountingRealFn};
use crate::common::{RealFn, Returns, Vector};
use crate::VectorReturns;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ mod: submodules
mod common;
mod conjugate_gradient;
mod factory;
mod quasi_newton;
//}}}
//{{{ pub use: common exports
pub use common::{Error as UnconstrainedError, Options as UnconstrainedOptions};
//}}}
//{{{ pub use: conjugate_gradient exports
pub use conjugate_gradient::{Direction, Options as ConjugateGradientOptions};
//}}}
//{{{ pub use: quasi_newton exports
pub use quasi_newton::{Options as QuasiNewtonOptions, UpdateMethod};
//}}}
//{{{ pub use: factory exports
pub use factory::Method as UnconstrainedMethod;
//}}}
//{{{ fn: minimize
#[trace_fn]
pub fn minimize<F: RealFn>(
    fcn: F,
    x0: Vector,
    method: UnconstrainedMethod,
) -> Result<VectorReturns, UnconstrainedError> {
    if method.uncon_opts().make_counting {
        let counting_fcn = arc_real_fn(CountingRealFn::new(fcn));
        let mut minimizer = factory::create(counting_fcn.clone(), x0, method);
        let mut ret = minimizer.minimize()?;
        let counting_fcn_lock = counting_fcn.lock().unwrap();
        ret.num_fun_evals = counting_fcn_lock.num_func_evals;
        ret.num_grad_evals = counting_fcn_lock.num_grad_evals;
        Ok(ret)
    } else {
        let mut minimizer = factory::create(fcn, x0, method);
        minimizer.minimize()
    }
}
//}}}
