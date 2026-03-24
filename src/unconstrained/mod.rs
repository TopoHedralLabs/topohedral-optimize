//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::{RealFn, Vector};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ mod: submodules
mod common;
mod conjugate_gradient;
mod factory;
mod quasi_newton;
//}}}
//{{{ pub use: common exports
pub use common::{
    ConvergedReason as UnconstrainedConvergedReason, Error as UnconstrainedError,
    Options as UnonstrainedOptions, Returns as UnconstrainedReturns, UnconstrainedMinimizer,
};
//}}}
//{{{ pub use: conjugate_gradient exports
pub use conjugate_gradient::{ConjugateGradient, Direction, Options as ConjugateGradientOptions};
//}}}
//{{{ pub use: quasi_newton exports
pub use quasi_newton::{Options as QuasiNewtonOptions, QuasiNewton, UpdateMethod};
//}}}
//{{{ pub use: factory exports
pub use factory::{create, Method as UnconstrainedMethod};
//}}}

//{{{ fn: minimize
pub fn minimize<F: RealFn>(
    fcn: F,
    x0: Vector,
    method: UnconstrainedMethod,
) -> Result<UnconstrainedReturns, UnconstrainedError>
{
    let mut minimizer = create(fcn, x0, method);
    return minimizer.minimize();
}
//}}}
