//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

mod common;
mod conjugate_gradient;
mod factory;
mod quasi_newton;

pub use common::{
    ConvergedReason as UnconstrainedConvergedReason, Error as UnconstrainedError,
    Options as UnonstrainedOptions, Returns as UnconstrainedReturns, UnconstrainedMinimizer,
};

pub use conjugate_gradient::{ConjugateGradient, Direction, Options as ConjugateGradientOptions};

pub use quasi_newton::{Options as QuasiNewtonOptions, QuasiNewton, UpdateMethod};

pub use factory::{create, Method};
