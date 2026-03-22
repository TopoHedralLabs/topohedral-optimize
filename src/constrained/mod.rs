//! Module implementing a set of constrained optimisation algorithms
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
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
use common::{
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
pub use factory::{create, Method as ConstrainedMethod};

//}}}
