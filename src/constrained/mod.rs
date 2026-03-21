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
//}}}

//{{{ pub use: exports
pub use augmented_lagrangian::AugmentedLagrangianFcn;
//}}}
