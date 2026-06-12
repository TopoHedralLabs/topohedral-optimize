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

//{{{ mod: submodules
mod asa;
mod common;
mod factory;
mod utils;
//}}}
pub use asa::{ActiveSetAlgorithm, Options as AsaOptions};
pub use common::{Error as BoundConstrainedError, Options as BoundConstrainedOptions};
pub use factory::{create, Method};
