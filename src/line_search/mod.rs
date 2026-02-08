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
mod factory;
mod interp;
mod thuente;
mod utils;

pub use common::{
    Error as LineSearchError, LineSearch, LineSearchFcn, Options as LineSearchOptions,
    Returns as LineSearchReturns,
};

pub use factory::{create, Method as LineSearchMethod};

pub use interp::Interp;
pub use interp::Options as InterpOptions;

pub use thuente::Options as ThuenteOptions;
pub use thuente::Thuente;

pub use utils::initial_step;
