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
mod common;
mod factory;
mod interp;
mod nocedal;
mod thuente;
mod utils;
//}}}
//{{{ pub use: common exports
pub use common::{
    Error as LineSearchError, LineSearch, LineSearchFcn, Options as LineSearchOptions,
    Returns as LineSearchReturns,
};
//}}}
//{{{ pub use: factory exports
pub use factory::{create, Method as LineSearchMethod};
//}}}
//{{{ pub use: interp exports
pub use interp::Interp;
pub use interp::Options as InterpOptions;
//}}}
//{{{ pub use: thuente exports
pub use thuente::Options as ThuenteOptions;
pub use thuente::Thuente;
//}}}
//{{{ pub use: nocedal exports
pub use nocedal::Nocedal;
pub use nocedal::Options as NocedalOptions;
//}}}
//{{{ pub use: utils exports
pub use utils::initial_step;
//}}}
