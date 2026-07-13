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

mod bounded;
mod brent;
mod common;
mod factory;
mod golden;

//{{{ pub use: bounded exports
pub use bounded::{Bounded, Options as BoundedOptions};
//}}}
//{{{ pub use: brent exports
pub use brent::{Bracket, Brent, Options as BrentOptions};
//}}}
//{{{ pub use: common exports
pub use common::{bracket, BracketOptions, BracketResult, Error as ScalarError};
//}}}
