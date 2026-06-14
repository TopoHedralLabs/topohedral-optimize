//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::line_search::LineSearchError;
//}}}
//{{{ dep imports
use thiserror::Error;
//}}}
//{{{ std imports
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ type: Options
pub type Options = crate::common::BaseOptions;
//}}}
//{{{ enum: Error
#[derive(Error, Debug)]
pub enum Error
{
    #[error("Linear search failed with error {0}")]
    LineSearch(#[from] LineSearchError),
    #[error("Maximum iterations of {0} reached")]
    MaxIterations(usize),
}
//}}}
