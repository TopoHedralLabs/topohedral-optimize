//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

use crate::RealFn;

//{{{ crate imports
use super::common::Options as BoundConstrainedOptions;
use crate::constraints::BoundsConstraints;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

pub struct Options
{
    boundcon_opts: BoundConstrainedOptions,
}

struct ProjectedFunction<F: RealFn>
{
    fcn: F,
    bounds: BoundsConstraints,
}
