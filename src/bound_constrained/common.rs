//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::{arc_real_fn, CountingRealFn, RealFn, RealVectorFn, Vector};
use crate::unconstrained::{UnconstrainedError, UnconstrainedMethod};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use thiserror::Error;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct: Options
pub struct Options
{
    pub uncon_method: UnconstrainedMethod,
}
//}}}
//{{{ enum: Error
#[derive(Error, Debug)]
pub enum Error
{
    #[error("Unconstrianed minimization failed with error {0}")]
    UnconstrainedError(#[from] UnconstrainedError),
    #[error("Maximum iterations of {0} reached")]
    MaxIterations(usize),
}
//}}}
//{{{ trait: ConstrainedMinimizer
pub trait BoundConstrainedMinimizer
{
    fn minimize(&mut self) -> Result<crate::Returns, Error>;
}
//}}}
