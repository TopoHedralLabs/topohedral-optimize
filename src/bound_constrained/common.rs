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
#[derive(Copy, Clone)]
pub struct Options
{
    pub grad_rtol: f64,
    pub grad_atol: f64,
    pub constraint_tol: f64,
    pub max_iter: u64,
    pub make_counting: bool,
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
