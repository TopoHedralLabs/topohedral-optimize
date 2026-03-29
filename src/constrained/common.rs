//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::Vector;
use crate::unconstrained::{UnconstrainedError, UnconstrainedMethod, UnconstrainedReturns};
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
}
//}}}

//{{{ enum: ConvergedReason
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ConvergedReason
{
    Rtol,
    Atol,
}
//}}}

//{{{ struct: Returns
#[derive(Clone, Debug)]
pub struct Returns
{
    pub xmin: Vector,
    pub fmin: f64,
    pub eq_values_min: Option<Vector>,
    pub ieq_values_min: Option<Vector>,
    pub reason: ConvergedReason,
    pub num_iterations: usize,
    pub num_fun_evals: usize,
    pub num_grad_evals: usize,
}
//}}}

//{{{ enum: Error
#[derive(Error, Debug)]
pub enum Error
{
    #[error("Unconstrianed minimization failed with error {0}")]
    UnconstrainedError(#[from] UnconstrainedError),
}
//}}}

#[derive(Clone, Debug)]
pub struct IterData
{
    pub fx: f64,
    pub x: Vector,
}

impl From<UnconstrainedReturns> for IterData
{
    fn from(ret: UnconstrainedReturns) -> Self
    {
        Self {
            fx: ret.fmin,
            x: ret.xmin,
        }
    }
}

//{{{ trait: ConstrainedMinimizer
pub trait ConstrainedMinimizer
{
    fn minimize(&mut self) -> Result<Returns, Error>;
}
//}}}
