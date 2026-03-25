//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::line_search::LineSearchError;
use crate::line_search::LineSearchMethod;
use crate::Vector;
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
    pub max_iter: u64,
    pub make_counting: bool,
    pub ls_method: LineSearchMethod,
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
    #[error("Linear search failed with error {0}")]
    LineSearch(#[from] LineSearchError),
    #[error("Maximum iterations of {0} reached")]
    MaxIterations(usize),
}
//}}}
//{{{ trait: UnconstrainedMinimizer
pub trait UnconstrainedMinimizer
{
    fn minimize(&mut self) -> Result<Returns, Error>;
}
//}}}
