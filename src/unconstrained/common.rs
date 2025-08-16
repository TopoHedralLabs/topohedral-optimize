//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports 
use crate::line_search::LineSearchError;
use crate::line_search::LineSearchMethod;
//}}}
//{{{ std imports 
//}}}
//{{{ dep imports 
use thiserror::Error;
//}}}
//--------------------------------------------------------------------------------------------------

#[derive(Copy, Clone)]
pub struct Options {
    pub grad_rtol: f64,
    pub grad_atol: f64,
    pub max_iter: u64,
    pub ls_method: LineSearchMethod,
}

#[derive(Copy, Clone, Debug)]
pub enum ConvergedReason {
    Rtol, 
    Atol,
}

#[derive(Copy, Clone, Debug)]
pub struct Returns<Vector> {
    pub xmin: Vector,
    pub fmin: f64,
    pub reason: ConvergedReason, 
    pub num_iterations: usize, 
    pub num_fun_evals: usize, 
    pub num_grad_evals: usize, 
}

#[derive(Error, Debug)] 
pub enum Error {
    #[error("Linear search failed with error {0}")]
    LineSearch(#[from] LineSearchError),
    #[error("Maximum iterations of {0} reached")]
    MaxIterations(usize),
}

pub trait UnconstrainedMinimizer {

    type Vector;
    fn minimize(&mut self) -> Result<Returns<Self::Vector>, Error>;
}