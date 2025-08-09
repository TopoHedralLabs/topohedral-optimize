//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports 
use crate::line_search::Error as LineSearchError;
use crate::line_search::Method as LineSearchMethod;
//}}}
//{{{ std imports 
//}}}
//{{{ dep imports 
use thiserror::Error;
//}}}
//--------------------------------------------------------------------------------------------------

pub struct Options {
    pub tol: f64,
    pub max_iter: u64,
    pub ls_method: LineSearchMethod,
}

pub struct Returns<Vector> {

    xmin: Vector,
    fmin: f64,
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
    fn minimize(self, x0: Self::Vector) -> Result<Returns<Self::Vector>, Error>;
}