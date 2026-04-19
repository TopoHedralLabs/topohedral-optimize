//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::line_search::LineSearchError;
use crate::line_search::LineSearchMethod;
use crate::Returns;
//}}}
//{{{ dep imports
use std::fmt;
use thiserror::Error;
//}}}
//{{{ std imports
use std::fmt::Display;
use std::fmt::Formatter;

//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct: Options
#[derive(Clone)]
pub struct Options
{
    pub grad_rtol: f64,
    pub grad_atol: f64,
    pub max_iter: u64,
    pub make_counting: bool,
    pub ls_method: LineSearchMethod,
}
//}}}
//{{{ impl Display for Options
impl Display for Options
{
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> fmt::Result
    {
        let out = format!(
            "grad_rtol = {:1.4e} grad_atol = {:1.4e} max_iter = {}",
            self.grad_rtol, self.grad_atol, self.max_iter
        );
        f.pad(&out)
    }
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
