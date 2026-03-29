//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::line_search::LineSearchError;
use crate::line_search::LineSearchMethod;
use crate::{RealFn, Vector};
//}}}
//{{{ std imports
use std::fmt::{self, Display, Formatter};
//}}}
//{{{ dep imports
use thiserror::Error;
use topohedral_linalg::VectorOps;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct: IterData
#[derive(Debug, Clone)]
pub struct IterData
{
    pub x: Vector,
    pub fx: f64,
    pub grad_fx: Vector,
    pub norm_grad_fx: f64,
}
//}}}
//{{{ impl: IterData
impl IterData
{
    pub fn new<F: RealFn>(
        mut fcn: F,
        x: &Vector,
    ) -> Self
    {
        let fx = fcn.eval(x);
        let grad_fx = fcn.grad(x);
        let norm_grad_fx = grad_fx.norm();
        IterData {
            x: x.clone(),
            fx,
            grad_fx,
            norm_grad_fx,
        }
    }
}
//}}}
//{{{ impl: Display for IterData
impl Display for IterData
{
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> fmt::Result
    {
        let fx = self.fx;
        let norm_grad_fx = self.norm_grad_fx;
        let out = format!("fx={fx:1.4e}, norm_grad_fx={norm_grad_fx:1.4e}");
        f.pad(&out)
    }
}
//}}}

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
