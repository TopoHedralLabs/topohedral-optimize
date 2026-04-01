//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::Vector;
use crate::unconstrained::{UnconstrainedError, UnconstrainedMethod, UnconstrainedReturns};
//}}}
//{{{ std imports
use std::fmt::{self, Display, Formatter};
//}}}
//{{{ dep imports
use thiserror::Error;
use topohedral_linalg::dvector::VecType;
use topohedral_linalg::VectorOps;
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
    #[error("Unconstrianed minimization failed with error {0}")]
    UnconstrainedError(#[from] UnconstrainedError),
}
//}}}

#[derive(Clone, Debug)]
pub struct IterData
{
    pub fx: f64,
    pub x: Vector,
    pub grad_x: Vector,
}

//{{{ impl: Display for IterData
impl Display for IterData
{
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> fmt::Result
    {
        let fx = self.fx;
        let norm_grad_fx = self.grad_x.norm();
        let out = format!("fx={fx:1.4e}, norm_grad_fx={norm_grad_fx:1.4e}");
        f.pad(&out)
    }
}
//}}}

impl From<UnconstrainedReturns> for IterData
{
    fn from(ret: UnconstrainedReturns) -> Self
    {
        let n = ret.xmin.len();
        Self {
            fx: ret.fmin,
            x: ret.xmin,
            grad_x: Vector::zeros_cvec(n, VecType::Col),
        }
    }
}

//{{{ trait: ConstrainedMinimizer
pub trait ConstrainedMinimizer
{
    fn minimize(&mut self) -> Result<Returns, Error>;
}
//}}}
