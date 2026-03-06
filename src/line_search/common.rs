//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{RealFn, RealFn1, Vector};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use thiserror::Error;
use topohedral_linalg::VectorOps;
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct LineSearchFcn
#[derive(Debug, Clone)]
pub struct LineSearchFcn<F: RealFn>
{
    pub f: F,
    pub x: Vector,
    pub dir: Vector,
}
//}}}
//{{{ impl LineSearchFcn
impl<F: RealFn> LineSearchFcn<F>
{
    #[trace_fn]
    pub fn new(
        f: F,
        x: Vector,
        dir: Vector,
    ) -> Self
    {
        Self { f, x, dir }
    }
}
//}}}
//{{{ impl: RealFn1 for LineSearchFcn
impl<F: RealFn> RealFn1 for LineSearchFcn<F>
{
    #[trace_fn]
    fn eval(
        &mut self,
        alpha: f64,
    ) -> f64
    {
        let x = self.x.clone() + alpha * self.dir.clone();
        self.f.eval(&x)
    }

    #[trace_fn]
    fn diff(
        &mut self,
        alpha: f64,
    ) -> f64
    {
        let x = self.x.clone() + alpha * self.dir.clone();
        let grad = self.f.grad(&x);
        grad.dot(&self.dir)
    }
}
//}}}
//{{{ enum: Error
#[derive(PartialEq, Error, Debug)]
pub enum Error
{
    #[error("Not decreasing")]
    NotDecreasing,
    #[error("Fails Armijo condition")]
    Armijo,
    #[error("Fails curvature condition")]
    Curvature,
    #[error("Max iterations reached")]
    MaxIterations,
    #[error("Step size too small")]
    StepSizeSmall,
    #[error("Step size too large")]
    StepSizeLarge,
    #[error("No step found")]
    NoStepFound,
}
//}}}
//{{{ struct: Options
/// Options for configuring a line search algorithm.
///
/// This struct contains the parameters needed to configure a line search algorithm,
/// such as the initial function value (`phi0`), the initial derivative value (`dphi0`),
/// and the Armijo and curvature conditions (`c1` and `c2`). The `method` field
/// specifies the line search method to use, which can be one of `FixedStep`, `Quadratic`,
/// or `Inexact`.
#[derive(Debug, Copy, Clone)]
pub struct Options
{
    pub c1: f64,
    pub c2: f64,
    pub step_min: f64,
    pub step_max: f64,
}
//}}}
//{{{ impl: Default for Options
impl Default for Options
{
    #[trace_fn]
    fn default() -> Self
    {
        Self {
            c1: 1e-4,
            c2: 0.9,
            step_min: 0.0,
            step_max: 50.0,
        }
    }
}
//}}}
//{{{ struct: Returns
/// The results of a line search algorithm.
///
/// This struct contains the following fields:
/// - `alpha`: The step size found by the line search.
/// - `falpha`: The function value at the step size `alpha`.
/// - `funcalls`: The number of function evaluations performed.
/// - `gradcalls`: The number of gradient evaluations performed.
#[derive(Debug, Copy, Clone)]
pub struct Returns
{
    pub alpha: f64,
    pub phi_alpha: f64,
    pub dphi_alpha: f64,
}
//}}}
//{{{ trait: LineSearch
pub trait LineSearch
{
    type Function: RealFn1;
    fn search(
        &mut self,
        phi0: f64,
        dphi0: f64,
        alpha1: f64,
    ) -> Result<Returns, Error>;
    fn update_fcn(
        &mut self,
        fcn: Self::Function,
    );
}
//}}}
