//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
use std::fmt::{self, Debug, Display, Formatter};
//}}}
//{{{ dep imports
use topohedral_linalg::DMatrix;
use topohedral_linalg::DVector;
use topohedral_linalg::VectorOps;
use topohedral_tracing::trace_fn;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ type: core aliases
pub type Vector = DVector<f64>;
pub type Matrix = DMatrix<f64>;
//}}}
//{{{ trait: DifferentiableFn
/// A differentiable function with associated input, output, and derivative types.
pub trait DifferentiableFn {
    type Input;
    type Output;
    type Derivative;

    fn eval(
        &mut self,
        x: &Self::Input,
    ) -> Self::Output;

    fn derivative(
        &mut self,
        x: &Self::Input,
    ) -> Self::Derivative;

    fn dimension(&self) -> usize {
        1
    }

    fn dimension_domain(&self) -> usize {
        self.dimension()
    }

    fn dimension_range(&self) -> usize {
        1
    }
}
//}}}
//{{{ impl: DifferentiableFn for &mut F
impl<F: DifferentiableFn + ?Sized> DifferentiableFn for &mut F {
    type Input = F::Input;
    type Output = F::Output;
    type Derivative = F::Derivative;

    fn eval(
        &mut self,
        x: &Self::Input,
    ) -> Self::Output {
        (**self).eval(x)
    }

    fn derivative(
        &mut self,
        x: &Self::Input,
    ) -> Self::Derivative {
        (**self).derivative(x)
    }

    fn dimension(&self) -> usize {
        (**self).dimension()
    }

    fn dimension_domain(&self) -> usize {
        (**self).dimension_domain()
    }

    fn dimension_range(&self) -> usize {
        (**self).dimension_range()
    }
}
//}}}
//{{{ trait: RealFn1
/// Stable-Rust equivalent of a trait alias for a differentiable `f64 -> f64` function.
pub trait RealFn1: DifferentiableFn<Input = f64, Output = f64, Derivative = f64> {}

impl<F> RealFn1 for F where F: DifferentiableFn<Input = f64, Output = f64, Derivative = f64> {}
//}}}
//{{{ trait: RealFn
/// Stable-Rust equivalent of a trait alias for a scalar-valued function on `Vector`.
pub trait RealFn:
    DifferentiableFn<Input = Vector, Output = f64, Derivative = Vector> + Debug
{
}

impl<F> RealFn for F where
    F: DifferentiableFn<Input = Vector, Output = f64, Derivative = Vector> + Debug + ?Sized
{
}
//}}}
//{{{ enum: ConvergedReason
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ConvergedReason {
    Rtol,
    Atol,
}
//}}}
//{{{ struct: Options
#[derive(Copy, Clone)]
pub struct BaseOptions {
    pub grad_rtol: f64,
    pub grad_atol: f64,
    pub max_iter: u64,
}
//}}}
//{{{ struct Returns
#[derive(Clone, Debug)]
pub struct Returns<T> {
    pub xmin: T,
    pub fmin: f64,
    pub reason: ConvergedReason,
    pub num_iterations: usize,
    pub num_fun_evals: usize,
    pub num_grad_evals: usize,
}
//}}}
//{{{ type: Returns aliases
pub type ScalarReturns = Returns<f64>;
pub type VectorReturns = Returns<Vector>;
//}}}
//{{{ trait: Minimizer
pub(crate) trait Minimizer {
    type Error;
    type Returns;

    fn minimize(&mut self) -> Result<Self::Returns, Self::Error>;
}
//}}}
//{{{ struct: IterData
#[derive(Debug, Clone)]
pub struct IterData {
    pub x: Vector,
    pub fx: f64,
    pub grad_fx: Vector,
    pub norm_grad_fx: f64,
}
//}}}
//{{{ impl: IterData
impl IterData {
    #[trace_fn]
    pub fn new<F: RealFn + ?Sized>(
        fcn: &mut F,
        x: &Vector,
    ) -> Self {
        let fx = fcn.eval(x);
        let grad_fx = fcn.derivative(x);
        let norm_grad_fx = grad_fx.norm();
        IterData {
            x: x.clone(),
            fx,
            grad_fx,
            norm_grad_fx,
        }
    }

    #[trace_fn]
    pub fn copy_from(
        &mut self,
        iter_data: &Self,
    ) {
        self.x.copy_from(&iter_data.x);
        self.fx = iter_data.fx;
        self.grad_fx.copy_from(&iter_data.grad_fx);
        self.norm_grad_fx = iter_data.norm_grad_fx;
    }
}
//}}}
//{{{ impl: Display for IterData
impl Display for IterData {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        let fx = self.fx;
        let norm_grad_fx = self.norm_grad_fx;
        let out = format!("fx={fx:.4e}, norm_grad_fx={norm_grad_fx:.4e}");
        f.pad(&out)
    }
}
//}}}
//{{{ struct: Evaluator
#[derive(Debug)]
pub(crate) struct Evaluator<'a, F: RealFn + ?Sized> {
    fcn: &'a mut F,
    pub num_func_evals: usize,
    pub num_grad_evals: usize,
}
//}}}
//{{{ impl: DifferentiableFn for Evaluator
impl<F: RealFn + ?Sized> DifferentiableFn for Evaluator<'_, F> {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;

    #[trace_fn]
    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        self.num_func_evals += 1;
        self.fcn.eval(x)
    }

    #[trace_fn]
    fn derivative(
        &mut self,
        x: &Vector,
    ) -> Vector {
        self.num_grad_evals += 1;
        self.fcn.derivative(x)
    }

    #[trace_fn]
    fn dimension(&self) -> usize {
        self.fcn.dimension()
    }
}
//}}}
//{{{ impl: Evaluator
impl<'a, F: RealFn + ?Sized> Evaluator<'a, F> {
    #[trace_fn]
    pub fn new(fcn: &'a mut F) -> Self {
        Self {
            fcn,
            num_func_evals: 0,
            num_grad_evals: 0,
        }
    }
}
//}}}
//{{{ trait: RealVectorFn
/// Stable-Rust equivalent of a trait alias for a vector-valued function on `Vector`.
pub trait RealVectorFn:
    DifferentiableFn<Input = Vector, Output = Vector, Derivative = Matrix> + Debug
{
}

impl<F> RealVectorFn for F where
    F: DifferentiableFn<Input = Vector, Output = Vector, Derivative = Matrix> + Debug + ?Sized
{
}
//}}}
