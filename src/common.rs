//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
use std::cell::RefCell;
use std::fmt::{self, Debug, Display, Formatter};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
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
//{{{ trait: RealFn1
/// Stable-Rust equivalent of a trait alias for a differentiable `f64 -> f64` function.
pub trait RealFn1: DifferentiableFn<Input = f64, Output = f64, Derivative = f64> {}

impl<F> RealFn1 for F where F: DifferentiableFn<Input = f64, Output = f64, Derivative = f64> {}
//}}}
//{{{ trait: RealFn
/// Stable-Rust equivalent of a trait alias for a scalar-valued function on `Vector`.
pub trait RealFn:
    DifferentiableFn<Input = Vector, Output = f64, Derivative = Vector> + Clone + Debug
{
}

impl<F> RealFn for F where
    F: DifferentiableFn<Input = Vector, Output = f64, Derivative = Vector> + Clone + Debug
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
    pub make_counting: bool,
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
    pub fn new<F: RealFn>(
        mut fcn: F,
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
//{{{ impl: DifferentiableFn for Rc<RefCell<T>>
impl<F> DifferentiableFn for Rc<RefCell<F>>
where
    F: DifferentiableFn,
{
    type Input = F::Input;
    type Output = F::Output;
    type Derivative = F::Derivative;

    #[trace_fn]
    fn eval(
        &mut self,
        x: &Self::Input,
    ) -> Self::Output {
        self.borrow_mut().eval(x)
    }

    #[trace_fn]
    fn derivative(
        &mut self,
        x: &Self::Input,
    ) -> Self::Derivative {
        self.borrow_mut().derivative(x)
    }

    #[trace_fn]
    fn dimension(&self) -> usize {
        self.borrow().dimension()
    }

    #[trace_fn]
    fn dimension_domain(&self) -> usize {
        self.borrow().dimension_domain()
    }

    #[trace_fn]
    fn dimension_range(&self) -> usize {
        self.borrow().dimension_range()
    }
}
//}}}
//{{{ impl: DifferentiableFn for Arc<Mutex<T>>
impl<F> DifferentiableFn for Arc<Mutex<F>>
where
    F: DifferentiableFn,
{
    type Input = F::Input;
    type Output = F::Output;
    type Derivative = F::Derivative;

    #[trace_fn]
    fn eval(
        &mut self,
        x: &Self::Input,
    ) -> Self::Output {
        self.lock().unwrap().eval(x)
    }

    #[trace_fn]
    fn derivative(
        &mut self,
        x: &Self::Input,
    ) -> Self::Derivative {
        self.lock().unwrap().derivative(x)
    }

    #[trace_fn]
    fn dimension(&self) -> usize {
        self.lock().unwrap().dimension()
    }

    #[trace_fn]
    fn dimension_domain(&self) -> usize {
        self.lock().unwrap().dimension_domain()
    }

    #[trace_fn]
    fn dimension_range(&self) -> usize {
        self.lock().unwrap().dimension_range()
    }
}
//}}}
//{{{ struct: CountingRealFn
#[derive(Clone, Debug)]
pub(crate) struct CountingRealFn<F: RealFn> {
    fcn: F,
    pub num_func_evals: usize,
    pub num_grad_evals: usize,
}
//}}}
//{{{ impl: DifferentiableFn for CountingRealFn
impl<F: RealFn> DifferentiableFn for CountingRealFn<F> {
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
//{{{ impl: CountingRealFn
impl<F: RealFn> CountingRealFn<F> {
    #[trace_fn]
    pub fn new(fcn: F) -> Self {
        Self {
            fcn,
            num_func_evals: 0,
            num_grad_evals: 0,
        }
    }

    #[trace_fn]
    pub(crate) fn inner_mut(&mut self) -> &mut F {
        &mut self.fcn
    }

    #[trace_fn]
    pub(crate) fn with_inner_mut<R>(
        &mut self,
        f: impl FnOnce(&mut F) -> R,
    ) -> R {
        f(&mut self.fcn)
    }
}
//}}}
//{{{ type: aliases for Rc<RefCell<F>> and Arc<Mutex<F>>
/// Type alias for a function wrapped in `Rc<RefCell<F>>`.
pub type RcRealFn<F> = Rc<RefCell<F>>;
/// Type alias for a function wrapped in `Arc<Mutex<F>>`.
pub type ArcRealFn<F> = Arc<Mutex<F>>;
//}}}
//{{{ fun: rc_real_fn
/// Creates a new reference-counted function using `Rc<RefCell>`.
#[trace_fn]
pub fn rc_real_fn<F: RealFn>(fcn: F) -> RcRealFn<F> {
    Rc::new(RefCell::new(fcn))
}
//}}}
//{{{ fun: arc_real_fn
/// Creates a new thread-safe reference-counted function using `Arc<Mutex>`.
#[trace_fn]
pub fn arc_real_fn<F: RealFn>(fcn: F) -> ArcRealFn<F> {
    Arc::new(Mutex::new(fcn))
}
//}}}
//{{{ trait: RealVectorFn
/// Stable-Rust equivalent of a trait alias for a vector-valued function on `Vector`.
pub trait RealVectorFn:
    DifferentiableFn<Input = Vector, Output = Vector, Derivative = Matrix> + Clone + Debug
{
}

impl<F> RealVectorFn for F where
    F: DifferentiableFn<Input = Vector, Output = Vector, Derivative = Matrix> + Clone + Debug
{
}
//}}}
//{{{ type: aliases for Rc<RefCell<F>> and Arc<Mutex<F>>
/// Type alias for a vector-valued function wrapped in `Rc<RefCell<F>>`.
pub type RcRealVectorFn<F> = Rc<RefCell<F>>;
/// Type alias for a vector-valued function wrapped in `Arc<Mutex<F>>`.
pub type ArcRealVectorFn<F> = Arc<Mutex<F>>;
//}}}
//{{{ fun: rc_real_vector_fn
/// Creates a new reference-counted vector-valued function using `Rc<RefCell>`.
#[trace_fn]
pub fn rc_real_vector_fn<F: RealVectorFn>(fcn: F) -> RcRealVectorFn<F> {
    Rc::new(RefCell::new(fcn))
}
//}}}
//{{{ fun: arc_real_vector_fn
/// Creates a new thread-safe vector-valued function using `Arc<Mutex>`.
#[trace_fn]
pub fn arc_real_vector_fn<F: RealVectorFn>(fcn: F) -> ArcRealVectorFn<F> {
    Arc::new(Mutex::new(fcn))
}
//}}}
