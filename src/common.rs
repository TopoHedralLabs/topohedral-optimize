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
use topohedral_linalg::dmatrix::DMatrix;
use topohedral_linalg::dvector::DVector;
use topohedral_linalg::VectorOps;
use topohedral_tracing::trace_fn;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ type: core aliases
pub type Vector = DVector<f64>;
pub type Matrix = DMatrix<f64>;
//}}}
//{{{ trait: RealFn1
/// 1D real-valued function trait
pub trait RealFn1
{
    fn eval(
        &mut self,
        x: f64,
    ) -> f64;
    fn diff(
        &mut self,
        x: f64,
    ) -> f64;
}
//}}}
//{{{ trait: RealFn
pub trait RealFn: Clone + Debug
{
    fn dimension(&self) -> usize;
    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64;
    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector;
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
    #[trace_fn]
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

    pub fn copy_from(
        &mut self,
        iter_data: &Self,
    )
    {
        self.x.copy_from(&iter_data.x);
        self.fx = iter_data.fx;
        self.grad_fx.copy_from(&iter_data.grad_fx);
        self.norm_grad_fx = iter_data.norm_grad_fx;
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
//{{{ impl: RealFn for Rc<RefCell<T>>
impl<F> RealFn for Rc<RefCell<F>>
where
    F: RealFn,
{
    fn dimension(&self) -> usize
    {
        self.borrow().dimension()
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        self.borrow_mut().eval(x)
    }

    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector
    {
        self.borrow_mut().grad(x)
    }
}
//}}}
//{{{ impl: RealFn for Arc<Mutex<T>>
impl<F> RealFn for Arc<Mutex<F>>
where
    F: RealFn,
{
    fn dimension(&self) -> usize
    {
        self.lock().unwrap().dimension()
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        self.lock().unwrap().eval(x)
    }

    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector
    {
        self.lock().unwrap().grad(x)
    }
}
//}}}
//{{{ struct: CountingRealFn
#[derive(Clone, Debug)]
pub(crate) struct CountingRealFn<F: RealFn>
{
    fcn: F,
    pub num_func_evals: usize,
    pub num_grad_evals: usize,
}
//}}}
//{{{ impl: RealFn for CountingRealFn
impl<F: RealFn> RealFn for CountingRealFn<F>
{
    fn dimension(&self) -> usize
    {
        self.fcn.dimension()
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        self.num_func_evals += 1;
        self.fcn.eval(x)
    }

    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector
    {
        self.num_grad_evals += 1;
        self.fcn.grad(x)
    }
}
//}}}
//{{{ impl: CountingRealFn
impl<F: RealFn> CountingRealFn<F>
{
    pub fn new(fcn: F) -> Self
    {
        Self {
            fcn,
            num_func_evals: 0,
            num_grad_evals: 0,
        }
    }

    pub(crate) fn inner_mut(&mut self) -> &mut F
    {
        &mut self.fcn
    }

    pub(crate) fn with_inner_mut<R>(
        &mut self,
        f: impl FnOnce(&mut F) -> R,
    ) -> R
    {
        f(&mut self.fcn)
    }
}
//}}}
//{{{ type: aliases for Rc<RefCell<F>> and Arc<Mutex<F>>
/// Type alias for a function wrapped in Rc<RefCell<F>>
pub type RcRealFn<F> = Rc<RefCell<F>>;
/// Type alias for a function wrapped in Arc<Mutex<F>>
pub type ArcRealFn<F> = Arc<Mutex<F>>;
//}}}
//{{{ fun: rc_real_fn
/// Creates a new reference-counted function using Rc<RefCell>
pub fn rc_real_fn<F: RealFn>(fcn: F) -> RcRealFn<F>
{
    Rc::new(RefCell::new(fcn))
}
//}}}
//{{{ fun: arc_real_fn
/// Creates a new thread-safe reference-counted function using Arc<Mutex>
pub fn arc_real_fn<F: RealFn>(fcn: F) -> ArcRealFn<F>
{
    Arc::new(Mutex::new(fcn))
}
//}}}
//{{{ trait: RealVectorFn
pub trait RealVectorFn: Clone + Debug
{
    fn dimension_domain(&self) -> usize;
    fn dimension_range(&self) -> usize;

    fn eval(
        &mut self,
        x: &Vector,
        val: &mut Vector,
    );
    fn grad(
        &mut self,
        x: &Vector,
        val: &mut Matrix,
    );
}
//}}}
//{{{ impl: RealVectorFn for Rc<RefCell<T>>
impl<T> RealVectorFn for Rc<RefCell<T>>
where
    T: RealVectorFn,
{
    fn dimension_domain(&self) -> usize
    {
        self.borrow().dimension_domain()
    }

    fn dimension_range(&self) -> usize
    {
        self.borrow().dimension_range()
    }

    fn eval(
        &mut self,
        x: &Vector,
        val: &mut Vector,
    )
    {
        self.borrow_mut().eval(x, val)
    }

    fn grad(
        &mut self,
        x: &Vector,
        val: &mut Matrix,
    )
    {
        self.borrow_mut().grad(x, val)
    }
}
//}}}
//{{{ impl: RealVectorFn for Arc<Mutex<T>>
impl<T> RealVectorFn for Arc<Mutex<T>>
where
    T: RealVectorFn,
{
    fn dimension_domain(&self) -> usize
    {
        self.lock().unwrap().dimension_domain()
    }

    fn dimension_range(&self) -> usize
    {
        self.lock().unwrap().dimension_range()
    }

    fn eval(
        &mut self,
        x: &Vector,
        val: &mut Vector,
    )
    {
        self.lock().unwrap().eval(x, val)
    }

    fn grad(
        &mut self,
        x: &Vector,
        val: &mut DMatrix<f64>,
    )
    {
        self.lock().unwrap().grad(x, val)
    }
}
//}}}
//{{{ type: aliases for Rc<RefCell<F>> and Arc<Mutex<F>>
/// Type alias for a vector-valued function wrapped in Rc<RefCell<F>>
pub type RcRealVectorFn<F> = Rc<RefCell<F>>;
/// Type alias for a vector-valued function wrapped in Arc<Mutex<F>>
pub type ArcRealVectorFn<F> = Arc<Mutex<F>>;
//}}}
//{{{ fun: rc_real_vector_fn
/// Creates a new reference-counted vector-valued function using Rc<RefCell>
pub fn rc_real_vector_fn<F: RealVectorFn>(fcn: F) -> RcRealVectorFn<F>
{
    Rc::new(RefCell::new(fcn))
}
//}}}
//{{{ fun: arc_real_vector_fn
/// Creates a new thread-safe reference-counted vector-valued function using Arc<Mutex>
pub fn arc_real_vector_fn<F: RealVectorFn>(fcn: F) -> ArcRealVectorFn<F>
{
    Arc::new(Mutex::new(fcn))
}
//}}}
