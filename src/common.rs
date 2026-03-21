//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
//}}}
//{{{ std imports
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
//}}}
//{{{ dep imports
use topohedral_linalg::dmatrix::DMatrix;
use topohedral_linalg::dvector::DVector;
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
//{{{ impl: RealFn for Rc<RefCell<T>>
impl<T> RealFn for Rc<RefCell<T>>
where
    T: RealFn,
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
impl<T> RealFn for Arc<Mutex<T>>
where
    T: RealFn,
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
