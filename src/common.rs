//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports 
//}}}
//{{{ std imports 
use std::rc::Rc;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use std::fmt::Debug;
//}}}
//{{{ dep imports 
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ trait: RealFn1
/// 1D real-valued function trait
pub trait RealFn1 {

    fn eval(&mut self, x: f64) -> f64;
    fn diff(&mut self, x: f64) -> f64;
}
//}}}
//{{{ trait: RealFn
pub trait RealFn: Clone + Debug {

    type Vector;

    fn eval(&mut self, x: &Self::Vector) -> f64;
    fn grad(&mut self, x: &Self::Vector) -> Self::Vector;

}
//}}}
//{{{ impl: RealFn for Rc<RefCell<T>> 
impl<T> RealFn for Rc<RefCell<T>> 
where 
    T: RealFn,
{
    type Vector = T::Vector;

    fn eval(&mut self, x: &Self::Vector) -> f64 {
        self.borrow_mut().eval(x)
    }

    fn grad(&mut self, x: &Self::Vector) -> Self::Vector {
        self.borrow_mut().grad(x)
    }
}
//}}}
//{{{ impl: RealFn for Arc<Mutex<T>> 
impl<T> RealFn for Arc<Mutex<T>> 
where 
    T: RealFn,
{
    type Vector = T::Vector;

    fn eval(&mut self, x: &Self::Vector) -> f64 {
        self.lock().unwrap().eval(x)
    }

    fn grad(&mut self, x: &Self::Vector) -> Self::Vector {
        self.lock().unwrap().grad(x)
    }
}
//}}}
//{{{ struct: CountingRealFcn
#[derive(Clone, Debug)]
pub struct CountingRealFn <F: RealFn> {
    fcn: F, 
    pub num_func_evals: usize, 
    pub num_grad_evals: usize, 
}
//}}}
//{{{ impl: RealFn for CountingRealFcn
impl<F: RealFn> RealFn for CountingRealFn<F> {

    type Vector = F::Vector;

    fn eval(&mut self, x: &Self::Vector) -> f64 {
        self.num_func_evals += 1;
        self.fcn.eval(x)
    }

    fn grad(&mut self, x: &Self::Vector) -> Self::Vector {
        self.num_grad_evals += 1;
        self.fcn.grad(x)
    }
}
//}}}
//{{{ impl: CountingRealFcn
impl<F: RealFn> CountingRealFn<F> {

    pub fn new(fcn: F) -> Self  {
        Self{
            fcn: fcn, 
            num_func_evals: 0, 
            num_grad_evals: 0
        }
    }
}
//}}}
