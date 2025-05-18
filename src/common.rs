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
//}}}
//{{{ dep imports 
//}}}
//--------------------------------------------------------------------------------------------------

/// 1D real-valued function trait
pub trait RealFn1 {

    fn eval(&self, x: f64) -> f64;
    fn diff(&self, x: f64) -> f64;
}




pub trait RealFn {

    type Vector;

    fn eval(&self, x: &Self::Vector) -> f64;
    fn grad(&self, x: &Self::Vector) -> Self::Vector;

}

impl<T> RealFn for Rc<RefCell<T>> 
where 
    T: RealFn,
{
    type Vector = T::Vector;

    fn eval(&self, x: &Self::Vector) -> f64 {
        self.borrow().eval(x)
    }

    fn grad(&self, x: &Self::Vector) -> Self::Vector {
        self.borrow().grad(x)
    }
}


impl<T> RealFn for Arc<Mutex<T>> 
where 
    T: RealFn,
{
    type Vector = T::Vector;

    fn eval(&self, x: &Self::Vector) -> f64 {
        self.lock().unwrap().eval(x)
    }

    fn grad(&self, x: &Self::Vector) -> Self::Vector {
        self.lock().unwrap().grad(x)
    }
}