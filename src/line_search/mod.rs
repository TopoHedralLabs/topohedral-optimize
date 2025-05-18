//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports 
use crate::{RealFn, RealFn1};
//}}}
//{{{ std imports 
use std::ops::{Add, Mul};
//}}}
//{{{ dep imports 
use topohedral_linalg::VectorOps;
//}}}
//--------------------------------------------------------------------------------------------------

pub struct LineSearchFcn<F: RealFn, Vector> {
    pub f: F,
    pub x: Vector,
    pub dir: Vector, 
}


impl<F: RealFn, Vector> LineSearchFcn<F, Vector> {
    pub fn new(f: F, x: Vector, dir: Vector) -> Self {
        Self { f, x, dir }
    }
}

impl<F: RealFn, Vector> RealFn1 for LineSearchFcn<F, Vector>
where
    F: RealFn<Vector = Vector>,
    Vector: Add<Output=Vector> + Mul<f64> + Clone + VectorOps<ScalarType = f64>, 
    f64: Mul<Vector, Output = Vector>,
{
    fn eval(&self, alpha: f64) -> f64 {
        let x = self.x.clone() + alpha * self.dir.clone();
        self.f.eval(&x)
    }

    fn diff(&self, alpha: f64) -> f64 {
        let x = self.x.clone() + alpha * self.dir.clone();
        let grad = self.f.grad(&x);
        grad.dot(&self.dir)
    }
}

pub trait LineSearcher {
    type Returns;
    type Error;
    fn search(&mut self, phi0: f64, dphi0: f64) ->  Result<Self::Returns, Self::Error>;
}