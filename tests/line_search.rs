
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(impl_trait_in_assoc_type)]

//{{{ crate imports
use topohedral_optimize::{RealFn, RealFn1, CountingRealFn, 
                           line_search::{Interp, InterpOptions, LineSearchFcn}};
use topohedral_optimize::line_search::Options as LineSearchOptions;
use topohedral_optimize::line_search::LineSearcher;
//}}}
//{{{ std imports
use std::{rc::Rc, sync::Mutex};
use std::sync::Arc;
use std::cell::RefCell;
//}}}
//{{{ dep imports
use ctor::ctor;
use topohedral_linalg::{
    MatMul,
    dvector::{DVector, VecType},
    dmatrix::{ DMatrix},
    scvector::SCVector,
    smatrix::{SMatrix},
    GreaterThan, VectorOps
};
use approx::assert_relative_eq;
use topohedral_tracing::*;

//}}}


//{{{ fun: init_logger
#[ctor]
fn init_logger() {
    init().unwrap();
}
//}}}


#[derive(Debug, Clone)]
struct QuadraticDynamic {
    n: usize,
    center: DVector<f64>,
    coeffs: DMatrix<f64>,
}

//{{{ collection Quadratic1D
#[derive(Debug, Clone)]
struct Quadratic1D{
    pub root1: f64, 
    pub root2: f64,
}
impl Quadratic1D{

    pub fn extrema(&self) -> f64 {
        (self.root1 + self.root2) / 2.0
    }
}
impl RealFn1 for Quadratic1D {
    fn eval(&mut self, x: f64) -> f64 {
        (x - self.root1) * (x - self.root2)
    }

    fn diff(&mut self, x: f64) -> f64 {
        2.0 * x - (self.root1 + self.root2)
    }
}
//}}}
//{{{ collection: Cubic1D
#[derive(Debug, Clone)]
struct Cubic1D {
    pub root1: f64, 
    pub root2: f64, 
    pub root3: f64
}
impl Cubic1D{

    fn extrema(&self) -> [f64; 2] {
        let (r1, r2, r3) = (self.root1, self.root2, self.root3);
        let a = 3.0;
        let b = -2.0 * (r1 + r2 + r3);
        let c = r1 * r2 + r1 * r3 + r2*r3;
        let v1 = (-b / (2.0 * a)) - ((b.powi(2) -  4.0 * a * c).sqrt() / (2.0*a));
        let v2 = (-b / (2.0 * a)) + ((b.powi(2) -  4.0 * a * c).sqrt() / (2.0*a));
        [v1, v2]
    }
}
impl RealFn1 for Cubic1D{
    fn eval(&mut self, x: f64) -> f64 {
        (x - self.root1) * (x - self.root2) * (x - self.root3)
    }

    fn diff(&mut self, x: f64) -> f64 {
        let mut out = 0.0;
        out += (x - self.root2) * (x - self.root3);
        out += (x - self.root1) * (x - self.root3);
        out += (x - self.root1) * (x - self.root2);
        out
    }
}
//}}}


#[test]
fn test_quadratic_1d() {
    let mut q1 = Quadratic1D{ root1: 1.0, root2: 2.0};
    let mut interp =  Interp::new(q1.clone(), 
        &InterpOptions{
            ls_opts: LineSearchOptions{
                c1: 1e-4, c2: 0.9
            },
            step1: 0.1, step2: 0.7
        } 
    );
    let alpha = 0.0; 
    let phi0 = q1.eval(alpha);
    let dphi0 = q1.diff(alpha);
    let out = interp.search(phi0, dphi0).unwrap();
    let exp_alpha = q1.extrema();
    assert_relative_eq!(out.alpha, exp_alpha, epsilon = 1e-10);
}

#[test]
fn test_cubic_1d() {
    let mut c1 = Cubic1D{root1: -1.0, root2: 0.0, root3: 1.0};
    let mut interp =  Interp::new(c1.clone(), 
        &InterpOptions{
            ls_opts: LineSearchOptions{
                c1: 1e-4, c2: 0.9
            },
            step1: 0.1, step2: 0.7
        } 
    );
    let alpha = 0.0; 
    let phi0 = c1.eval(alpha);
    let dphi0 = c1.diff(alpha);
    let out = interp.search(phi0, dphi0).unwrap();
    let exp_alpha = c1.extrema()[1];
    assert_relative_eq!(out.alpha, exp_alpha, epsilon = 1e-10);
}