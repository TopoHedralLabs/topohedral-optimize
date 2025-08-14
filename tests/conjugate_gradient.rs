#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(impl_trait_in_assoc_type)]

//{{{ crate imports
use topohedral_optimize::{RealFn};
use topohedral_optimize::line_search::{InterpOptions, LineSearchOptions, LineSearchMethod};
use topohedral_optimize::unconstrained::{UnconstrainedMinimizer, UnonstrainedOptions, ConjugateGradient, ConjugateGradientOptions, Direction};
//}}}
//{{{ std imports
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



//{{{ struct: Quadratic
#[derive(Debug, Clone, Copy)]
struct Quadratic
{
    xmin: SCVector<f64, 5>,
}
//}}}
//{{{ impl: RealFn for Quadratic
impl RealFn for Quadratic {

    type Vector = SCVector<f64, 5>;

    fn eval(&mut self, x: &Self::Vector) -> f64 {
        
        let tmp: Self::Vector = (x - &self.xmin).into();
        let mut out = 0.0;
        for i in 0..5 {
            out += tmp[i].powi(2);
        }
        out
    }

    fn grad(&mut self, x_in: &Self::Vector) -> Self::Vector {
        
        let tmp: Self::Vector = (x_in - &self.xmin).into();
        let mut out = Self::Vector::zeros(); 
        for i in 0..5 {
            out[i] = 2.0 * tmp[i];
        }
        out
    }
}
//}}}

#[test]
fn test_quadratic() {

    let quad = Quadratic{
        xmin: SCVector::<f64, 5>::from_col_slice(&[1000.0, -100.0, 0.0 , 567.0, -23.0])
    };

    let x0 = SCVector::<f64, 5>::from_col_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);

    let mut cg = ConjugateGradient::new(quad, x0, ConjugateGradientOptions{
        uncon_opts: UnonstrainedOptions{
            grad_rtol: 1e-6, 
            grad_atol: 1e-8,
            max_iter: 100,
            ls_method: LineSearchMethod::Interp(InterpOptions{
                ls_opts: LineSearchOptions {
                    c1: 1e-4, 
                    c2: 0.4,
                },
                step1: 0.5, 
                step2: 1.0,
                scale_factor: 1.5
            })
        }, 
        direction: Direction::Steepest, 
        restart: 10,
    });

    let ret = cg.minimize().unwrap();
    
    print!("{ret:?}")


}
//{{{ struct: Quartic 
#[derive(Debug, Clone, Copy)]
struct Quartic {
    xmin: SCVector<f64, 5>
}
//}}}
//{{{ impl: RealFn for Quartic
impl RealFn for Quartic {
    type Vector = SCVector<f64, 5>;

    fn eval(&mut self, x: &Self::Vector) -> f64 {
        let tmp: Self::Vector = (x - &self.xmin).into();
        let mut out = 0.0;
        for i in 0..5 {
            out += tmp[i].powi(4);
        }
        out
    }

    fn grad(&mut self, x_in: &Self::Vector) -> Self::Vector {
        let tmp: Self::Vector = (x_in - &self.xmin).into();
        let mut out = Self::Vector::zeros();
        for i in 0..5 {
            out[i] = 4.0 * tmp[i].powi(3);
        }
        out
    }
}
//}}}

#[test]
fn test_quartic() {

    let quart = Quartic{
        xmin: SCVector::<f64, 5>::from_col_slice(&[10.0, 10.0, 10.0, 10.0, 10.0])
    };

    let scale_factor = 20.0;
    let jump = 1.0;
    let mut offset_dir = SCVector::<f64, 5>::from_col_slice(&[1e-3, 1.0, 0.5, 3.0, 1.0]);
    offset_dir = offset_dir.normalize();

    let x0 = quart.xmin.clone() + scale_factor * offset_dir;

    let mut cg = ConjugateGradient::new(quart, x0, ConjugateGradientOptions{
        uncon_opts: UnonstrainedOptions{
            grad_rtol: 1e-6, 
            grad_atol: 1e-8,
            max_iter: 1000,
            ls_method: LineSearchMethod::Interp(InterpOptions{
                ls_opts: LineSearchOptions {
                    c1: 1e-4, 
                    c2: 0.4,
                },
                step1: 0.5 * jump, 
                step2: jump ,
                scale_factor: 1.5
            })
        }, 
        direction: Direction::Steepest, 
        restart: 100,
    });

    let ret = cg.minimize().unwrap();
    
    print!("{ret:?}")


}
