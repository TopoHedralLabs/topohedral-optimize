//! Purpose of these tests are to test the function traits of the library and see if they are
//! compatible with a range of common vector types.
//!
//--------------------------------------------------------------------------------------------------
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(impl_trait_in_assoc_type)]

//{{{ crate imports
use topohedral_optimize::{line_search::LineSearchFcn, RealFn, RealFn1};
//}}}
//{{{ std imports
use std::{rc::Rc, sync::Mutex};
use std::sync::Arc;
use std::cell::RefCell;
//}}}
//{{{ dep imports
use topohedral_linalg::{
    MatMul,
    scvector::SCVector,
    smatrix::{SMatrix},
    GreaterThan, VectorOps
};
use approx::assert_relative_eq;
//}}}
//--------------------------------------------------------------------------------------------------
//{{{ colleciton: QuadraticStatic
//{{{ struct: QadraticStatic
#[allow(clippy::identity_op)]
#[derive(Debug, Clone)]
struct QuadraticStatic<const N: usize>
where
    [(); N * 1]:,
    [(); N * N]:,
{
    center: SCVector<f64, N>,
    coeffs: SMatrix<f64, N, N>,
}
//}}}
//{{{ impl: RealFn for QuadraticStatic
impl<const N: usize> RealFn for QuadraticStatic<N>
where
    [(); N * 1]:,
    [(); N * N]:,
    (): GreaterThan<N, 1>,
{
    type Vector = SCVector<f64, N>;

    fn eval(&mut self, x: &Self::Vector) -> f64 {
        let x1 = *x - self.center;
        let x2 = self.coeffs.matmul(&x1);
        x1.dot(&x2)
    }

    fn grad(&mut self, x: &Self::Vector) -> Self::Vector {
        let mut out = SCVector::zeros();
        // first term
        for i in 0..N {
            out[i] = 2.0 * self.coeffs[(i, i)] * x[i];
            for j in 0..N {
                if i != j {
                    out[i] += (self.coeffs[(i, j)] + self.coeffs[(j, i)]) * x[j];
                }
            }
        }
        // second and third terms
        for i in 0..N {
            for j in 0..N {
                out[i] +=
                    self.center[j] * self.coeffs[(j, i)] + self.coeffs[(i, j)] * self.center[j];
            }
        }
        out
    }
}
//}}}
//{{{ impl QuadraticStatic<3>
impl QuadraticStatic<3>
{
    fn new1() -> Self {

        let center = SCVector::<f64, 3>::zeros();
        let coeffs = SMatrix::<f64, 3, 3>::from_row_slice(&[
            5.0, 1.0, 2.0,
            1.0, 5.0, 3.0,
            2.0, 3.0, 5.0,
        ]);
        Self { center, coeffs }
    }

    fn update_center(&mut self, new_center: SCVector<f64, 3>)
    {
        self.center = new_center;
    }
}
//}}}
//{{{ test: test_quadratic_static_3d
#[test]
fn test_quadratic_static_3d()
{
    let mut f = QuadraticStatic::<3>::new1();

    let x1 = SCVector::<f64, 3>::zeros();
    let fx1 = f.eval(&x1);
    assert_relative_eq!(fx1, 0.0, epsilon = 1e-10);
    let grad_fx1 = f.grad(&x1);
    let exp_grad_fx1 = SCVector::<f64, 3>::zeros();
    for (actual, expected) in grad_fx1.iter().zip(exp_grad_fx1.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-10);
    }
    

    let x2 = SCVector::<f64, 3>::ones();
    let fx2 = f.eval(&x2);
    assert_relative_eq!(fx2, 27.0);
    let grad_fx2 = f.grad(&x2);
    let exp_grad_fx2 = SCVector::<f64, 3>::from_col_slice(&[16.0, 18.0, 20.0]);
    for (actual, expected) in grad_fx2.iter().zip(exp_grad_fx2.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-10);
    }
}
//}}}
//{{{ test: test_quadratic_static_3d_line_search
#[test]
fn test_quadratic_static_3d_line_search() {

    let mut line_fcn1 = LineSearchFcn {
        f:  QuadraticStatic::<3>::new1(),
        x: SCVector::<f64, 3>::zeros(),
        dir:  SCVector::<f64, 3>::from_col_slice(&[1.0, -2.0, 1.0]),
    };


    let phi1 = line_fcn1.eval(0.0);
    let dphi1 = line_fcn1.diff(0.0);
    assert_relative_eq!(phi1, 0.0, epsilon = 1e-10);
    assert_relative_eq!(dphi1, 0.0, epsilon = 1e-10);


    line_fcn1.x = SCVector::<f64, 3>::ones();
    let phi2 = line_fcn1.eval(0.0);
    let dphi2 = line_fcn1.diff(0.0);
    assert_relative_eq!(phi2, 27.0, epsilon = 1e-10);
    assert_relative_eq!(dphi2, 16.0 - 2.0*18.0 + 20.0, epsilon = 1e-10);

}
//}}}
//{{{ test: test_quadratic_static_rc_line_search
#[test]
fn test_quadratic_static_rc_line_search() {

    let fcn1 = Rc::new(RefCell::new(QuadraticStatic::<3>::new1()));
    let x = SCVector::<f64, 3>::zeros();
    let dir = SCVector::<f64, 3>::from_col_slice(&[1.0, -2.0, 1.0]);
    let mut line_fcn1 = LineSearchFcn {
        f: fcn1.clone(), 
        x, 
        dir
    };

    let phi1 = line_fcn1.eval(0.0);
    let dphi1 = line_fcn1.diff(0.0);
    assert_relative_eq!(phi1, 0.0, epsilon = 1e-10);
    assert_relative_eq!(dphi1, 0.0, epsilon = 1e-10);


    line_fcn1.x = SCVector::<f64, 3>::ones();
    let phi2 = line_fcn1.eval(0.0);
    let dphi2 = line_fcn1.diff(0.0);
    assert_relative_eq!(phi2, 27.0, epsilon = 1e-10);
    assert_relative_eq!(dphi2, 16.0 - 2.0*18.0 + 20.0, epsilon = 1e-10);

    // check that this has alterned same underlying memory

}
//}}}
//{{{ test: test_quadratic_static_arc_line_search
#[test]
fn test_quadratic_static_arc_line_search() {

    let fcn1 = Arc::new(Mutex::new(QuadraticStatic::<3>::new1()));

    let x = SCVector::<f64, 3>::zeros();
    let dir = SCVector::<f64, 3>::from_col_slice(&[1.0, -2.0, 1.0]);
    let mut line_fcn1 = LineSearchFcn {
        f: fcn1.clone(), 
        x, 
        dir
    };

    let phi1 = line_fcn1.eval(0.0);
    let dphi1 = line_fcn1.diff(0.0);
    assert_relative_eq!(phi1, 0.0, epsilon = 1e-10);
    assert_relative_eq!(dphi1, 0.0, epsilon = 1e-10);


    line_fcn1.x = SCVector::<f64, 3>::ones();
    let phi2 = line_fcn1.eval(0.0);
    let dphi2 = line_fcn1.diff(0.0);
    assert_relative_eq!(phi2, 27.0, epsilon = 1e-10);
    assert_relative_eq!(dphi2, 16.0 - 2.0*18.0 + 20.0, epsilon = 1e-10);


}
//}}}
//}}}
//{{{ collection: QuadraticDynamic
//{{{ struct: QuadraticDynamic
//}}}
//}}}
