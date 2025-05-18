//! Purpose of these tests are to test the function traits of the library and see if they are
//! compatible with a range of common vector types.
//!
//--------------------------------------------------------------------------------------------------
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(impl_trait_in_assoc_type)]

use approx::assert_relative_eq;
//{{{ crate imports
//}}}
//{{{ std imports
use std::rc::Rc;
//}}}
//{{{ dep imports
use topohedral_linalg::{
    MatMul,
    dvector::{DVector, VecType},
    dmatrix::{DMatrix, EvaluateDMatrix},
    scvector::SCVector,
    smatrix::SMatrix,
    GreaterThan, VectorOps
};
use topohedral_optimize::{RealFn, line_search::LineSearchFcn};
use std::cell::RefCell;
//}}}
//--------------------------------------------------------------------------------------------------

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
            out[i] = 2.0 * self.coeffs[(i, i)];
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
    let x = SCVector::<f64, 3>::zeros();
    let fx = f.eval(&x);
    assert_relative_eq!(fx, 0.0, epsilon = 1e-10);
}
//}}}
//{{{ test: test_quadratic_static_3d_line_search
#[test]
fn test_quadratic_static_3d_line_search() {

    let center = SCVector::<f64, 3>::from_col_slice(&[1.0, 2.0, 3.0]);
    let coeffs = SMatrix::<f64, 3, 3>::from_row_slice(&[
        1.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 3.0,
    ]);

    let mut f = QuadraticStatic::<3> {
        center,
        coeffs,
    };

    let x = SCVector::<f64, 3>::from_col_slice(&[1.0, 2.0, 3.0]);
    let dir = SCVector::<f64, 3>::from_col_slice(&[1.0, 1.0, 1.0]);
    let mut line_search = LineSearchFcn {
        f: f.clone(),
        x,
        dir,
    };

    f.eval(&x);
}
//}}}
#[test]
fn test_quadratic_staticstateful_line_search() {

    let fcn1 = Rc::new(RefCell::new(QuadraticStatic::<3>::new1()));
    let x = SCVector::<f64, 3>::from_col_slice(&[1.0, 2.0, 3.0]);
    let dir = SCVector::<f64, 3>::from_col_slice(&[1.0, 1.0, 1.0]);
    let line_fcn1 = LineSearchFcn {
        f: fcn1.clone(), 
        x, 
        dir
    };





}
//{{{ struct: QuadraticDynamic
struct QuadraticDynamic {
    n: usize,
    center: DVector<f64>,
    coeffs: DMatrix<f64>,
}
//}}}
//{{{ impl RealFn for QuadraticDynamic
impl RealFn for QuadraticDynamic
{
    type Vector = DVector<f64>;

    fn eval(&mut self, x: &Self::Vector) -> f64 {
        assert_eq!(x.len(), self.center.len());
        let x1 = (x - &self.center).evald();
        let x2 = self.coeffs.matmul(&x1);
        x1.dot(&x2)
    }

    fn grad(&mut self, x: &Self::Vector) -> Self::Vector {
        let mut out = DVector::<f64>::zeros(self.n, 1);
        // first term
        for i in 0..self.n {
            out[i] = 2.0 * self.coeffs[(i, i)];
            for j in 0..self.n {
                if i != j {
                    out[i] += (self.coeffs[(i, j)] + self.coeffs[(j, i)]) * x[j];
                }
            }
        }
        // second and third terms
        for i in 0..self.n {
            for j in 0..self.n {
                out[i] +=
                    self.center[j] * self.coeffs[(j, i)] + self.coeffs[(i, j)] * self.center[j];
            }
        }
        out
    }
}
//}}}
//{{{ test: test_quadratic_dynamic
#[test]
fn test_quadratic_dynamic_3d()
{
    let center = DVector::<f64>::from_slice_vec(&[1.0, 2.0, 3.0], 3, VecType::Col);
    let coeffs = DMatrix::<f64>::from_row_slice(&[
        1.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 3.0,
    ], 3, 3);

    let mut f = QuadraticDynamic {
        n: 3,
        center,
        coeffs,
    };

    let x = DVector::<f64>::from_slice_vec(&[1.0, 2.0, 3.0], 3, VecType::Col);
    let fx = f.eval(&x);
    assert_relative_eq!(fx, 0.0, epsilon = 1e-10);

}
//}}}

