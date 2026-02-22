//! Purpose of these tests are to test the function traits of the library with dynamic vectors.
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use topohedral_optimize::{line_search::LineSearchFcn, RealFn, RealFn1};
//}}}
//{{{ std imports
use std::cell::RefCell;
use std::sync::Arc;
use std::{rc::Rc, sync::Mutex};
//}}}
//{{{ dep imports
use approx::assert_relative_eq;
use topohedral_linalg::dmatrix::DMatrix;
use topohedral_linalg::dvector::{DVector, VecType};
use topohedral_linalg::{MatMul, VectorOps};
//}}}
//--------------------------------------------------------------------------------------------------

fn colvec(values: &[f64]) -> DVector<f64>
{
    DVector::<f64>::from_slice_vec(values, values.len(), VecType::Col)
}

//{{{ struct: QuadraticDynamic
#[derive(Debug, Clone)]
struct QuadraticDynamic
{
    center: DVector<f64>,
    coeffs: DMatrix<f64>,
}
//}}}
//{{{ impl: RealFn for QuadraticDynamic
impl RealFn for QuadraticDynamic
{
    fn eval(
        &mut self,
        x: &DVector<f64>,
    ) -> f64
    {
        let x1 = x.clone() - self.center.clone();
        let x2 = (&self.coeffs).matmul(&x1);
        x1.dot(&x2)
    }

    fn grad(
        &mut self,
        x: &DVector<f64>,
    ) -> DVector<f64>
    {
        let n = x.len();
        let mut out = DVector::<f64>::zeros_cvec(n, VecType::Col);

        for i in 0..n
        {
            out[i] = 2.0 * self.coeffs[(i, i)] * x[i];
            for j in 0..n
            {
                if i != j
                {
                    out[i] += (self.coeffs[(i, j)] + self.coeffs[(j, i)]) * x[j];
                }
            }
        }

        for i in 0..n
        {
            for j in 0..n
            {
                out[i] +=
                    self.center[j] * self.coeffs[(j, i)] + self.coeffs[(i, j)] * self.center[j];
            }
        }

        out
    }
}
//}}}
//{{{ impl: QuadraticDynamic
impl QuadraticDynamic
{
    fn new1() -> Self
    {
        let center = DVector::<f64>::zeros_cvec(3, VecType::Col);
        let coeffs =
            DMatrix::<f64>::from_row_slice(&[5.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 5.0], 3, 3);
        Self { center, coeffs }
    }
}
//}}}

//{{{ test: test_quadratic_dynamic_3d
#[test]
fn test_quadratic_dynamic_3d()
{
    let mut f = QuadraticDynamic::new1();

    let x1 = DVector::<f64>::zeros_cvec(3, VecType::Col);
    let fx1 = f.eval(&x1);
    assert_relative_eq!(fx1, 0.0, epsilon = 1e-10);
    let grad_fx1 = f.grad(&x1);
    let exp_grad_fx1 = DVector::<f64>::zeros_cvec(3, VecType::Col);
    for (actual, expected) in grad_fx1.iter().zip(exp_grad_fx1.iter())
    {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-10);
    }

    let x2 = DVector::<f64>::ones_cvec(3, VecType::Col);
    let fx2 = f.eval(&x2);
    assert_relative_eq!(fx2, 27.0);
    let grad_fx2 = f.grad(&x2);
    let exp_grad_fx2 = colvec(&[16.0, 18.0, 20.0]);
    for (actual, expected) in grad_fx2.iter().zip(exp_grad_fx2.iter())
    {
        assert_relative_eq!(*actual, *expected, epsilon = 1e-10);
    }
}
//}}}
//{{{ test: test_quadratic_dynamic_3d_line_search
#[test]
fn test_quadratic_dynamic_3d_line_search()
{
    let mut line_fcn1 = LineSearchFcn {
        f: QuadraticDynamic::new1(),
        x: DVector::<f64>::zeros_cvec(3, VecType::Col),
        dir: colvec(&[1.0, -2.0, 1.0]),
    };

    let phi1 = line_fcn1.eval(0.0);
    let dphi1 = line_fcn1.diff(0.0);
    assert_relative_eq!(phi1, 0.0, epsilon = 1e-10);
    assert_relative_eq!(dphi1, 0.0, epsilon = 1e-10);

    line_fcn1.x = DVector::<f64>::ones_cvec(3, VecType::Col);
    let phi2 = line_fcn1.eval(0.0);
    let dphi2 = line_fcn1.diff(0.0);
    assert_relative_eq!(phi2, 27.0, epsilon = 1e-10);
    assert_relative_eq!(dphi2, 16.0 - 2.0 * 18.0 + 20.0, epsilon = 1e-10);
}
//}}}
//{{{ test: test_quadratic_dynamic_rc_line_search
#[test]
fn test_quadratic_dynamic_rc_line_search()
{
    let fcn1 = Rc::new(RefCell::new(QuadraticDynamic::new1()));
    let x = DVector::<f64>::zeros_cvec(3, VecType::Col);
    let dir = colvec(&[1.0, -2.0, 1.0]);
    let mut line_fcn1 = LineSearchFcn {
        f: fcn1.clone(),
        x,
        dir,
    };

    let phi1 = line_fcn1.eval(0.0);
    let dphi1 = line_fcn1.diff(0.0);
    assert_relative_eq!(phi1, 0.0, epsilon = 1e-10);
    assert_relative_eq!(dphi1, 0.0, epsilon = 1e-10);

    line_fcn1.x = DVector::<f64>::ones_cvec(3, VecType::Col);
    let phi2 = line_fcn1.eval(0.0);
    let dphi2 = line_fcn1.diff(0.0);
    assert_relative_eq!(phi2, 27.0, epsilon = 1e-10);
    assert_relative_eq!(dphi2, 16.0 - 2.0 * 18.0 + 20.0, epsilon = 1e-10);
}
//}}}
//{{{ test: test_quadratic_dynamic_arc_line_search
#[test]
fn test_quadratic_dynamic_arc_line_search()
{
    let fcn1 = Arc::new(Mutex::new(QuadraticDynamic::new1()));

    let x = DVector::<f64>::zeros_cvec(3, VecType::Col);
    let dir = colvec(&[1.0, -2.0, 1.0]);
    let mut line_fcn1 = LineSearchFcn {
        f: fcn1.clone(),
        x,
        dir,
    };

    let phi1 = line_fcn1.eval(0.0);
    let dphi1 = line_fcn1.diff(0.0);
    assert_relative_eq!(phi1, 0.0, epsilon = 1e-10);
    assert_relative_eq!(dphi1, 0.0, epsilon = 1e-10);

    line_fcn1.x = DVector::<f64>::ones_cvec(3, VecType::Col);
    let phi2 = line_fcn1.eval(0.0);
    let dphi2 = line_fcn1.diff(0.0);
    assert_relative_eq!(phi2, 27.0, epsilon = 1e-10);
    assert_relative_eq!(dphi2, 16.0 - 2.0 * 18.0 + 20.0, epsilon = 1e-10);
}
//}}}
