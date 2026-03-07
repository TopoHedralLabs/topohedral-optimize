//! Purpose of these tests are to test the function traits of the library with dynamic vectors.
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use topohedral_optimize::{
    arc_real_vector_fn, line_search::LineSearchFcn, rc_real_vector_fn, ArcRealVectorFn, RealFn,
    RealFn1, RealVectorFn, RcRealVectorFn, Vector,
};

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

fn colvec(values: &[f64]) -> Vector
{
    DVector::<f64>::from_slice_vec(values, values.len(), VecType::Col)
}

fn assert_vector_close(
    actual: &Vector,
    expected: &Vector,
)
{
    for (a, e) in actual.iter().zip(expected.iter())
    {
        assert_relative_eq!(*a, *e, epsilon = 1e-10);
    }
}

fn assert_matrix_close(
    actual: &DMatrix<f64>,
    expected: &DMatrix<f64>,
    rows: usize,
    cols: usize,
)
{
    for i in 0..rows
    {
        for j in 0..cols
        {
            assert_relative_eq!(actual[(i, j)], expected[(i, j)], epsilon = 1e-10);
        }
    }
}

//{{{ struct: QuadraticDynamic
#[derive(Debug, Clone)]
struct QuadraticDynamic
{
    center: Vector,
    coeffs: DMatrix<f64>,
}
//}}}
//{{{ impl: RealFn for QuadraticDynamic
impl RealFn for QuadraticDynamic
{
    fn dimension(&self) -> usize
    {
        self.center.len()
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        let x1 = x.clone() - self.center.clone();
        let x2 = (&self.coeffs).matmul(&x1);
        x1.dot(&x2)
    }

    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector
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

//{{{ struct: LinearVectorDynamic
#[derive(Debug, Clone)]
struct LinearVectorDynamic
{
    jacobian: DMatrix<f64>,
    bias: Vector,
    rows: usize,
    cols: usize,
}
//}}}
//{{{ impl: RealVectorFn for LinearVectorDynamic
impl RealVectorFn for LinearVectorDynamic
{
    fn eval(
        &mut self,
        x: &Vector,
        val: &mut Vector,
    )
    {
        for i in 0..self.rows
        {
            val[i] = self.bias[i];
            for j in 0..self.cols
            {
                val[i] += self.jacobian[(i, j)] * x[j];
            }
        }
    }

    fn grad(
        &mut self,
        _x: &Vector,
        val: &mut DMatrix<f64>,
    )
    {
        for i in 0..self.rows
        {
            for j in 0..self.cols
            {
                val[(i, j)] = self.jacobian[(i, j)];
            }
        }
    }
}
//}}}
//{{{ impl: LinearVectorDynamic
impl LinearVectorDynamic
{
    fn new1() -> Self
    {
        let rows = 2;
        let cols = 3;
        let jacobian =
            DMatrix::<f64>::from_row_slice(&[1.0, -2.0, 0.5, -1.0, 3.0, 4.0], rows, cols);
        let bias = colvec(&[0.5, -1.0]);
        Self {
            jacobian,
            bias,
            rows,
            cols,
        }
    }
}
//}}}
//{{{ fun: run_linear_vector_checks
fn run_linear_vector_checks<F: RealVectorFn>(mut f: F)
{
    let x = colvec(&[2.0, -1.0, 3.0]);
    let mut value = DVector::<f64>::zeros_cvec(2, VecType::Col);
    let mut jac = DMatrix::<f64>::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 3);
    let exp_value = colvec(&[6.0, 6.0]);
    let exp_jac = DMatrix::<f64>::from_row_slice(&[1.0, -2.0, 0.5, -1.0, 3.0, 4.0], 2, 3);

    f.eval(&x, &mut value);
    f.grad(&x, &mut jac);

    assert_vector_close(&value, &exp_value);
    assert_matrix_close(&jac, &exp_jac, 2, 3);
}
//}}}
//{{{ test: test_linear_vector_dynamic
#[test]
fn test_linear_vector_dynamic()
{
    run_linear_vector_checks(LinearVectorDynamic::new1());
}
//}}}
//{{{ test: test_linear_vector_dynamic_rc
#[test]
fn test_linear_vector_dynamic_rc()
{
    let fcn: RcRealVectorFn<LinearVectorDynamic> =
        rc_real_vector_fn(LinearVectorDynamic::new1());
    run_linear_vector_checks(fcn);
}
//}}}
//{{{ test: test_linear_vector_dynamic_arc
#[test]
fn test_linear_vector_dynamic_arc()
{
    let fcn: ArcRealVectorFn<LinearVectorDynamic> =
        arc_real_vector_fn(LinearVectorDynamic::new1());
    run_linear_vector_checks(fcn);
}
//}}}
