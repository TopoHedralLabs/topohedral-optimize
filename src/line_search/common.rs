//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{RealFn, RealFn1, Vector};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use thiserror::Error;
use topohedral_linalg::VectorOps;
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct: LineSearchFcn
#[derive(Debug, Clone)]
pub struct LineSearchFcn<F: RealFn> {
    pub f: F,
    pub x: Vector,
    pub dir: Vector,
}
//}}}
//{{{ impl: LineSearchFcn
impl<F: RealFn> LineSearchFcn<F> {
    #[trace_fn]
    pub fn new(
        f: F,
        x: Vector,
        dir: Vector,
    ) -> Self {
        Self { f, x, dir }
    }

    #[trace_fn]
    pub fn output_data(
        &mut self,
        alpha: f64,
    ) -> (Vector, f64, Vector) {
        let new_x: Vector = (&self.x + alpha * &self.dir).into();
        let new_fx = self.f.eval(&new_x);
        let new_grad_fx = self.f.derivative(&new_x);
        (new_x, new_fx, new_grad_fx)
    }
}
//}}}
//{{{ impl: RealFn1 for LineSearchFcn
impl<F: RealFn> crate::DifferentiableFn for LineSearchFcn<F> {
    type Input = f64;
    type Output = f64;
    type Derivative = f64;
    #[trace_fn]
    fn eval(
        &mut self,
        alpha: &f64,
    ) -> f64 {
        let x = self.x.clone() + *alpha * self.dir.clone();
        self.f.eval(&x)
    }

    #[trace_fn]
    fn derivative(
        &mut self,
        alpha: &f64,
    ) -> f64 {
        let x = self.x.clone() + *alpha * self.dir.clone();
        let grad = self.f.derivative(&x);
        grad.dot(&self.dir)
    }
}
//}}}
//{{{ enum: Error
#[derive(PartialEq, Error, Debug)]
pub enum Error {
    #[error("Not decreasing")]
    NotDecreasing,
    #[error("Fails Armijo condition")]
    Armijo,
    #[error("Fails curvature condition")]
    Curvature,
    #[error("Max iterations reached")]
    MaxIterations,
    #[error("Step size too small")]
    StepSizeSmall,
    #[error("Step size too large")]
    StepSizeLarge,
    #[error("No step found")]
    NoStepFound,
}
//}}}
//{{{ struct: Options
/// Options for configuring a line search algorithm.
///
/// This struct contains the parameters needed to configure a line search algorithm,
/// such as the initial function value (`phi0`), the initial derivative value (`dphi0`),
/// and the Armijo and curvature conditions (`c1` and `c2`). The `method` field
/// specifies the line search method to use, which can be one of `FixedStep`, `Quadratic`,
/// or `Inexact`.
#[derive(Debug, Copy, Clone)]
pub struct Options {
    pub c1: f64,
    pub c2: f64,
    pub step_min: f64,
    pub step_max: f64,
}
//}}}
//{{{ impl: Default for Options
impl Default for Options {
    #[trace_fn]
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            step_min: 0.0,
            step_max: 50.0,
        }
    }
}
//}}}
//{{{ struct: Returns
/// The results of a line search algorithm.
///
/// This struct contains the following fields:
/// - `alpha`: The step size found by the line search.
/// - `phi_alpha`: The function value at the step size `alpha`.
#[derive(Debug, Clone)]
pub struct Returns {
    pub alpha: f64,
    pub phi_alpha: f64,
}
//}}}
//{{{ trait: LineSearch
pub trait LineSearch {
    type Function: RealFn1;
    fn search(
        &mut self,
        phi0: f64,
        dphi0: f64,
        alpha1: f64,
    ) -> Result<Returns, Error>;
}
//}}}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::DifferentiableFn;

    //{{{ std imports
    use std::cell::RefCell;
    use std::sync::Arc;
    use std::{rc::Rc, sync::Mutex};
    //}}}
    //{{{ dep imports
    use approx::assert_relative_eq;
    use topohedral_linalg::DMatrix;
    use topohedral_linalg::{DVector, VecType};
    use topohedral_linalg::{MatMul, VectorOps};
    //}}}

    #[trace_fn]
    fn colvec(values: &[f64]) -> Vector {
        DVector::<f64>::from_slice_vec(values, values.len(), VecType::Col)
    }

    //{{{ struct: QuadraticDynamic
    #[derive(Debug, Clone)]
    struct QuadraticDynamic {
        center: Vector,
        coeffs: DMatrix<f64>,
    }
    //}}}
    //{{{ impl: RealFn for QuadraticDynamic
    impl DifferentiableFn for QuadraticDynamic {
        type Input = Vector;
        type Output = f64;
        type Derivative = Vector;
        #[trace_fn]
        fn dimension(&self) -> usize {
            self.center.len()
        }

        #[trace_fn]
        fn eval(
            &mut self,
            x: &Vector,
        ) -> f64 {
            let x1 = x.clone() - self.center.clone();
            let x2 = (&self.coeffs).matmul(&x1);
            x1.dot(&x2)
        }

        #[trace_fn]
        fn derivative(
            &mut self,
            x: &Vector,
        ) -> Vector {
            let n = x.len();
            let mut out = DVector::<f64>::zeros_vec(n, VecType::Col);

            for i in 0..n {
                out[i] = 2.0 * self.coeffs[(i, i)] * x[i];
                for j in 0..n {
                    if i != j {
                        out[i] += (self.coeffs[(i, j)] + self.coeffs[(j, i)]) * x[j];
                    }
                }
            }

            for i in 0..n {
                for j in 0..n {
                    out[i] +=
                        self.center[j] * self.coeffs[(j, i)] + self.coeffs[(i, j)] * self.center[j];
                }
            }

            out
        }
    }
    //}}}
    //{{{ impl: QuadraticDynamic
    impl QuadraticDynamic {
        #[trace_fn]
        fn new1() -> Self {
            let center = DVector::<f64>::zeros_vec(3, VecType::Col);
            let coeffs = DMatrix::<f64>::from_row_slice(
                &[5.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 5.0],
                3,
                3,
            );
            Self { center, coeffs }
        }
    }
    //}}}
    //{{{ test: test_quadratic_dynamic_3d_line_search
    #[test]
    #[trace_fn]
    fn test_quadratic_dynamic_3d_line_search() {
        let mut line_fcn1 = LineSearchFcn {
            f: QuadraticDynamic::new1(),
            x: DVector::<f64>::zeros_vec(3, VecType::Col),
            dir: colvec(&[1.0, -2.0, 1.0]),
        };

        let phi1 = line_fcn1.eval(&0.0);
        let dphi1 = line_fcn1.derivative(&0.0);
        assert_relative_eq!(phi1, 0.0, epsilon = 1e-10);
        assert_relative_eq!(dphi1, 0.0, epsilon = 1e-10);

        line_fcn1.x = DVector::<f64>::ones_vec(3, VecType::Col);
        let phi2 = line_fcn1.eval(&0.0);
        let dphi2 = line_fcn1.derivative(&0.0);
        assert_relative_eq!(phi2, 27.0, epsilon = 1e-10);
        assert_relative_eq!(dphi2, 16.0 - 2.0 * 18.0 + 20.0, epsilon = 1e-10);
    }
    //}}}
    //{{{ test: test_quadratic_dynamic_rc_line_search
    #[test]
    #[trace_fn]
    fn test_quadratic_dynamic_rc_line_search() {
        let fcn1 = Rc::new(RefCell::new(QuadraticDynamic::new1()));
        let x = DVector::<f64>::zeros_vec(3, VecType::Col);
        let dir = colvec(&[1.0, -2.0, 1.0]);
        let mut line_fcn1 = LineSearchFcn {
            f: fcn1.clone(),
            x,
            dir,
        };

        let phi1 = line_fcn1.eval(&0.0);
        let dphi1 = line_fcn1.derivative(&0.0);
        assert_relative_eq!(phi1, 0.0, epsilon = 1e-10);
        assert_relative_eq!(dphi1, 0.0, epsilon = 1e-10);

        line_fcn1.x = DVector::<f64>::ones_vec(3, VecType::Col);
        let phi2 = line_fcn1.eval(&0.0);
        let dphi2 = line_fcn1.derivative(&0.0);
        assert_relative_eq!(phi2, 27.0, epsilon = 1e-10);
        assert_relative_eq!(dphi2, 16.0 - 2.0 * 18.0 + 20.0, epsilon = 1e-10);
    }
    //}}}
    //{{{ test: test_quadratic_dynamic_arc_line_search
    #[test]
    #[trace_fn]
    fn test_quadratic_dynamic_arc_line_search() {
        let fcn1 = Arc::new(Mutex::new(QuadraticDynamic::new1()));

        let x = DVector::<f64>::zeros_vec(3, VecType::Col);
        let dir = colvec(&[1.0, -2.0, 1.0]);
        let mut line_fcn1 = LineSearchFcn {
            f: fcn1.clone(),
            x,
            dir,
        };

        let phi1 = line_fcn1.eval(&0.0);
        let dphi1 = line_fcn1.derivative(&0.0);
        assert_relative_eq!(phi1, 0.0, epsilon = 1e-10);
        assert_relative_eq!(dphi1, 0.0, epsilon = 1e-10);

        line_fcn1.x = DVector::<f64>::ones_vec(3, VecType::Col);
        let phi2 = line_fcn1.eval(&0.0);
        let dphi2 = line_fcn1.derivative(&0.0);
        assert_relative_eq!(phi2, 27.0, epsilon = 1e-10);
        assert_relative_eq!(dphi2, 16.0 - 2.0 * 18.0 + 20.0, epsilon = 1e-10);
    }
    //}}}
}
//}}}
