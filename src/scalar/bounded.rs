//! Bounded scalar minimization.
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::Error as ScalarError;
use crate::{
    common::{Minimizer, ScalarReturns},
    ConvergedReason, RealFn1,
};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::{trace, trace_fn};
//}}}
//--------------------------------------------------------------------------------------------------

const DEFUALT_XTOL: f64 = 1e-5;
const DEFUALT_MAX_ITER: usize = 100;

//{{{ struct: Options
#[derive(Copy, Clone)]
pub struct Options {
    /// Bounds
    pub bounds: (f64, f64),
    /// Absolute tolerance on `x` used as the termination criterion.
    pub xatol: f64,
    /// Maximum number of function evaluations.
    pub max_iter: usize,
}
//}}}
impl Options {
    pub fn new(
        lower: f64,
        upper: f64,
    ) -> Result<Self, ScalarError> {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(ScalarError::NonFiniteBounds(lower, upper));
        }
        if lower > upper {
            return Err(ScalarError::InvalidBounds(lower, upper));
        }
        Ok(Self {
            bounds: (lower, upper),
            xatol: DEFUALT_XTOL,
            max_iter: DEFUALT_MAX_ITER,
        })
    }
}
//{{{ struct: Bounded
/// Bounded minimization of a scalar function over a finite interval `[lower, upper]`.
pub struct Bounded<F: RealFn1> {
    fcn: F,
    opts: Options,
}
//}}}
//{{{ impl: Bounded
impl<F: RealFn1> Bounded<F> {
    #[trace_fn]
    pub fn new(
        fcn: F,
        opts: Options,
    ) -> Self {
        Self { fcn, opts }
    }
}
//}}}
//{{{ impl: Minimizer for Bounded
impl<F: RealFn1> Minimizer for Bounded<F> {
    type Error = ScalarError;
    type Returns = ScalarReturns;

    #[trace_fn]
    fn minimize(&mut self) -> Result<Self::Returns, Self::Error> {
        let sqrt_eps = 2.2e-16_f64.sqrt();
        let golden_mean = 0.5 * (3.0 - 5.0_f64.sqrt());

        // `a`/`b` are the shrinking bracket, initially the full bounds.
        let mut a = self.opts.bounds.0;
        let mut b = self.opts.bounds.1;
        let mut fulc = a + golden_mean * (b - a);
        let mut nfc = fulc;
        let mut xf = fulc;
        let mut rat = 0.0_f64;
        let mut e = 0.0_f64;
        let mut x = xf;
        let mut fx = self.fcn.eval(&x);
        let mut num = 1usize;
        let mut fu = f64::INFINITY;

        let mut ffulc = fx;
        let mut fnfc = fx;
        let mut xm = 0.5 * (a + b);
        let mut tol1 = sqrt_eps * xf.abs() + self.opts.xatol / 3.0;
        let mut tol2 = 2.0 * tol1;

        //{{{ trace
        trace!(target: "scalar", "bounded: searching [{a:.4e}, {b:.4e}], x0 = {xf:.4e}, f0 = {fx:.4e}");
        //}}}

        while (xf - xm).abs() > (tol2 - 0.5 * (b - a)) {
            let mut golden = true;

            // Check for parabolic fit.
            if e.abs() > tol1 {
                golden = false;
                let mut r = (xf - nfc) * (fx - ffulc);
                let mut q = (xf - fulc) * (fx - fnfc);
                let mut p = (xf - fulc) * q - (xf - nfc) * r;
                q = 2.0 * (q - r);
                if q > 0.0 {
                    p = -p;
                }
                q = q.abs();
                r = e;
                e = rat;

                // Check for acceptability of parabola.
                if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                    rat = p / q;
                    x = xf + rat;

                    if (x - a) < tol2 || (b - x) < tol2 {
                        let si = if xm >= xf { 1.0 } else { -1.0 };
                        rat = tol1 * si;
                    }
                } else {
                    golden = true;
                }
            }

            if golden {
                // Do a golden-section step.
                e = if xf >= xm { a - xf } else { b - xf };
                rat = golden_mean * e;
            }

            let si = if rat >= 0.0 { 1.0 } else { -1.0 };
            x = xf + si * rat.abs().max(tol1);
            fu = self.fcn.eval(&x);
            num += 1;

            if fu <= fx {
                if x >= xf {
                    a = xf;
                } else {
                    b = xf;
                }
                fulc = nfc;
                ffulc = fnfc;
                nfc = xf;
                fnfc = fx;
                xf = x;
                fx = fu;
            } else {
                if x < xf {
                    a = x;
                } else {
                    b = x;
                }
                if fu <= fnfc || nfc == xf {
                    fulc = nfc;
                    ffulc = fnfc;
                    nfc = x;
                    fnfc = fu;
                } else if fu <= ffulc || fulc == xf || fulc == nfc {
                    fulc = x;
                    ffulc = fu;
                }
            }

            xm = 0.5 * (a + b);
            tol1 = sqrt_eps * xf.abs() + self.opts.xatol / 3.0;
            tol2 = 2.0 * tol1;

            //{{{ trace
            trace!(target: "scalar", "bounded: eval {num}, xf = {xf:.4e}, fx = {fx:.4e}");
            //}}}

            if num >= self.opts.max_iter {
                //{{{ trace
                trace!(target: "scalar", "bounded: exceeded max_iter = {}", self.opts.max_iter);
                //}}}
                return Err(ScalarError::MaxIterations(self.opts.max_iter));
            }
        }

        if xf.is_nan() || fx.is_nan() || fu.is_nan() {
            //{{{ trace
            trace!(target: "scalar", "bounded: NaN encountered");
            //}}}
            return Err(ScalarError::NaN);
        }

        //{{{ trace
        trace!(target: "scalar", "bounded: converged, xmin = {xf:.4e}, fmin = {fx:.4e}, evals = {num}");
        //}}}

        Ok(ScalarReturns {
            xmin: xf,
            fmin: fx,
            reason: ConvergedReason::Atol,
            num_iterations: num,
            num_fun_evals: num,
            num_grad_evals: 0,
        })
    }
}
//}}}
