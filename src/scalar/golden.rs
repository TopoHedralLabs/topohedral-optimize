//! Unbounded scalar minimization via golden-section search.
//!
//! Ported from SciPy's `_minimize_scalar_golden` (`scipy/optimize/_optimize.py`):
//! an analog of the bisection method that shrinks a bracketing interval using
//! the golden ratio, without relying on parabolic interpolation.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::{
    resolve_bracket, Bracket, BracketOptions, BracketResult, Error as ScalarError,
};
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
const DEFUALT_MAX_ITER: usize = 5000;

//{{{ struct: Options
#[derive(Copy, Clone)]
pub struct Options {
    /// Initial bracket
    pub bracket: Bracket,
    /// Relative tolerance on `x` used as the termination criterion.
    pub xtol: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
}
//}}}
impl Options {
    pub fn new(bracket: Bracket) -> Self {
        Self {
            bracket,
            xtol: DEFUALT_XTOL,
            max_iter: DEFUALT_MAX_ITER,
        }
    }
}
//{{{ struct: Golden
/// Unbounded minimization of a scalar function using golden-section search.
///
/// Ported from SciPy's `_minimize_scalar_golden` (`scipy/optimize/_optimize.py`).
pub struct Golden<F: RealFn1> {
    fcn: F,
    opts: Options,
}
//}}}
//{{{ impl: Golden
impl<F: RealFn1> Golden<F> {
    #[trace_fn]
    pub fn new(
        fcn: F,
        opts: Options,
    ) -> Self {
        Self { fcn, opts }
    }
}
//}}}
//{{{ impl: Minimizer for Golden
impl<F: RealFn1> Minimizer for Golden<F> {
    type Error = ScalarError;
    type Returns = ScalarReturns;

    #[trace_fn]
    fn minimize(&mut self) -> Result<Self::Returns, Self::Error> {
        // Golden ratio conjugate: 2.0 / (1.0 + sqrt(5.0)).
        const GR: f64 = 0.61803399;
        const GC: f64 = 1.0 - GR;

        let BracketResult {
            xa,
            xb,
            xc,
            mut num_fun_evals,
            ..
        } = resolve_bracket(&mut self.fcn, self.opts.bracket, BracketOptions::default())?;

        //{{{ trace
        trace!(target: "scalar", "golden: bracket xa = {xa:.4e}, xb = {xb:.4e}, xc = {xc:.4e}");
        //}}}

        let mut x0 = xa;
        let mut x3 = xc;
        let (mut x1, mut x2);
        if (xc - xb).abs() > (xb - xa).abs() {
            x1 = xb;
            x2 = xb + GC * (xc - xb);
        } else {
            x2 = xb;
            x1 = xb - GC * (xb - xa);
        }
        let mut f1 = self.fcn.eval(&x1);
        let mut f2 = self.fcn.eval(&x2);
        num_fun_evals += 2;
        let mut nit = 0usize;

        while nit < self.opts.max_iter {
            if (x3 - x0).abs() <= self.opts.xtol * (x1.abs() + x2.abs()) {
                break;
            }

            if f2 < f1 {
                x0 = x1;
                x1 = x2;
                x2 = GR * x1 + GC * x3;
                f1 = f2;
                f2 = self.fcn.eval(&x2);
            } else {
                x3 = x2;
                x2 = x1;
                x1 = GR * x2 + GC * x0;
                f2 = f1;
                f1 = self.fcn.eval(&x1);
            }
            num_fun_evals += 1;

            //{{{ trace
            trace!(target: "scalar", "golden: iter {nit}, x1 = {x1:.4e}, x2 = {x2:.4e}, f1 = {f1:.4e}, f2 = {f2:.4e}");
            //}}}

            nit += 1;
        }

        let (xmin, fval) = if f1 < f2 { (x1, f1) } else { (x2, f2) };

        if nit >= self.opts.max_iter {
            //{{{ trace
            trace!(target: "scalar", "golden: exceeded max_iter = {}", self.opts.max_iter);
            //}}}
            return Err(ScalarError::MaxIterations(self.opts.max_iter));
        }

        if xmin.is_nan() || fval.is_nan() {
            //{{{ trace
            trace!(target: "scalar", "golden: NaN encountered");
            //}}}
            return Err(ScalarError::NaN);
        }

        //{{{ trace
        trace!(target: "scalar", "golden: converged, xmin = {xmin:.4e}, fmin = {fval:.4e}, evals = {num_fun_evals}");
        //}}}

        Ok(ScalarReturns {
            xmin,
            fmin: fval,
            reason: ConvergedReason::Rtol,
            num_iterations: nit,
            num_fun_evals,
            num_grad_evals: 0,
        })
    }
}
//}}}
