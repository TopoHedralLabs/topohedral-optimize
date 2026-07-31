//! Unbounded scalar minimization via Brent's method.
//!
//! Ported from SciPy's `Brent` class / `_minimize_scalar_brent`
//! (`scipy/optimize/_optimize.py`): inverse parabolic interpolation combined
//! with golden-section search, isolating a local minimum starting from a
//! bracketing triple.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::{
    resolve_bracket, Bracket, BracketOptions, BracketResult, Error as ScalarError,
};
use crate::{
    common::{validate_nonzero, validate_positive_finite, Minimizer, ScalarReturns},
    ConvergedReason, RealFn1, ValidationError,
};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::{trace, trace_fn};
//}}}
//--------------------------------------------------------------------------------------------------

const DEFAULT_XTOL: f64 = 1e-5;
const DEFAULT_MAX_ITER: usize = 100;

//{{{ struct: Options
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
/// Options for Brent's scalar minimizer.
pub struct Options {
    /// Initial bracket
    pub(crate) bracket: Bracket,
    /// Relative tolerance on `x` used as the termination criterion.
    pub(crate) xtol: f64,
    /// Maximum number of iterations.
    pub(crate) max_iter: usize,
}
//}}}
impl Options {
    /// Creates options using the supplied initial bracket.
    pub fn new(bracket: Bracket) -> Self {
        Self {
            bracket,
            xtol: DEFAULT_XTOL,
            max_iter: DEFAULT_MAX_ITER,
        }
    }

    /// Returns the initial bracket specification.
    pub const fn bracket(&self) -> Bracket {
        self.bracket
    }

    /// Returns the relative tolerance on the minimizer.
    pub const fn x_tolerance(&self) -> f64 {
        self.xtol
    }

    /// Returns the iteration limit.
    pub const fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns options with a different relative tolerance.
    pub const fn with_x_tolerance(
        mut self,
        tolerance: f64,
    ) -> Self {
        self.xtol = tolerance;
        self
    }

    /// Returns options with a different iteration limit.
    pub const fn with_max_iter(
        mut self,
        max_iter: usize,
    ) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Validates this configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError`] if the tolerance is non-positive or
    /// non-finite, or if the iteration limit is zero.
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_positive_finite("x_tolerance", self.xtol)?;
        validate_nonzero("max_iter", self.max_iter as u64)
    }
}
//{{{ struct: Brent
/// Unbounded minimization of a scalar function using Brent's method.
///
/// Ported from SciPy's `Brent` class / `_minimize_scalar_brent`
/// (`scipy/optimize/_optimize.py`).
pub struct Brent<F: RealFn1> {
    fcn: F,
    opts: Options,
}
//}}}
//{{{ impl: Brent
impl<F: RealFn1> Brent<F> {
    #[trace_fn]
    pub fn new(
        fcn: F,
        opts: Options,
    ) -> Self {
        Self { fcn, opts }
    }
}
//}}}
//{{{ impl: Minimizer for Brent
impl<F: RealFn1> Minimizer for Brent<F> {
    type Error = ScalarError;
    type Returns = ScalarReturns;

    #[trace_fn]
    fn minimize(&mut self) -> Result<Self::Returns, Self::Error> {
        const MINTOL: f64 = 1.0e-11;
        const CG: f64 = 0.3819660;

        let BracketResult {
            xa,
            xb,
            xc,
            fb,
            mut num_fun_evals,
            ..
        } = resolve_bracket(&mut self.fcn, self.opts.bracket, BracketOptions::default())?;

        //{{{ trace
        trace!(target: "scalar", "brent: bracket xa = {xa:.4e}, xb = {xb:.4e}, xc = {xc:.4e}");
        //}}}

        let mut x = xb;
        let mut w = xb;
        let mut v = xb;
        let mut fx = fb;
        let mut fw = fb;
        let mut fv = fb;

        let (mut a, mut b) = if xa < xc { (xa, xc) } else { (xc, xa) };

        let mut deltax = 0.0_f64;
        let mut rat = 0.0_f64;
        let mut iter = 0usize;

        while iter < self.opts.max_iter {
            let tol1 = self.opts.xtol * x.abs() + MINTOL;
            let tol2 = 2.0 * tol1;
            let xmid = 0.5 * (a + b);

            if (x - xmid).abs() < (tol2 - 0.5 * (b - a)) {
                break;
            }

            if deltax.abs() <= tol1 {
                // Do a golden-section step.
                deltax = if x >= xmid { a - x } else { b - x };
                rat = CG * deltax;
            } else {
                // Do a parabolic step.
                let tmp1 = (x - w) * (fx - fv);
                let tmp2_0 = (x - v) * (fx - fw);
                let mut p = (x - v) * tmp2_0 - (x - w) * tmp1;
                let mut tmp2 = 2.0 * (tmp2_0 - tmp1);
                if tmp2 > 0.0 {
                    p = -p;
                }
                tmp2 = tmp2.abs();
                let dx_temp = deltax;
                deltax = rat;

                // Check parabolic fit.
                if p > tmp2 * (a - x)
                    && p < tmp2 * (b - x)
                    && p.abs() < (0.5 * tmp2 * dx_temp).abs()
                {
                    rat = p / tmp2;
                    let u = x + rat;
                    if (u - a) < tol2 || (b - u) < tol2 {
                        rat = if xmid - x >= 0.0 { tol1 } else { -tol1 };
                    }
                } else {
                    deltax = if x >= xmid { a - x } else { b - x };
                    rat = CG * deltax;
                }
            }

            let u = if rat.abs() < tol1 {
                if rat >= 0.0 {
                    x + tol1
                } else {
                    x - tol1
                }
            } else {
                x + rat
            };
            let fu = self.fcn.eval(&u);
            num_fun_evals += 1;

            if fu > fx {
                if u < x {
                    a = u;
                } else {
                    b = u;
                }
                if fu <= fw || w == x {
                    v = w;
                    w = u;
                    fv = fw;
                    fw = fu;
                } else if fu <= fv || v == x || v == w {
                    v = u;
                    fv = fu;
                }
            } else {
                if u >= x {
                    a = x;
                } else {
                    b = x;
                }
                v = w;
                w = x;
                x = u;
                fv = fw;
                fw = fx;
                fx = fu;
            }

            //{{{ trace
            trace!(target: "scalar", "brent: iter {iter}, x = {x:.4e}, fx = {fx:.4e}");
            //}}}

            iter += 1;
        }

        if iter >= self.opts.max_iter {
            //{{{ trace
            trace!(target: "scalar", "brent: exceeded max_iter = {}", self.opts.max_iter);
            //}}}
            return Err(ScalarError::MaxIterations(self.opts.max_iter));
        }

        if x.is_nan() || fx.is_nan() {
            //{{{ trace
            trace!(target: "scalar", "brent: NaN encountered");
            //}}}
            return Err(ScalarError::NaN);
        }

        //{{{ trace
        trace!(target: "scalar", "brent: converged, xmin = {x:.4e}, fmin = {fx:.4e}, evals = {num_fun_evals}");
        //}}}

        Ok(ScalarReturns {
            xmin: x,
            fmin: fx,
            reason: ConvergedReason::Rtol,
            num_iterations: iter,
            num_fun_evals,
            num_grad_evals: 0,
        })
    }
}
//}}}
