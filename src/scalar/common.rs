//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::RealFn1;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use thiserror::Error;
use topohedral_tracing::{trace, trace_fn};
//}}}
//--------------------------------------------------------------------------------------------------

#[derive(Error, Debug)]
pub enum Error {
    #[error("Maximum iterations of {0} reached")]
    MaxIterations(usize),
    #[error(
        "The algorithm terminated without finding a valid bracket, consider trying different \
         initial points"
    )]
    InvalidBracket,
    #[error("Optimization bounds must be finite scalars, got lower = {0}, upper = {1}")]
    NonFiniteBounds(f64, f64),
    #[error("The lower bound {0} exceeds the upper bound {1}")]
    InvalidBounds(f64, f64),
    #[error("Function evaluation returned NaN")]
    NaN,
    #[error(
        "Bracketing values (xa, xb, xc) = ({0}, {1}, {2}) do not fulfill this requirement: \
         (xa < xb) and (xb < xc)"
    )]
    InvalidBracketOrder(f64, f64, f64),
    #[error(
        "Bracketing values (xa, xb, xc) do not fulfill this requirement: (f(xb) < f(xa)) and \
         (f(xb) < f(xc))"
    )]
    InvalidBracketValues,
}

#[derive(Copy, Clone)]
pub struct BracketOptions {
    pub grow_limit: f64,
    pub max_iter: usize,
}

impl Default for BracketOptions {
    fn default() -> Self {
        Self {
            grow_limit: 110.0,
            max_iter: 1000,
        }
    }
}

//{{{ struct: BracketResult
/// Result of a successful [`bracket`] search.
///
/// `xa`, `xb`, `xc` are ordered (either increasing or decreasing) such that `xb`
/// lies between `xa` and `xc`, and the corresponding function values satisfy
/// `fb < fa` and `fb < fc`, i.e. `(xa, fa)`, `(xb, fb)`, `(xc, fc)` bracket a
/// local minimum of the function.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BracketResult {
    pub xa: f64,
    pub xb: f64,
    pub xc: f64,
    pub fa: f64,
    pub fb: f64,
    pub fc: f64,
    pub num_fun_evals: usize,
}
//}}}
//{{{ fun: bracket
/// Downhill bracket search for a local minimum.
///
/// Given a function `f` and two distinct initial points `a` and `b`, searches
/// in the downhill direction, as defined by the two initial points, and
/// returns three points that bracket a local minimum of `f`.
///
/// Ported from SciPy's `scipy.optimize.bracket`
/// (`scipy/optimize/_optimize.py`).
///
/// # Errors
///
/// - [`Error::MaxIterations`] if `opts.max_iter` growth steps are performed
///   without finding a bracket.
/// - [`Error::InvalidBracket`] if the algorithm terminates without satisfying
///   the validity conditions of a bracket, i.e. `fb < fa`, `fb < fc`, `xb`
///   strictly between `xa` and `xc`, and all three points finite.
#[trace_fn]
pub fn bracket<F: RealFn1>(
    f: &mut F,
    a: f64,
    b: f64,
    opts: BracketOptions,
) -> Result<BracketResult, Error> {
    const GOLD: f64 = 1.618034;
    const VERY_SMALL_NUM: f64 = 1e-21;

    let mut xa = a;
    let mut xb = b;
    let mut fa = f.eval(xa);
    let mut fb = f.eval(xb);

    if fa < fb {
        std::mem::swap(&mut xa, &mut xb);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut xc = xb + GOLD * (xb - xa);
    let mut fc = f.eval(xc);
    let mut num_fun_evals = 3usize;
    let mut iter = 0usize;

    //{{{ trace
    trace!(target: "scalar", "bracket: starting with xa = {xa:.4e}, xb = {xb:.4e}, xc = {xc:.4e}");
    //}}}

    while fc < fb {
        let tmp1 = (xb - xa) * (fb - fc);
        let tmp2 = (xb - xc) * (fb - fa);
        let val = tmp2 - tmp1;
        let denom = if val.abs() < VERY_SMALL_NUM {
            2.0 * VERY_SMALL_NUM
        } else {
            2.0 * val
        };
        let mut w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
        let wlim = xb + opts.grow_limit * (xc - xb);

        if iter > opts.max_iter {
            //{{{ trace
            trace!(target: "scalar", "bracket: exceeded max_iter = {}", opts.max_iter);
            //}}}
            return Err(Error::MaxIterations(opts.max_iter));
        }
        iter += 1;

        let fw;
        if (w - xc) * (xb - w) > 0.0 {
            let mut fw_inner = f.eval(w);
            num_fun_evals += 1;
            if fw_inner < fc {
                xa = xb;
                xb = w;
                fa = fb;
                fb = fw_inner;
                break;
            } else if fw_inner > fb {
                xc = w;
                fc = fw_inner;
                break;
            }
            w = xc + GOLD * (xc - xb);
            fw_inner = f.eval(w);
            num_fun_evals += 1;
            fw = fw_inner;
        } else if (w - wlim) * (wlim - xc) >= 0.0 {
            w = wlim;
            fw = f.eval(w);
            num_fun_evals += 1;
        } else if (w - wlim) * (xc - w) > 0.0 {
            let mut fw_inner = f.eval(w);
            num_fun_evals += 1;
            if fw_inner < fc {
                xb = xc;
                xc = w;
                w = xc + GOLD * (xc - xb);
                fb = fc;
                fc = fw_inner;
                fw_inner = f.eval(w);
                num_fun_evals += 1;
            }
            fw = fw_inner;
        } else {
            w = xc + GOLD * (xc - xb);
            fw = f.eval(w);
            num_fun_evals += 1;
        }

        xa = xb;
        xb = xc;
        xc = w;
        fa = fb;
        fb = fc;
        fc = fw;
    }

    let cond1 = (fb < fc && fb <= fa) || (fb < fa && fb <= fc);
    let cond2 = (xa < xb && xb < xc) || (xc < xb && xb < xa);
    let cond3 = xa.is_finite() && xb.is_finite() && xc.is_finite();

    if !(cond1 && cond2 && cond3) {
        //{{{ trace
        trace!(target: "scalar", "bracket: invalid bracket xa = {xa:.4e}, xb = {xb:.4e}, xc = {xc:.4e}");
        //}}}
        return Err(Error::InvalidBracket);
    }

    //{{{ trace
    trace!(target: "scalar", "bracket: found xa = {xa:.4e}, xb = {xb:.4e}, xc = {xc:.4e} in {num_fun_evals} evals");
    //}}}

    Ok(BracketResult {
        xa,
        xb,
        xc,
        fa,
        fb,
        fc,
        num_fun_evals,
    })
}
//}}}
//{{{ enum: Bracket
/// Specifies how the initial bracketing triple for [`resolve_bracket`] is obtained.
#[derive(Copy, Clone, Default)]
pub enum Bracket {
    /// Search for a bracket automatically, starting from the default points `(0, 1)`.
    #[default]
    Auto,
    /// Search for a bracket starting downhill from the two given points.
    Points(f64, f64),
    /// Use an explicit three-point bracket `(xa, xb, xc)`.
    ///
    /// Validated so that, after ordering `xa` and `xc`, `xb` lies strictly
    /// between them and `f(xb) < f(xa)` and `f(xb) < f(xc)`.
    Triple(f64, f64, f64),
}
//}}}
//{{{ fun: resolve_bracket
/// Resolves a [`Bracket`] specification into a validated [`BracketResult`].
///
/// For [`Bracket::Auto`] and [`Bracket::Points`], delegates to [`bracket`]. For
/// [`Bracket::Triple`], validates the triple directly rather than searching.
///
/// # Errors
///
/// - [`Error::InvalidBracketOrder`] if, after ordering `xa` and `xc`, `xb`
///   does not lie strictly between them.
/// - [`Error::InvalidBracketValues`] if `f(xb)` is not lower than both
///   `f(xa)` and `f(xc)`.
/// - Any error from [`bracket`] for the `Auto`/`Points` variants.
#[trace_fn]
pub fn resolve_bracket<F: RealFn1>(
    f: &mut F,
    spec: Bracket,
    opts: BracketOptions,
) -> Result<BracketResult, Error> {
    match spec {
        Bracket::Auto => bracket(f, 0.0, 1.0, opts),
        Bracket::Points(xa, xb) => bracket(f, xa, xb, opts),
        Bracket::Triple(xa0, xb, xc0) => {
            let (xa, xc) = if xa0 > xc0 { (xc0, xa0) } else { (xa0, xc0) };
            if !(xa < xb && xb < xc) {
                return Err(Error::InvalidBracketOrder(xa, xb, xc));
            }
            let fa = f.eval(xa);
            let fb = f.eval(xb);
            let fc = f.eval(xc);
            if !(fb < fa && fb < fc) {
                return Err(Error::InvalidBracketValues);
            }
            Ok(BracketResult {
                xa,
                xb,
                xc,
                fa,
                fb,
                fc,
                num_fun_evals: 3,
            })
        }
    }
}
//}}}
//{{{ mod: tests
#[cfg(test)]
mod tests {
    use super::*;

    struct ScalarFunction<F: Fn(f64) -> f64> {
        f: F,
    }

    impl<F: Fn(f64) -> f64> ScalarFunction<F> {
        fn new(f: F) -> Self {
            Self { f }
        }
    }

    impl<F: Fn(f64) -> f64> RealFn1 for ScalarFunction<F> {
        fn eval(
            &mut self,
            x: f64,
        ) -> f64 {
            (self.f)(x)
        }

        fn diff(
            &mut self,
            _x: f64,
        ) -> f64 {
            unimplemented!("not needed by bracket")
        }
    }

    fn parabola(x: f64) -> f64 {
        (x - 1.0).powi(2)
    }

    fn quartic(x: f64) -> f64 {
        (x - 2.0).powi(4) + 3.0
    }

    fn assert_valid_bracket(
        res: &BracketResult,
        f: &mut impl RealFn1,
    ) {
        assert!(
            (res.xa < res.xb && res.xb < res.xc) || (res.xc < res.xb && res.xb < res.xa),
            "xb must lie strictly between xa and xc: xa = {}, xb = {}, xc = {}",
            res.xa,
            res.xb,
            res.xc
        );
        assert!(
            res.fb < res.fa && res.fb < res.fc,
            "xb must be the lowest point"
        );

        // cross-check reported function values against the function itself
        assert!((f.eval(res.xa) - res.fa).abs() < 1e-12);
        assert!((f.eval(res.xb) - res.fb).abs() < 1e-12);
        assert!((f.eval(res.xc) - res.fc).abs() < 1e-12);
    }

    #[test]
    fn test_bracket_parabola_downhill() {
        let mut f = ScalarFunction::new(parabola);
        let res = bracket(&mut f, 0.0, 1.0, BracketOptions::default()).unwrap();
        assert_valid_bracket(&res, &mut f);
    }

    #[test]
    fn test_bracket_parabola_uphill_initial_points() {
        // fa < fb initially (both left of the minimum), forcing the initial swap
        let mut f = ScalarFunction::new(parabola);
        let res = bracket(&mut f, -1.0, -0.5, BracketOptions::default()).unwrap();
        assert_valid_bracket(&res, &mut f);
    }

    #[test]
    fn test_bracket_quartic() {
        let mut f = ScalarFunction::new(quartic);
        let res = bracket(&mut f, 0.0, 1.0, BracketOptions::default()).unwrap();
        assert_valid_bracket(&res, &mut f);
        assert!(
            (res.xa < 2.0 && 2.0 < res.xc) || (res.xc < 2.0 && 2.0 < res.xa),
            "bracket should straddle the true minimum at x = 2: xa = {}, xc = {}",
            res.xa,
            res.xc
        );
    }

    #[test]
    fn test_bracket_max_iterations_exceeded() {
        // A function that is monotonically decreasing everywhere never brackets a
        // minimum, so with a tiny iteration budget the search should report
        // MaxIterations rather than loop forever.
        let mut f = ScalarFunction::new(|x: f64| -x);
        let opts = BracketOptions {
            grow_limit: 110.0,
            max_iter: 2,
        };
        let res = bracket(&mut f, 0.0, 1.0, opts);
        assert!(matches!(res, Err(Error::MaxIterations(2))));
    }

    #[test]
    fn test_bracket_funcalls_counts_all_evaluations() {
        let mut f = ScalarFunction::new(parabola);
        let res = bracket(&mut f, 0.0, 1.0, BracketOptions::default()).unwrap();
        assert!(res.num_fun_evals >= 3);
    }
}
//}}}
