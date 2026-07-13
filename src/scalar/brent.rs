//! Unbounded scalar minimization via Brent's method.
//!
//! Ported from SciPy's `Brent` class / `_minimize_scalar_brent`
//! (`scipy/optimize/_optimize.py`): inverse parabolic interpolation combined
//! with golden-section search, isolating a local minimum starting from a
//! bracketing triple.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::{bracket, BracketOptions, BracketResult, Error as ScalarError};
use crate::{common::ScalarReturns, ConvergedReason, Minimizer, RealFn1};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::{trace, trace_fn};
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum: Bracket
/// Specifies how the initial bracketing triple is obtained.
#[derive(Copy, Clone, Default)]
pub enum Bracket
{
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
//{{{ struct: Options
#[derive(Copy, Clone)]
pub struct Options
{
    /// Relative tolerance on `x` used as the termination criterion.
    pub xtol: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
}
//}}}
//{{{ impl: Default for Options
impl Default for Options
{
    fn default() -> Self
    {
        Self {
            xtol: 1.48e-8,
            max_iter: 500,
        }
    }
}
//}}}
//{{{ struct: Brent
/// Unbounded minimization of a scalar function using Brent's method.
///
/// Ported from SciPy's `Brent` class / `_minimize_scalar_brent`
/// (`scipy/optimize/_optimize.py`).
pub struct Brent<F: RealFn1>
{
    fcn: F,
    bracket: Bracket,
    opts: Options,
}
//}}}
//{{{ impl: Brent
impl<F: RealFn1> Brent<F>
{
    #[trace_fn]
    pub fn new(
        fcn: F,
        bracket: Bracket,
        opts: Options,
    ) -> Self
    {
        Self { fcn, bracket, opts }
    }

    #[trace_fn]
    fn get_bracket_info(&mut self) -> Result<BracketResult, ScalarError>
    {
        match self.bracket
        {
            Bracket::Auto => bracket(&mut self.fcn, 0.0, 1.0, BracketOptions::default()),
            Bracket::Points(xa, xb) => bracket(&mut self.fcn, xa, xb, BracketOptions::default()),
            Bracket::Triple(xa0, xb, xc0) =>
            {
                let (xa, xc) = if xa0 > xc0 { (xc0, xa0) } else { (xa0, xc0) };
                if !(xa < xb && xb < xc)
                {
                    return Err(ScalarError::InvalidBracketOrder(xa, xb, xc));
                }
                let fa = self.fcn.eval(xa);
                let fb = self.fcn.eval(xb);
                let fc = self.fcn.eval(xc);
                if !(fb < fa && fb < fc)
                {
                    return Err(ScalarError::InvalidBracketValues);
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
}
//}}}
//{{{ impl: Minimizer for Brent
impl<F: RealFn1> Minimizer for Brent<F>
{
    type Error = ScalarError;
    type Returns = ScalarReturns;

    #[trace_fn]
    fn minimize(&mut self) -> Result<Self::Returns, Self::Error>
    {
        const MINTOL: f64 = 1.0e-11;
        const CG: f64 = 0.3819660;

        let BracketResult {
            xa,
            xb,
            xc,
            fb,
            mut num_fun_evals,
            ..
        } = self.get_bracket_info()?;

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

        while iter < self.opts.max_iter
        {
            let tol1 = self.opts.xtol * x.abs() + MINTOL;
            let tol2 = 2.0 * tol1;
            let xmid = 0.5 * (a + b);

            if (x - xmid).abs() < (tol2 - 0.5 * (b - a))
            {
                break;
            }

            if deltax.abs() <= tol1
            {
                // Do a golden-section step.
                deltax = if x >= xmid { a - x } else { b - x };
                rat = CG * deltax;
            }
            else
            {
                // Do a parabolic step.
                let tmp1 = (x - w) * (fx - fv);
                let tmp2_0 = (x - v) * (fx - fw);
                let mut p = (x - v) * tmp2_0 - (x - w) * tmp1;
                let mut tmp2 = 2.0 * (tmp2_0 - tmp1);
                if tmp2 > 0.0
                {
                    p = -p;
                }
                tmp2 = tmp2.abs();
                let dx_temp = deltax;
                deltax = rat;

                // Check parabolic fit.
                if p > tmp2 * (a - x) && p < tmp2 * (b - x) && p.abs() < (0.5 * tmp2 * dx_temp).abs()
                {
                    rat = p / tmp2;
                    let u = x + rat;
                    if (u - a) < tol2 || (b - u) < tol2
                    {
                        rat = if xmid - x >= 0.0 { tol1 } else { -tol1 };
                    }
                }
                else
                {
                    deltax = if x >= xmid { a - x } else { b - x };
                    rat = CG * deltax;
                }
            }

            let u = if rat.abs() < tol1
            {
                if rat >= 0.0 { x + tol1 } else { x - tol1 }
            }
            else
            {
                x + rat
            };
            let fu = self.fcn.eval(u);
            num_fun_evals += 1;

            if fu > fx
            {
                if u < x
                {
                    a = u;
                }
                else
                {
                    b = u;
                }
                if fu <= fw || w == x
                {
                    v = w;
                    w = u;
                    fv = fw;
                    fw = fu;
                }
                else if fu <= fv || v == x || v == w
                {
                    v = u;
                    fv = fu;
                }
            }
            else
            {
                if u >= x
                {
                    a = x;
                }
                else
                {
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

        if iter >= self.opts.max_iter
        {
            //{{{ trace
            trace!(target: "scalar", "brent: exceeded max_iter = {}", self.opts.max_iter);
            //}}}
            return Err(ScalarError::MaxIterations(self.opts.max_iter));
        }

        if x.is_nan() || fx.is_nan()
        {
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
//{{{ mod: tests
#[cfg(test)]
mod tests
{
    use super::*;

    struct ScalarFunction<G: Fn(f64) -> f64>
    {
        f: G,
    }

    impl<G: Fn(f64) -> f64> ScalarFunction<G>
    {
        fn new(f: G) -> Self
        {
            Self { f }
        }
    }

    impl<G: Fn(f64) -> f64> RealFn1 for ScalarFunction<G>
    {
        fn eval(
            &mut self,
            x: f64,
        ) -> f64
        {
            (self.f)(x)
        }

        fn diff(
            &mut self,
            _x: f64,
        ) -> f64
        {
            unimplemented!("not needed by brent")
        }
    }

    fn parabola(x: f64) -> f64
    {
        (x - 1.0).powi(2)
    }

    fn quartic(x: f64) -> f64
    {
        (x - 2.0).powi(4) + 3.0
    }

    #[test]
    fn test_brent_parabola_two_point_bracket()
    {
        let f = ScalarFunction::new(parabola);
        let mut minimizer = Brent::new(f, Bracket::Points(0.0, 1.0), Options::default());
        let res = minimizer.minimize().unwrap();
        assert!((res.xmin - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_brent_parabola_three_point_bracket()
    {
        let f = ScalarFunction::new(parabola);
        let mut minimizer = Brent::new(f, Bracket::Triple(-1.0, 0.5, 3.0), Options::default());
        let res = minimizer.minimize().unwrap();
        assert!((res.xmin - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_brent_parabola_auto_bracket()
    {
        let f = ScalarFunction::new(parabola);
        let mut minimizer = Brent::new(f, Bracket::Auto, Options::default());
        let res = minimizer.minimize().unwrap();
        assert!((res.xmin - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_brent_quartic()
    {
        let f = ScalarFunction::new(quartic);
        let mut minimizer = Brent::new(f, Bracket::Points(0.0, 1.0), Options::default());
        let res = minimizer.minimize().unwrap();
        assert!((res.xmin - 2.0).abs() < 1e-2);
        assert!((res.fmin - 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_brent_invalid_triple_order()
    {
        let f = ScalarFunction::new(parabola);
        let mut minimizer = Brent::new(f, Bracket::Triple(0.0, 0.9, 0.5), Options::default());
        let result = minimizer.minimize();
        assert!(matches!(result, Err(ScalarError::InvalidBracketOrder(_, _, _))));
    }

    #[test]
    fn test_brent_invalid_triple_values()
    {
        // Monotone function: f(xb) is not below both f(xa) and f(xc).
        let f = ScalarFunction::new(|x: f64| x);
        let mut minimizer = Brent::new(f, Bracket::Triple(0.0, 0.5, 1.0), Options::default());
        let result = minimizer.minimize();
        assert!(matches!(result, Err(ScalarError::InvalidBracketValues)));
    }

    #[test]
    fn test_brent_max_iterations_exceeded()
    {
        let f = ScalarFunction::new(parabola);
        let opts = Options {
            xtol: 1e-14,
            max_iter: 2,
        };
        let mut minimizer = Brent::new(f, Bracket::Points(0.0, 1.0), opts);
        let result = minimizer.minimize();
        assert!(matches!(result, Err(ScalarError::MaxIterations(2))));
    }
}
//}}}
