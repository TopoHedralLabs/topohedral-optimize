//! Implementation of the Nocedal line search method
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common as com;
use super::common::{Error, LineSearch, Returns};
use super::utils::{cubicmin3, quadmin, satisfies_armijo};
use crate::RealFn1;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct: Options
#[derive(Copy, Clone, Default, Debug)]
pub struct Options
{
    pub ls_opts: com::Options,
    pub maxiter: usize,
    pub zoom_maxiter: usize,
}
//}}}
//{{{ struct: Nocedal
pub struct Nocedal<F: RealFn1>
{
    pub opts: Options,
    pub(crate) f: F,
}
//}}}
//{{{ impl: Nocedal
impl<F: RealFn1> Nocedal<F>
{
    #[trace_fn]
    pub fn new(
        f: F,
        opts: Options,
    ) -> Self
    {
        Self { f, opts }
    }
}
//}}}
//{{{ impl: LineSearch for Nocedal
impl<F: RealFn1> LineSearch for Nocedal<F>
{
    type Function = F;
    #[trace_fn]
    fn search(
        &mut self,
        phi0: f64,
        dphi0: f64,
        mut alpha1: f64,
    ) -> Result<Returns, Error>
    {
        //{{{ trace
        trace!(target: "ls", "Entering with phi0 = {:1.4e} dphi0 = {:1.4e}", phi0, dphi0);
        //}}}

        if dphi0 > 0.0
        {
            return Err(Error::NotDecreasing);
        }
        let c1 = self.opts.ls_opts.c1;
        let c2 = self.opts.ls_opts.c2;
        let mut alpha0 = 0.0;
        let mut phi_a0 = phi0;
        let mut dphi_a0 = dphi0;
        let mut dphi_a1 = 0.0;
        let _ = dphi_a1;
        let mut phi_a1 = self.f.eval(alpha1);
        let max_iter = self.opts.maxiter;

        for i in 0..max_iter
        {
            //{{{ trace
            trace!(target: "ls", "--------------------------------------- nocedal it = {}", i);
            trace!(target: "ls", "alpha0 = {:1.4e} alpha1 = {:1.4e}", alpha0, alpha1);
            trace!(target: "ls", "phi_a0 = {:1.4e} phi1 {:1.4e}", phi_a0, phi_a1);
            trace!(target: "ls", "dphi_aa0 = {:1.4e} dphi1 {:1.4e}", dphi_a0, dphi_a1);
            //}}}

            if alpha1 < f64::EPSILON * 100.0
            {
                //{{{ trace
                error!(target: "ls", "Too small step size detected");
                //}}}
                return Err(Error::StepSizeSmall);
            }

            let not_first_iteration = i > 0;
            let not_decreasing = phi_a1 >= phi_a0;

            // First check if current step is armijo-acceptable, if not then try zoom and check
            // again.
            if !satisfies_armijo(c1, alpha1, phi_a0, dphi_a0, phi_a1)
                || (not_decreasing && not_first_iteration)
            {
                //{{{ trace
                trace!(target: "ls", "Does not satisfy armijo");
                //}}}
                let zoom_result = zoom(
                    alpha0,
                    alpha1,
                    phi_a0,
                    phi_a1,
                    dphi_a0,
                    &mut self.f,
                    phi0,
                    dphi0,
                    c1,
                    c2,
                    self.opts.zoom_maxiter,
                );

                let (alpha_tmp, phi_tmp, _dphi_tmp) = match zoom_result
                {
                    None =>
                    {
                        //{{{ trace
                        error!(target: "ls","Zoom failed");
                        //}}}
                        return Err(Error::NotDecreasing);
                    }
                    Some(result) => result,
                };

                //{{{ trace
                error!(target: "ls", "Leaving with {:1.4e} {:1.4e} {:1.4e}", alpha_tmp, phi_tmp, _dphi_tmp);
                //}}}
                return Ok(Returns {
                    alpha: alpha_tmp,
                    phi_alpha: phi_tmp,
                });
            }

            // current step is armijo-acceptable, so check if curvature-accepttable
            dphi_a1 = self.f.diff(alpha1);
            //{{{ trace
            trace!(target: "ls", "dphi_a1 = {dphi_a1:1.4e}");
            //}}}
            if dphi_a1.abs() <= -c2 * dphi0
            {
                //{{{ trace
                trace!(target: "ls", "Satisfies curvature");
                trace!(target: "ls","Returning alpha = {:1.4e} falpha = {:1.4e}", alpha1, phi_a1);
                //}}}
                return Ok(Returns {
                    alpha: alpha1,
                    phi_alpha: phi_a1,
                });
            }

            if dphi_a1 >= 0.0
            {
                //{{{ trace
                trace!(target: "ls", "Curvature is positive {:1.4e}", dphi_a1);
                //}}}
                let zoom_result = zoom(
                    alpha1,
                    alpha0,
                    phi_a1,
                    phi_a0,
                    dphi_a1,
                    &mut self.f,
                    phi0,
                    dphi0,
                    c1,
                    c2,
                    self.opts.zoom_maxiter,
                );
                let (alpha_tmp, phi_tmp, _dphi_tmp) = match zoom_result
                {
                    None =>
                    {
                        //{{{  trace
                        error!(target: "ls", "Zoom failed");
                        //}}}
                        return Err(Error::NotDecreasing);
                    }
                    Some(result) =>
                    {
                        //{{{ trace
                        trace!(target: "ls","Zoom succeeded with result {:1.4e} {:1.4e} {:1.4e}",
                                  result.0, result.1, result.2);
                        //}}}
                        result
                    }
                };
                //{{{ trace
                error!(target: "ls", "Returning alpha = {:1.4e} falpha = {:1.4e}", alpha_tmp, phi_tmp);
                //}}}
                return Ok(Returns {
                    alpha: alpha_tmp,
                    phi_alpha: phi_tmp,
                });
            }

            //{{{ trace
            trace!(target: "ls", "Doubling alpha for next iteration to {:1.4e}", 2.0 * alpha1);
            //}}}
            let alpha2 = 2.0 * alpha1;
            alpha0 = alpha1;
            alpha1 = alpha2;
            phi_a0 = phi_a1;
            dphi_a0 = dphi_a1;
            phi_a1 = self.f.eval(alpha1);
        }

        //{{{ trace
        error!("Reached max number of iterations");
        //}}}
        Ok(Returns {
            alpha: alpha1,
            phi_alpha: phi_a1,
        })
    }
}
//}}}
//{{{ fun: zoom
#[allow(clippy::too_many_arguments, clippy::identity_op)]
#[trace_fn]
fn zoom<F: RealFn1>(
    mut a_lo: f64,
    mut a_hi: f64,
    mut phi_lo: f64,
    mut phi_hi: f64,
    mut dphi_lo: f64,
    phi_fcn: &mut F,
    phi0: f64,
    dphi0: f64,
    c1: f64,
    c2: f64,
    max_iter: usize,
) -> Option<(f64, f64, f64)>
where
{
    //{{{ trace
    trace!(target: "ls", "Entering with a_lo = {:1.4e} a_hi = {:1.4e} phi_lo = {:1.4e} phi_hi = {:1.4e}", a_lo, a_hi, phi_lo, phi_hi);
    trace!(target: "ls", "phi0 = {:1.4e} dphi0 = {:1.4e} c1 = {:1.4e} c2 = {:1.4e}", phi0, dphi0, c1, c2);
    //}}}
    let mut iter = 0;
    let delta1 = 0.2;
    let delta2 = 0.1;
    let mut phi_rec = phi0;
    let mut a_rec = 0.0;
    let mut cchk = 0.0;
    let mut qchk = 0.0;
    let _ = qchk;
    loop
    {
        //{{{ trace
        debug!(target: "ls", "...............zoom iter = {}", iter);
        trace!(target: "ls", "cchk = {:1.4e}  qchk = {:1.4e}", cchk, qchk);
        //}}}

        let dalpha = a_hi - a_lo;

        let (a, b) = if dalpha < 0.0
        {
            (a_hi, a_lo)
        }
        else
        {
            (a_lo, a_hi)
        };

        //{{{ trace
        trace!(target: "ls", "dalpha = {:1.4e} a_lo = {:1.4e} a_hi = {:1.4e}", dalpha, a_lo, a_hi);
        //}}}

        let mut opt_a_j: Option<f64> = None;

        // first try cubic interpolation
        if iter > 0
        {
            //{{{ trace
            trace!(target: "ls", "trying cubic interpolation");
            //}}}
            opt_a_j = cubicmin3(a_lo, phi_lo, dphi_lo, a_hi, phi_hi, a_rec, phi_rec);
            cchk = delta1 * dalpha;
        }

        // if not good enough first try quadratic interpolation
        if iter == 0
            || opt_a_j.is_none()
            || opt_a_j.unwrap() > b - cchk
            || opt_a_j.unwrap() < a + cchk
        {
            //{{{ trace
            trace!(target: "ls", "trying quadratic interpolation");
            //}}}
            qchk = delta2 * dalpha;
            opt_a_j = quadmin(a_lo, phi_lo, dphi_lo, a_hi, phi_hi);

            // finally try bisection
            if opt_a_j.is_none() || opt_a_j.unwrap() > b - qchk || opt_a_j.unwrap() < a + qchk
            {
                //{{{ trace
                trace!(target: "ls", "Trying bisection");
                //}}}
                opt_a_j = Some(0.5 * (a + b));
            }
        }

        let a_j = opt_a_j.unwrap();
        //{{{ trace
        trace!(target: "ls", "New value of a_j = {:1.4e}", a_j);
        //}}}
        // try new value of alpha
        let phi_aj = phi_fcn.eval(a_j);

        let not_sat_armijo = !satisfies_armijo(c1, a_j, phi0, dphi0, phi_aj);
        let not_decreasing = phi_aj >= phi_lo;

        if not_sat_armijo || not_decreasing
        {
            //{{{ trace
            trace!(target: "ls", "Failed armijo condition with {} {}", not_sat_armijo, not_decreasing);
            //}}}
            phi_rec = phi_hi;
            a_rec = a_hi;
            a_hi = a_j;
            phi_hi = phi_aj;
        }
        else
        {
            //{{{ trace
            trace!(target: "ls", "Passed armijo condition");
            //}}}
            let dphi_aj = phi_fcn.diff(a_j);
            if dphi_aj.abs() <= -c2 * dphi0
            {
                //{{{ trace
                trace!(target: "ls","Passed curvature condition");
                error!(target: "ls", "Returning a_j = {:1.4e} phi_aj = {:1.4e} dphi_aj = {:1.4e}",
                      a_j, phi_aj, dphi_aj);
                //}}}
                return Some((a_j, phi_aj, dphi_aj));
            }

            if dphi_aj * (a_hi - a_lo) >= 0.0
            {
                phi_rec = phi_hi;
                a_rec = a_hi;
                a_hi = a_lo;
                phi_hi = phi_lo;
            }
            else
            {
                phi_rec = phi_lo;
                a_rec = a_lo;
            }

            a_lo = a_j;
            phi_lo = phi_aj;
            dphi_lo = dphi_aj;
        }
        iter += 1;
        if iter == max_iter
        {
            //{{{ trace
            error!(target: "ls", "Reached max iterations in zoom");
            //}}}
            return None;
        }
    }
}
//}}}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests
{
    use super::*;
    use approx::assert_relative_eq;

    struct ScalarFunction<F: Fn(f64) -> f64, G: Fn(f64) -> f64>
    {
        f: F,
        df_dx: G,
    }

    impl<F: Fn(f64) -> f64, G: Fn(f64) -> f64> ScalarFunction<F, G>
    {
        #[trace_fn]
        pub fn new(
            f: F,
            df_dx: G,
        ) -> Self
        {
            Self { f, df_dx }
        }
    }

    impl<F: Fn(f64) -> f64, G: Fn(f64) -> f64> RealFn1 for ScalarFunction<F, G>
    {
        #[trace_fn]
        fn eval(
            &mut self,
            x: f64,
        ) -> f64
        {
            (self.f)(x)
        }

        #[trace_fn]
        fn diff(
            &mut self,
            x: f64,
        ) -> f64
        {
            (self.df_dx)(x)
        }
    }

    //{{{ collection: zoom tests
    #[test]
    #[trace_fn]
    fn test_zoom_quad_left()
    {
        let f = |x: f64| (x - 2.0).powi(2);
        let df_dx = |x: f64| 2.0 * (x - 2.0);
        let mut sf = ScalarFunction::new(f, df_dx);
        let phi0 = sf.eval(0.0);
        let dphi0 = sf.diff(0.0);
        let phi1 = sf.eval(1.0);
        let res = zoom(
            0.0, 1.0, phi0, phi1, dphi0, &mut sf, phi0, dphi0, 1e-4, 0.9, 10,
        );

        assert!(res.is_some());
        let res = res.unwrap();
        assert_relative_eq!(res.0, 0.5, epsilon = 1e-6);
        assert_relative_eq!(res.1, 2.25, epsilon = 1e-6);
        assert_relative_eq!(res.2, -3.0, epsilon = 1e-6);
    }

    #[test]
    #[trace_fn]
    fn test_zoom_quad_center()
    {
        let f = |x: f64| (x - 2.0).powi(2);
        let df_dx = |x: f64| 2.0 * (x - 2.0);
        let mut sf = ScalarFunction::new(f, df_dx);

        let a0 = -10.0;
        let phi0 = sf.eval(a0);
        let dphi0 = sf.diff(a0);
        let a_lo = -5.0;
        let a_hi = 15.0;
        let phi_lo = sf.eval(a_lo);
        let dphi_lo = sf.diff(a_lo);
        let phi_hi = sf.eval(a_hi);
        let res = zoom(
            a_lo, a_hi, phi_lo, phi_hi, dphi_lo, &mut sf, phi0, dphi0, 1e-4, 0.9, 10,
        );

        assert!(res.is_some());
        let res = res.unwrap();
        assert_relative_eq!(res.0, 2.0, epsilon = 1e-6);
        assert_relative_eq!(res.1, 0.0, epsilon = 1e-6);
        assert_relative_eq!(res.2, 0.0, epsilon = 1e-6);
    }

    #[test]
    #[trace_fn]
    fn test_zoom_quad_right_none()
    {
        let f = |x: f64| (x - 2.0).powi(2);
        let df_dx = |x: f64| 2.0 * (x - 2.0);
        let mut sf = ScalarFunction::new(f, df_dx);

        let a0 = 3.0;
        let phi0 = sf.eval(a0);
        let dphi0 = sf.diff(a0);
        let a_lo = 5.0;
        let a_hi = 100.0;
        let phi_lo = sf.eval(a_lo);
        let dphi_lo = sf.diff(a_lo);
        let phi_hi = sf.eval(a_hi);
        let res = zoom(
            a_lo, a_hi, phi_lo, phi_hi, dphi_lo, &mut sf, phi0, dphi0, 1e-4, 0.9, 10,
        );

        assert!(res.is_none());
    }

    #[test]
    #[trace_fn]
    fn test_zoom_quad_right_some()
    {
        let f = |x: f64| (x - 2.0).powi(2);
        let df_dx = |x: f64| 2.0 * (x - 2.0);
        let mut sf = ScalarFunction::new(f, df_dx);

        let a0 = 1.0;
        let phi0 = sf.eval(a0);
        let dphi0 = sf.diff(a0);
        let a_lo = 1.5;
        let a_hi = 100.0;
        let phi_lo = sf.eval(a_lo);
        let dphi_lo = sf.diff(a_lo);
        let phi_hi = sf.eval(a_hi);
        let res = zoom(
            a_lo, a_hi, phi_lo, phi_hi, dphi_lo, &mut sf, phi0, dphi0, 1e-4, 0.9, 10,
        );

        assert!(res.is_some());
        let res = res.unwrap();
        assert_relative_eq!(res.0, 2.0, epsilon = 1e-6);
        assert_relative_eq!(res.1, 0.0, epsilon = 1e-6);
        assert_relative_eq!(res.2, 0.0, epsilon = 1e-6);
    }
    //}}}
}
//}}}
