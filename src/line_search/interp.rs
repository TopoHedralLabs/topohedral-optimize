//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common as com;
use super::common::{
    Error, Error as LineSearchError, LineSearchFcn, LineSearcher, Options as LineSearchOptions,
    Returns,
};
use super::utils::{cubicmin, quadmin};
use crate::line_search::utils::{quadcubmin, satisfies_armijo, satisfies_wolfe};
use crate::RealFn1;
//}}}
//{{{ std imports
use std::ops::{Add, Mul};
use topohedral_linalg::VectorOps;
//}}}
//{{{ dep imports
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
struct GuessData {
    a: f64,
    phi_a: f64,
    dphi_a: f64,
    b: f64,
    phi_b: f64,
    c: f64,
    phi_c: f64,
}

#[derive(Debug, Copy, Clone)]
pub struct Options {
    pub ls_opts: com::Options,
    pub step1: f64,
    pub step2: f64,
    pub scale_factor: f64,
    pub maxiter: usize,
}

pub struct Interp<F: RealFn1> {
    pub opts: Options,
    pub(crate) f: F,
}

impl<F: RealFn1> Interp<F> {
    pub fn new(f: F, opts: Options) -> Self {
        Self { opts: opts, f: f }
    }

    fn guess_is_ok(&mut self, guess_data: GuessData) -> Option<(f64, f64, f64)> {
        //{{{ trace
        info!("--- Entering guess_is_ok ---");
        trace!("Data: {:?}", guess_data);
        //}}}
        let GuessData {
            a,
            phi_a,
            dphi_a,
            b,
            phi_b,
            c,
            phi_c,
        } = guess_data;

        let c1 = self.opts.ls_opts.c1;
        let c2 = self.opts.ls_opts.c2;
        let best_guess_opt = quadcubmin(&mut self.f, a, phi_a, dphi_a, b, phi_b, c, phi_c);

        if best_guess_opt.is_none() {
            return None;
        }

        let (alpha, phi_alpha) = best_guess_opt.unwrap();
        let dphi_alpha = self.f.diff(alpha);
        if satisfies_wolfe(c1, c2, phi_a, dphi_a, alpha, phi_alpha, dphi_alpha).is_ok() {
            return Some((alpha, phi_alpha, dphi_alpha));
        }
        None
    }
}

impl<F: RealFn1> LineSearcher for Interp<F> {
    type Function = F;

    fn search(&mut self, phi0: f64, dphi0: f64) -> Result<Returns, Error> {
        //{{{ trace
        error!(target: "ls", "--- Entering search ---");
        info!(target: "ls", "phi0={phi0} dphi0={dphi0}");
        //}}}
        let Options {
            ls_opts: _,
            step1,
            step2,
            scale_factor,
            maxiter,
        } = self.opts;

        let a = 0.0;
        let mut b_low = step1;
        let mut c_low = step2;
        let mut b_high = step1;
        let mut c_high = step2;
        let phi_a = phi0;
        let dphi_a = dphi0;
        let mut phi_b_low = self.f.eval(b_high);
        let mut phi_c_low = self.f.eval(c_high);
        let mut phi_b_high = phi_b_low;
        let mut phi_c_high = phi_b_high;

        if let Some((alpha, phi_alpha, dphi_alpha)) = self.guess_is_ok(GuessData {
            a: a,
            phi_a: phi_a,
            dphi_a: dphi_a,
            b: b_low,
            phi_b: phi_b_low,
            c: c_low,
            phi_c: phi_c_low,
        }) {
            return Ok(Returns {
                alpha: alpha,
                phi_alpha: phi_alpha,
                dphi_alpha: dphi_alpha,
            });
        }

        let inv_scale_factor = 1.0 / scale_factor;
        for i in 0..maxiter {
            //{{{ trace
            info!("---------------------------------- i = {i}");
            //}}}
            b_low = b_low * inv_scale_factor;
            phi_b_low = self.f.eval(b_low);
            c_low = c_low * inv_scale_factor;
            phi_c_low = self.f.eval(c_low);
            let guess_data_low = GuessData {
                a: a,
                phi_a: phi_a,
                dphi_a: dphi_a,
                b: b_low,
                phi_b: phi_b_low,
                c: c_low,
                phi_c: phi_c_low,
            };
            //{{{ trace
            debug!("Looking low:\n{guess_data_low:?}");
            //}}}
            if let Some((alpha, phi_alpha, dphi_alpha)) = self.guess_is_ok(guess_data_low) {
                //{{{ trace
                info!("Low guess found acceptable step: alpha = {alpha}, phi_alpha = {phi_alpha}, dphi_alpha = {dphi_alpha}");
                info!("--- leaving search() ----");
                //}}}
                return Ok(Returns {
                    alpha: alpha,
                    phi_alpha: phi_alpha,
                    dphi_alpha: dphi_alpha,
                });
            }

            b_high = b_high * scale_factor;
            phi_b_high = self.f.eval(b_high);
            c_high = c_high * scale_factor;
            phi_c_high = self.f.eval(c_high);
            let guess_data_high = GuessData {
                a: a,
                phi_a: phi_a,
                dphi_a: dphi_a,
                b: b_high,
                phi_b: phi_b_high,
                c: c_high,
                phi_c: phi_c_high,
            };
            //{{{ trace
            debug!("Looking high:\n{guess_data_high:?}");
            //}}}
            if let Some((alpha, phi_alpha, dphi_alpha)) = self.guess_is_ok(guess_data_high) {
                //{{{ trace
                info!("High guess found acceptable step: alpha = {alpha}, phi_alpha = {phi_alpha}, dphi_alpha = {dphi_alpha}");
                info!("--- leaving search() ----");
                //}}}
                return Ok(Returns {
                    alpha: alpha,
                    phi_alpha: phi_alpha,
                    dphi_alpha: dphi_alpha,
                });
            }
        }
        Err(LineSearchError::MaxIterations)
    }

    fn update_fcn(&mut self, fcn: Self::Function) {
        self.f = fcn;
    }
}
