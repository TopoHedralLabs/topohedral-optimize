//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common as com;
use super::common::{
    Error, Error as LineSearchError, LineSearch, LineSearchFcn, Options as LineSearchOptions,
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
const XTRAPL: f64 = 1.1;
const XTRAPU: f64 = 4.0;

pub struct Options {
    pub ls_opts: com::Options,
    pub maxiter: usize,
}

pub struct Thuente<F: RealFn1> {
    pub opts: Options,
    pub(crate) f: F,

    step_min: f64, 
    step_max: f64,
    cur_alpha1: f64,
    phi_alpha1: f64, 
    dphi_alpha1: f64, 
    cur_alpha2: f64,
    phi_alpha2: f64, 
    dphi_alpha2: f64, 
    cur_width: f64,
    prev_width: f64,
    step: f64,
    bracketed: bool,
}

impl<F: RealFn1> Thuente<F> {
    pub fn new(f: F, opts: Options) -> Self {

        
        Self {
            opts: opts,
            f: f,
            step_min: 0.0, 
            step_max: 0.0,
            cur_alpha1: 0.0,
            phi_alpha1: 0.0, 
            dphi_alpha1: 0.0, 
            cur_alpha2: 0.0,
            phi_alpha2: 0.0, 
            dphi_alpha2: 0.0,
            cur_width: 0.0,
            prev_width: 0.0,
            step: 0.0,
            bracketed: false
        }
    }

    fn initialise(&mut self, phi0: f64, dphi0: f64) {
        self.step_min = 0.0;
        self.step_max = self.opts.ls_opts.step_init * (1.0 + XTRAPU);

        self.cur_alpha1 = 0.0;
        self.phi_alpha1 = phi0;
        self.dphi_alpha1 = dphi0;
        self.cur_alpha2 = 0.0;
        self.phi_alpha2 = dphi0;
        self.dphi_alpha2 = dphi0;

        self.cur_width = self.opts.ls_opts.step_max - self.opts.ls_opts.step_min;
        self.prev_width = 2.0 * self.cur_width;
    }

    fn stage1(&mut self, alpha_init: f64, phi0: f64, dphi0: f64) -> (f64, f64, f64) {

        let c1 = self.opts.ls_opts.c1;

        let mut iter = 0;

        let mut alpha = alpha_init;
        let mut phi = phi0;
        let mut dphi = dphi0;

        while iter < self.opts.maxiter {

            if phi <= (phi0 + c1 * dphi0 * alpha) && dphi >= 0.0 {
                return (alpha, phi, dphi);
            }

            if phi <= self.phi_alpha1 && phi > c1 * dphi0 {

                let mod_phi_alpha1 = self.phi_alpha1 - (phi0 + c1 * dphi0 * self.cur_alpha1);
                let dmod_phi_alpha1 = self.dphi_alpha1 - c1 * dphi0;
                let mod_phi_alpha2 = self.phi_alpha2 - (phi0 + c1 * dphi0 * self.cur_alpha2);
                let dmod_phi_alpha2 = self.dphi_alpha2 - c1 * dphi0;
                let mod_phi = phi - (phi0 + c1 * dphi0 * alpha);
                let dmod_phi = dphi - c1 * dphi0;

                let interval_update_data = IntervalUpdateData {
                    step_alpha1: self.cur_alpha1, 
                    phi_alpha1:  mod_phi_alpha1,
                    dphi_alpha1: dmod_phi_alpha1,
                    step_alpha2: self.cur_alpha2,
                    phi_alpha2: mod_phi_alpha2,
                    dphi_alpha2: dmod_phi_alpha2,
                    step: alpha, 
                    phi: mod_phi, 
                    dphi: dmod_phi, 
                    bracketed: self.bracketed, 
                    lower_bound: self.step_min, 
                    upper_bound: self.step_max,
                };

                let interval_update_ret = interval_update_step(interval_update_data);
                self.phi_alpha1 = interval_update_ret.phi_alpha1 + (phi0 + c1 * dphi0 * self.cur_alpha1);
                self.dphi_alpha1 = interval_update_ret.dphi_alpha1 + c1 * dphi0;
                self.phi_alpha2 = interval_update_ret.phi_alpha2 + (phi0 + c1 * dphi0 * self.cur_alpha2);
                self.dphi_alpha2 = interval_update_ret.dphi_alpha2 + c1 * dphi0;
            }
            else { 

                let interval_update_data = IntervalUpdateData {
                    step_alpha1: self.cur_alpha1, 
                    phi_alpha1: self.phi_alpha1, 
                    dphi_alpha1: self.dphi_alpha1, 
                    step_alpha2: self.cur_alpha2, 
                    phi_alpha2: self.phi_alpha2, 
                    dphi_alpha2: self.dphi_alpha2, 
                    step: alpha, 
                    phi: phi, 
                    dphi: dphi, 
                    bracketed: self.bracketed, 
                    lower_bound: self.step_min, 
                    upper_bound: self.step_max,
                };
                let interval_update_ret = interval_update_step(interval_update_data);
                self.cur_alpha1 = interval_update_ret.step_alpha1;
                self.phi_alpha1 = interval_update_ret.phi_alpha1;
                self.dphi_alpha1 = interval_update_ret.dphi_alpha1;
                self.cur_alpha2 = interval_update_ret.step_alpha2;
                self.phi_alpha2 = interval_update_ret.phi_alpha2;
                self.dphi_alpha2 = interval_update_ret.dphi_alpha2;
                self.step = interval_update_ret.step;
                self.bracketed = interval_update_ret.bracketed;
            }

            iter += 1;
        }
        (0.0, 0.0, 0.0)
    }

    fn stage2(&mut self, alpha: f64, phi: f64, dphi: f64) -> (f64, f64, f64) {


        (0.0, 0.0, 0.0)
    }
}

impl<F: RealFn1> LineSearch for Thuente<F> {
    type Function = F;

    fn search(&mut self, phi0: f64, dphi0: f64) -> Result<Returns, Error> {

        let alpha_init = self.opts.ls_opts.step_init;

        self.initialise(phi0, dphi0);
        let (mut alpha, mut phi, mut dphi) = self.stage1(alpha_init, phi0, dphi0);
        (alpha, phi, dphi) = self.stage2(alpha, phi, dphi);


        Err(Error::NoStepFound)
    }

    fn update_fcn(&mut self, fcn: Self::Function) {
        self.f = fcn;
    }
}

struct IntervalUpdateData {
    step_alpha1: f64, 
    phi_alpha1: f64, 
    dphi_alpha1: f64, 
    step_alpha2: f64, 
    phi_alpha2: f64, 
    dphi_alpha2: f64, 
    step: f64, 
    phi: f64, 
    dphi: f64,
    bracketed: bool, 
    lower_bound: f64, 
    upper_bound: f64, 
}

struct IntervalUpdateReturns {
    step_alpha1: f64, 
    phi_alpha1: f64, 
    dphi_alpha1: f64, 
    step_alpha2: f64, 
    phi_alpha2: f64, 
    dphi_alpha2: f64, 
    step: f64, 
    bracketed: bool, 
}

fn interval_update_step(data: IntervalUpdateData)  -> IntervalUpdateReturns{


    IntervalUpdateReturns {
        step_alpha1: data.step_alpha1,
        phi_alpha1: data.phi_alpha1,
        dphi_alpha1: data.dphi_alpha1,
        step_alpha2: data.step_alpha2,
        phi_alpha2: data.phi_alpha2,
        dphi_alpha2: data.dphi_alpha2,
        step: data.step,
        bracketed: data.bracketed
    }
}
