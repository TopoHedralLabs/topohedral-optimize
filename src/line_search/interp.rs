//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::{Error, LineSearchFcn, LineSearcher, Returns};
use super::common as com;
use super::utils::{cubicmin, quadmin};
use crate::line_search::utils::satisfies_wolfe;
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

#[derive(Debug, Copy, Clone)]
pub struct Options {
    pub ls_opts: com::Options,
    pub step1: f64,
    pub step2: f64,
    pub scale_factor: f64
}

pub struct Interp<F: RealFn1> {
    pub opts: Options,
    pub(crate) f: F,
}

impl<F: RealFn1> Interp<F> {
    pub fn new(f: F, opts: Options) -> Self { 
        Self{
            opts: opts, 
            f: f
        }
    }
}


impl<F: RealFn1> LineSearcher for Interp<F> {

    type Function = F;

    fn search(&mut self, phi0: f64, dphi0: f64) -> Result<Returns, Error> {

        error!(target: "ls", "--- Entering search ---");
        info!(target: "ls", "phi0={phi0} dphi0={dphi0}");

        let step0 = 0.0;
        let Options {
            ls_opts,
            step1,
            step2,
            scale_factor
        } = self.opts;

        let phi1 = self.f.eval(step1);
        let phi2 = self.f.eval(step2);

        info!(target: "ls", "step1 = {step1} phi1 = {phi1} step2 = {step2} phi2 = {phi2}");

        let to_pair = |x: &Option<f64>| -> Option<(f64, f64)> {
            match x {
                None => None,
                Some(alpha) => Some((*alpha, self.f.eval(*alpha))),
            }
        };

        let cubic_min_alpha = cubicmin(0.0, phi0, dphi0, step1, phi1, step2, phi2);
        let quad_min1_alpha = quadmin(0.0, phi0, dphi0, step1, phi1);
        let quad_min2_alpha = quadmin(0.0, phi0, dphi0, step2, phi2);
        let opt_min_value = [cubic_min_alpha, quad_min1_alpha, quad_min2_alpha]
            .iter()
            .map(to_pair)
            .filter_map(|x| {
                //{{{ trace
                trace!(target: "ls", "alpha-falpha pair: {:?}", x);
                //}}}
                x
            })
            .min_by(|x, y| {x.1.partial_cmp(&y.1).unwrap()});

        if opt_min_value.is_none()
        {
            info!(target: "ls", "No value found");
            return Err(Error::NotDecreasing);
        }

        let (alpha_min, fmin) = opt_min_value.unwrap();
        //{{{ trace
        trace!(target: "ls", "returning alpha ={alpha_min:1.4e} fmin = {fmin:1.4e}");
        //}}}
        Ok(Returns{
            alpha: alpha_min, falpha:fmin 
        })


    }
    
    fn update_fcn(&mut self, fcn: Self::Function) {
        self.f = fcn;
    }
}
