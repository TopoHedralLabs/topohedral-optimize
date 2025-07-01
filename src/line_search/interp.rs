//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::{Error, LineSearchFcn, LineSearcher, Options, Returns};
use super::utils::{cubicmin, quadmin};
use crate::RealFn1;
//}}}
//{{{ std imports
use std::ops::{Add, Mul};
use topohedral_linalg::VectorOps;
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------

#[derive(Debug, Copy, Clone)]
pub struct InterpOpts {
    ls_opts: Options,
    step1: f64,
    step2: f64,
}

pub struct InterpLineSearch<F: RealFn1> {
    pub opts: InterpOpts,
    pub(crate) f: F,
}

impl<F: RealFn1> InterpLineSearch<F> {}

impl<F: RealFn1> LineSearcher for InterpLineSearch<F> {
    type Error = Error;
    type Returns = Returns;

    fn search(&mut self, phi0: f64, dphi0: f64) -> Result<Self::Returns, Self::Error> {
        let step0 = 0.0;
        let InterpOpts {
            ls_opts,
            step1,
            step2,
        } = self.opts;
        let phi1 = self.f.eval(step1);
        let phi2 = self.f.eval(step2);

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
            .filter_map(|x| x)
            .min_by(|x, y| {x.1.partial_cmp(&y.1).unwrap()});

        if opt_min_value.is_none()
        {
            return Err(Error::NotDecreasing);
        }

        // let min_value = opt_min_value.unwrap();
        // Some(Returns{
        //     min_value.0, min_value.1, 
        // })
        todo!()


    }
}
