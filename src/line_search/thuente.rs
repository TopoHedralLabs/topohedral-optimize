//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

use serde_json::Value;

//{{{ crate imports
use super::common as com;
use super::common::{Error, LineSearch, Returns};
use crate::RealFn1;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
//}}}
//--------------------------------------------------------------------------------------------------
const XTRAPL: f64 = 1.1;
const XTRAPU: f64 = 4.0;

pub struct Options {
    pub ls_opts: com::Options,
    pub maxiter: usize,
}

struct Values {
    alpha: f64,
    phi: f64,
    dphi: f64,
}

struct Bracket {
    low: f64,
    high: f64,
}

impl Bracket {
    fn width(&self) -> f64 {
        return self.high - self.low;
    }
}

pub struct Thuente<F: RealFn1> {
    pub opts: Options,
    pub(crate) f: F,
    vals1: Values,
    vals2: Values,
    bracket: Option<Bracket>,
}

impl<F: RealFn1> Thuente<F> {
    pub fn new(f: F, opts: Options) -> Self {
        Self {
            opts,
            f,
            vals1: Values {
                alpha: 0.0,
                phi: 0.0,
                dphi: 0.0,
            },
            vals2: Values {
                alpha: 0.0,
                phi: 0.0,
                dphi: 0.0,
            },
            bracket: None,
        }
    }

    fn initialise(&mut self, phi0: f64, dphi0: f64) {}
}

impl<F: RealFn1> LineSearch for Thuente<F> {
    type Function = F;

    fn search(&mut self, phi0: f64, dphi0: f64) -> Result<Returns, Error> {
        let alpha_init = self.opts.ls_opts.step_init;

        self.initialise(phi0, dphi0);

        Err(Error::NoStepFound)
    }

    fn update_fcn(&mut self, fcn: Self::Function) {
        self.f = fcn;
    }
}

struct StepArgs {
    low_values: Values,
    high_values: Values,
    cur_values: Values,
    bracketed: bool,
    step_min: f64,
    step_max: f64,
}

struct StepReturn {
    low_values: Values,
    high_values: Values,
    alpha: f64,
    bracket: bool,
}

fn findStepCase(args: &StepArgs) -> u8 {
    let cur_phi_is_greater = args.cur_values.phi > args.low_values.phi;
    // case 1: A higher function value. The minimum is bracketed.
    if cur_phi_is_greater {
        return 1;
    }

    let sign_dphi_cur = args.cur_values.dphi.signum();
    let sign_dphi_low = args.low_values.dphi.signum();
    let dphi_has_opposite_sign = sign_dphi_cur * sign_dphi_low < 0.0;
    // case 2: Lower function value and derivatives of opposite signs. Min is bracketed
    if dphi_has_opposite_sign {
        return 2;
    }

    let decreasing_derivative_magnitude = args.cur_values.dphi.abs() < args.low_values.dphi.abs();
    // case 3: Lower function value, derivatives same sign, magnitude of derivative is decreasing
    // in direction of step, bracketing remains unchanged
    if decreasing_derivative_magnitude {
        return 3;
    }

    // case 4: Lower function value, derivatives same sign, magnitude of derivative is increasing
    // in direction of step.
    return 4;
}

//{{{ fn: stepCase1
/// First case: A higher function value. The minimum is bracketed.
/// If the cubic step is closer to stx than the quadratic step, the
/// cubic step is taken, otherwise the average of the cubic and
/// quadratic steps is taken.
fn stepCase1(args: &StepArgs) -> (f64, bool) {
    todo!()
}
//}}}
//{{{ fn: stepCase2
/// Second case: A lower function value and derivatives of opposite
/// sign. The minimum is bracketed. If the cubic step is farther from
/// stp than the secant step, the cubic step is taken, otherwise the
/// secant step is taken.
fn stepCase2(args: &StepArgs) -> (f64, bool){
    todo!()
}
//}}}
//{{{ fn: stepCase3
fn stepCase3(args: &StepArgs) -> (f64, bool){ 
    todo!()
}
//}}}
//{{{ fn: stepCase4
fn stepCase4(args: &StepArgs) -> (f64, bool){
    todo!()
}
//}}}

fn step(args: &StepArgs) -> StepReturn {

    let (new_step, bracket) = match findStepCase(args) {
        1 => stepCase1(args),
        2 => stepCase2(args),
        3 => stepCase3(args),
        4 => stepCase4(args),
        _ => panic!("Unexpected case"),
    };

    // update interval with the minimizer

    todo!()
}
