//! Implementation of the More-Thuente line search.
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common as com;
use super::common::{Error, LineSearch, Returns};
use crate::RealFn1;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------
const XTRAPL: f64 = 1.1;
const XTRAPU: f64 = 4.0;

//{{{ struct: Options
#[derive(Default, Copy, Clone)]
pub struct Options {
    pub ls_opts: com::Options,
    pub maxiter: usize,
}
//}}}
//{{{ struct: Values
#[derive(Debug, Default, Copy, Clone)]
struct Values {
    alpha: f64,
    phi: f64,
    dphi: f64,
}
//}}}
//{{{ impl: Values
impl Values {
    fn modify_forward(&mut self, gtest: f64) {
        self.phi -= self.alpha * gtest;
        self.dphi -= gtest;
    }

    fn modify_back(&mut self, gtest: f64) {
        self.phi += self.alpha * gtest;
        self.dphi += gtest;
    }
}
//}}}
//{{{ struct: ThuenteData
#[derive(Default)]
struct ThuenteData {
    finit: f64,
    ginit: f64,
    gtest: f64,
    ftest: f64,
    width: f64,
    width1: f64,
    stmin: f64,
    stmax: f64,
}
//}}}
//{{{ struct: Thuente
#[derive(Default)]
pub struct Thuente<F: RealFn1> {
    pub opts: Options,
    pub(crate) f: F,
    interval_endpoint1: Values,
    interval_endpoint2: Values,
    bracketed: bool,
    stage: u8,
    data: ThuenteData,
}
//}}}
//{{{ impl: Thuente
impl<F: RealFn1> Thuente<F> {
    //{{{ fn: new
    pub fn new(f: F, opts: Options) -> Self {
        Self {
            opts,
            f,
            interval_endpoint1: Values {
                alpha: 0.0,
                phi: 0.0,
                dphi: 0.0,
            },
            interval_endpoint2: Values {
                alpha: 0.0,
                phi: 0.0,
                dphi: 0.0,
            },
            bracketed: false,
            stage: 1,
            data: ThuenteData::default(),
        }
    }
    //}}}
    //{{{ fn: interval_width
    fn interval_width(&self) -> f64 {
        return (self.interval_endpoint1.alpha - self.interval_endpoint2.alpha).abs();
    }
    //}}}
    //{{{ fn: interval_midpoint
    fn interval_midpoint(&self) -> f64 {
        return 0.5 * (self.interval_endpoint1.alpha + self.interval_endpoint2.alpha);
    }
    //}}}
    //{{{ fn: interval_min
    fn interval_min(&self) -> f64 {
        return f64::min(self.interval_endpoint1.alpha, self.interval_endpoint2.alpha);
    }
    //}}}
    //{{{ fn: interval_max
    fn interval_max(&self) -> f64 {
        return f64::max(self.interval_endpoint1.alpha, self.interval_endpoint2.alpha);
    }
    //}}}
    //{{{ fn: initialize
    fn initialize(&mut self, phi0: f64, dphi0: f64, alpha1: f64) {
        let ftol = self.opts.ls_opts.c1;
        let finit = phi0;
        let ginit = dphi0;
        let gtest = ftol * ginit;

        let width = self.opts.ls_opts.step_max - self.opts.ls_opts.step_min;
        let width1 = width / 0.5;
        let stmin = 0.0;
        let stmax = alpha1 + XTRAPU * alpha1;
        self.interval_endpoint1 = Values {
            alpha: 0.0,
            phi: finit,
            dphi: ginit,
        };
        self.interval_endpoint2 = Values {
            alpha: 0.0,
            phi: finit,
            dphi: ginit,
        };

        self.data = ThuenteData {
            finit,
            ginit,
            gtest,
            ftest: 0.0,
            width,
            width1,
            stmin,
            stmax,
        };
    }
    //}}}
    //{{{ fn: iter_step
    fn iter_step(&mut self, cur_values: &Values) -> Values {
        error!(target: "ls", "--- Entering iter_step ---");

        let &ThuenteData {
            finit,
            ginit: _,
            gtest,
            mut ftest,
            mut width,
            mut width1,
            mut stmin,
            mut stmax,
        } = &self.data;

        ftest = finit + cur_values.alpha * gtest;

        if self.stage == 1 && cur_values.phi <= ftest && cur_values.dphi >= 0.0 {
            self.stage = 2;
        }

        let mut new_step;

        if self.stage == 1 && cur_values.phi < self.interval_endpoint1.phi && cur_values.phi > ftest
        {
            trace!(target: "ls", "entering stage 1");
            // Define the modified function and derivative values.
            let mut interval_intpoint = cur_values.clone();
            interval_intpoint.modify_forward(gtest);

            let mut interval_endpoint_1 = self.interval_endpoint1.clone();
            interval_endpoint_1.modify_forward(gtest);

            let mut interval_endpoint_2 = self.interval_endpoint2.clone();
            interval_endpoint_2.modify_forward(gtest);

            let ret = bracket_step(&StepArgs {
                interval_endpoint_1,
                interval_endpoint_2,
                interval_intpoint,
                bracketed: self.bracketed,
                step_min: self.data.stmin,
                step_max: self.data.stmax,
            });

            self.interval_endpoint1 = ret.interval_endpoint_1;
            self.interval_endpoint1.modify_back(gtest);
            self.interval_endpoint2 = ret.interval_endpoint_2;
            self.interval_endpoint2.modify_back(gtest);
            self.bracketed = ret.bracket;
            new_step = ret.alpha;
        } else {
            trace!(target: "ls", "entering stage 2");
            let ret = bracket_step(&StepArgs {
                interval_endpoint_1: self.interval_endpoint1.clone(),
                interval_endpoint_2: self.interval_endpoint2.clone(),
                interval_intpoint: cur_values.clone(),
                bracketed: self.bracketed,
                step_min: self.data.stmin,
                step_max: self.data.stmax,
            });

            self.interval_endpoint1 = ret.interval_endpoint_1;
            self.interval_endpoint2 = ret.interval_endpoint_2;
            self.bracketed = ret.bracket;
            new_step = ret.alpha
        }

        // decide if bisection is needed
        if self.bracketed {
            if self.interval_width() > 0.66 * self.data.width1 {
                new_step = self.interval_midpoint();
            }
            self.data.width1 = self.data.width;
            self.data.width = self.interval_width();
        }

        // Set the minimum and maximum steps allowed for stp.
        if self.bracketed {
            self.data.stmin = self.interval_min();
            self.data.stmax = self.interval_max();
        } else {
            self.data.stmin = new_step + XTRAPL * (new_step - self.interval_endpoint1.alpha);
            self.data.stmax = new_step + XTRAPU * (new_step - self.interval_endpoint1.alpha);
        }

        // force step to be within bounds
        new_step = new_step.clamp(self.opts.ls_opts.step_min, self.opts.ls_opts.step_max);

        // if further progress not possible, let new_step be the best obtained so far
        let step_out_of_bounds =
            self.bracketed && (new_step <= self.data.stmin || new_step >= self.data.stmax);
        let interval_too_small =
            self.bracketed && (self.data.stmax - self.data.stmin < 1e-14 * self.data.stmax);
        if step_out_of_bounds || interval_too_small {
            new_step = self.interval_endpoint1.alpha;
        }

        let new_phi = self.f.eval(new_step);
        let new_dphi = self.f.diff(new_step);
        return Values {
            alpha: new_step,
            phi: new_phi,
            dphi: new_dphi,
        };
    }
    //}}}
    //{{{ fn: convergence_reached
    fn convergence_reached(&self, values: &Values) -> bool {
        let &Values {
            alpha,
            phi: f,
            dphi: g,
        } = &values;

        let ftest = self.data.finit + alpha * self.data.gtest;
        let gtol = self.opts.ls_opts.c2;
        return *f <= ftest && g.abs() <= -gtol * self.data.ginit;
    }
    //}}}
}
//}}}
//{{{ impl: LineSearch for Thuente
impl<F: RealFn1> LineSearch for Thuente<F> {
    type Function = F;

    fn search(&mut self, phi0: f64, dphi0: f64, alpha1: f64) -> Result<Returns, Error> {
        //{{{ trace
        error!(target: "ls", "--- Entering search ---");
        info!(target: "ls", "phi0={phi0:1.3e} dphi0={dphi0:1.3e} alpha1 = {alpha1:1.3e}");
        //}}}
        self.initialize(phi0, dphi0, alpha1);
        let mut cur_step = Values {
            alpha: alpha1,
            phi: self.f.eval(alpha1),
            dphi: self.f.diff(alpha1),
        };

        for iter in 0..self.opts.maxiter {
            //{{{ trace
            trace!(target: "ls", ".................................... iter = {iter}");
            trace!(target: "ls", "{cur_step:?}");
            //}}}
            if self.convergence_reached(&cur_step) {
                //{{{ trace
                trace!(target: "ls", "reached convergence");
                error!(target: "ls", "--- Leaving search ---");
                //}}}
                return Ok(Returns {
                    alpha: cur_step.alpha,
                    phi_alpha: cur_step.phi,
                    dphi_alpha: cur_step.dphi,
                });
            }

            cur_step = self.iter_step(&cur_step);
        }

        trace!(target: "ls", "Max iterations reached, min not found");
        error!(target: "ls", "--- Leaving search ---");
        Err(Error::NoStepFound)
    }

    fn update_fcn(&mut self, fcn: Self::Function) {
        self.f = fcn;
    }
}
//}}}
//{{{ fn: bracket_step
fn bracket_step(args: &StepArgs) -> StepReturn {
    error!(target: "ls", "--- entering bracket_step ---");
    let (new_step, bracket) = match find_step_case(args) {
        1 => step_case1(args),
        2 => step_case2(args),
        3 => step_case3(args),
        4 => step_case4(args),
        _ => panic!("Unexpected case"),
    };
    // update interval with the minimizer

    let mut ret = StepReturn::default();
    ret.alpha = new_step;
    ret.bracket = bracket;

    // update interval containting the minimizer
    if args.interval_intpoint.phi > args.interval_endpoint_1.phi {
        ret.interval_endpoint_1 = args.interval_endpoint_1;
        ret.interval_endpoint_2 = args.interval_intpoint;
    } else {
        ret.interval_endpoint_1 = args.interval_intpoint;

        if args.deriv_sign_is_opposite() {
            ret.interval_endpoint_2 = args.interval_endpoint_1;
        } else {
            ret.interval_endpoint_2 = args.interval_endpoint_2;
        }
    }

    error!(target: "ls", "--- leaving bracket_step ---");
    ret
}
//}}}
//{{{ coll: StepArgs
#[derive(Default, Debug)]
struct StepArgs {
    interval_endpoint_1: Values,
    interval_endpoint_2: Values,
    interval_intpoint: Values,
    bracketed: bool,
    step_min: f64,
    step_max: f64,
}
impl StepArgs {
    fn deriv_sign_is_opposite(&self) -> bool {
        let sign_intpint = self.interval_intpoint.dphi.signum();
        let sign_endpoint_1 = self.interval_endpoint_1.dphi.signum();
        return sign_intpint * sign_endpoint_1 < 0.0;
    }
}
//}}}
//{{{ struct: StepReturn
#[derive(Default)]
struct StepReturn {
    interval_endpoint_1: Values,
    interval_endpoint_2: Values,
    alpha: f64,
    bracket: bool,
}
//}}}
//{{{ fn: find_step_case
fn find_step_case(args: &StepArgs) -> u8 {
    let cur_phi_is_greater = args.interval_intpoint.phi > args.interval_endpoint_1.phi;
    // case 1: A higher function value. The minimum is bracketed.
    if cur_phi_is_greater {
        return 1;
    }

    // case 2: Lower function value and derivatives of opposite signs. Min is bracketed
    if args.deriv_sign_is_opposite() {
        return 2;
    }

    let decreasing_derivative_magnitude =
        args.interval_intpoint.dphi.abs() < args.interval_endpoint_1.dphi.abs();
    // case 3: Lower function value, derivatives same sign, magnitude of derivative is decreasing
    // in direction of step, bracketing remains unchanged
    if decreasing_derivative_magnitude {
        return 3;
    }

    // case 4: Lower function value, derivatives same sign, magnitude of derivative is increasing
    // in direction of step.
    return 4;
}
//}}}
//{{{ fn: step_case_1
/// First case: A higher function value. The minimum is bracketed.
/// If the cubic step is closer to stx than the quadratic step, the
/// cubic step is taken, otherwise the average of the cubic and
/// quadratic steps is taken.
fn step_case1(args: &StepArgs) -> (f64, bool) {
    error!(target: "ls", "--- entering step_case1 ---");
    let &Values {
        alpha: stx,
        phi: fx,
        dphi: dx,
    } = &args.interval_endpoint_1;
    let &Values {
        alpha: stp,
        phi: fp,
        dphi: dp,
    } = &args.interval_intpoint;
    let theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
    let s = theta.abs().max(dx.abs()).max(dp.abs());
    let sign = if stp < stx { -1.0 } else { 1.0 };
    let gamma = sign * s * ((theta / s).powi(2) - (dx / s) * (dp / s)).sqrt();
    let p = (gamma - dx) + theta;
    let q = (gamma - dx) + gamma + dp;
    let r = p / q;
    let cubic_step = stx + r * (stp - stx);
    let quad_step = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx);

    let bracket = true;
    let cubic_step_smaller = (cubic_step - stx).abs() <= (quad_step - stx).abs();
    let step = if cubic_step_smaller {
        cubic_step
    } else {
        0.5 * (quad_step + cubic_step)
    };

    error!(target: "ls", "--- leaving step_case1 ---");
    return (step, bracket);
}
//}}}
//{{{ fn: step_case_2
/// Second case: A lower function value and derivatives of opposite
/// sign. The minimum is bracketed. If the cubic step is farther from
/// stp than the secant step, the cubic step is taken, otherwise the
/// secant step is taken.
fn step_case2(args: &StepArgs) -> (f64, bool) {
    error!(target: "ls", "--- entering step_case2 ---");
    let &Values {
        alpha: stx,
        phi: fx,
        dphi: dx,
    } = &args.interval_endpoint_1;
    let &Values {
        alpha: stp,
        phi: fp,
        dphi: dp,
    } = &args.interval_intpoint;
    let theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
    let s = theta.abs().max(dx.abs()).max(dp.abs());
    let sign = if stp > stx { -1.0 } else { 1.0 };
    let gamma = sign * s * ((theta / s).powi(2) - (dx / s) * (dp / s)).sqrt();
    let p = (gamma - dp) + theta;
    let q = (gamma - dp) + gamma + dx;
    let r = p / q;
    let cubic_step = stp + r * (stx - stp);
    let quad_step = stp + (dp / (dp - dx)) * (stx - stp);

    let bracket = true;
    let cubic_step_larger = (cubic_step - stp).abs() > (quad_step - stp).abs();
    let step = if cubic_step_larger {
        cubic_step
    } else {
        quad_step
    };

    error!(target: "ls", "--- leaving step_case2 ---");
    return (step, bracket);
}
//}}}
//{{{ fn: step_case_3
fn step_case3(args: &StepArgs) -> (f64, bool) {
    error!(target: "ls", "--- entering step_case2 ---");
    let &Values {
        alpha: stx,
        phi: fx,
        dphi: dx,
    } = &args.interval_endpoint_1;
    let &Values {
        alpha: stp,
        phi: fp,
        dphi: dp,
    } = &args.interval_intpoint;
    let &Values {
        alpha: sty,
        phi: _,
        dphi: _,
    } = &args.interval_endpoint_2;

    // The cubic step is computed only if the cubic tends to infinity
    // in the direction of the step or if the minimum of the cubic
    // is beyond stp. Otherwise the cubic step is defined to be the
    // secant step.
    let theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
    let s = theta.abs().max(dx.abs()).max(dp.abs());
    let sign = if stp > stx { -1.0 } else { 1.0 };
    let gamma = sign * s * ((theta / s).powi(2) - (dx / s) * (dp / s)).sqrt();
    let p = (gamma - dp) + theta;
    let q = (gamma + (dx - dp)) + gamma;
    let r = p / q;

    // The case gamma = 0 only arises if the cubic does not tend
    // to infinity in the direction of the step.
    let cubic_step = if r < 0.0 && gamma != 0.0 {
        stp + r * (stx - stp)
    } else if stp > stx {
        args.step_max
    } else {
        args.step_min
    };

    let quad_step = stp + (dp / (dp - dx)) * (stx - stp);

    let step = if args.bracketed {
        // # A minimizer has been bracketed. If the cubic step is
        // # closer to stp than the secant step, the cubic step is
        // # taken, otherwise the secant step is taken.
        let cubic_step_smaller = (cubic_step - stp).abs() < (quad_step - stp).abs();
        let step_tmp = if cubic_step_smaller {
            cubic_step
        } else {
            quad_step
        };

        let step = if step_tmp > stx {
            (stp + 0.66 * (sty - stp)).min(step_tmp)
        } else {
            (stp + 0.66 * (sty - stp)).max(step_tmp)
        };

        step
    } else {
        // A minimizer has not been bracketed. If the cubic step is
        // farther from stp than the secant step, the cubic step is
        // taken, otherwise the secant step is taken.
        let cubic_step_smaller = (cubic_step - stp).abs() < (quad_step - stp).abs();
        let step_tmp = if cubic_step_smaller {
            cubic_step
        } else {
            quad_step
        };
        step_tmp.clamp(args.step_min, args.step_max)
    };

    error!(target: "ls", "--- leaving step_case3 ---");
    return (step, args.bracketed);
}
//}}}
//{{{ fn: step_case_4
fn step_case4(args: &StepArgs) -> (f64, bool) {
    error!(target: "ls", "--- entering step_case4 ---");
    let &Values {
        alpha: stx,
        phi: _,
        dphi: dx,
    } = &args.interval_endpoint_1;
    let &Values {
        alpha: stp,
        phi: fp,
        dphi: dp,
    } = &args.interval_intpoint;
    let &Values {
        alpha: sty,
        phi: fy,
        dphi: dy,
    } = &args.interval_endpoint_2;

    if !args.bracketed {
        if stp > stx {
            return (args.step_max, args.bracketed);
        } else {
            return (args.step_min, args.bracketed);
        }
    }

    let theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp;
    let s = theta.abs().max(dx.abs()).max(dp.abs());
    let sign = if stp > sty { -1.0 } else { 1.0 };
    let gamma = sign * s * ((theta / s).powi(2) - (dy / s) * (dp / s)).sqrt();
    let p = (gamma - dp) + theta;
    let q = ((gamma - dp) + gamma) + dy;
    let r = p / q;
    let cubic_step = stp + r * (sty - stp);

    error!(target: "ls", "--- leaving step_case4 ---");
    return (cubic_step, args.bracketed);
}
//}}}
