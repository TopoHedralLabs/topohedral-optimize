//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::Options as BoundConstrainedOptions;
use super::common::{lift, restrict};
use super::utils::CircularBuffer;
use crate::bound_constrained::asa::Phase::Ua;
use crate::common::{Minimizer, Vector};
use crate::constraints::BoundsConstraints;
use crate::constraints::{BoundSignature, BoundStatus};
use crate::unconstrained::{minimize, UnconstrainedMethod};
use crate::ConvergedReason;
use crate::{IterData, RealFn};
//}}}
//{{{ dep imports
#[allow(unused_imports)]
use topohedral_linalg::MatrixOps;
use topohedral_linalg::{ReduceOps, VectorOps};
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

const ALPHA: f64 = 0.5;
const BETA: f64 = 1.5;
const SMALL: f64 = 1e-20;
const DEFUALT_MU: f64 = 0.1;
const DEFAULT_RHO: f64 = 0.5;
const DEFAULT_N1: usize = 2;
const DEFAULT_N2: usize = 1;
const DEFAULT_MEMORY: usize = 8;
const DEFAULT_DELTA: f64 = 1e-4;
const DEFAULT_ETA: f64 = 0.5;
const DEFAULT_ALPHA_MIN: f64 = 1e-20;
const DEFAULT_ALPHA_MAX: f64 = 1e20;

//{{{ struct: Options
#[derive(Clone)]
pub struct Options
{
    pub bound_opts: BoundConstrainedOptions,
    /// Settings for internal minimization
    pub unconstrained_method: UnconstrainedMethod,
    /// Tolerance for ||g_I|| < mu * ||d^1|| means "face solved enough"
    pub mu: f64,
    /// Decay factor applied to mu when test triggers
    pub rho: f64,
    /// First counter controlling switch phase
    pub n1: usize,
    /// Second counter controlling switch phase
    pub n2: usize,
    /// No. of previous steps stored in memory
    pub memory: usize,
    /// Armijo descent constant
    pub delta: f64,
    /// Backtracking factor
    pub eta: f64,
    /// Minimum allowed step
    pub alpha_min: f64,
    /// Maximum allowed step
    pub alpha_max: f64,
}
//}}}
//{{{ impl Optoins
impl Options
{
    pub fn new(
        bound_opts: BoundConstrainedOptions,
        unconstrained_method: UnconstrainedMethod,
    ) -> Self
    {
        Self {
            bound_opts,
            unconstrained_method,
            mu: DEFUALT_MU,
            rho: DEFAULT_RHO,
            n1: DEFAULT_N1,
            n2: DEFAULT_N2,
            memory: DEFAULT_MEMORY,
            delta: DEFAULT_DELTA,
            eta: DEFAULT_ETA,
            alpha_min: DEFAULT_ALPHA_MIN,
            alpha_max: DEFAULT_ALPHA_MAX,
        }
    }
}

//}}}
//{{{ struct: BoundedFunction
#[derive(Debug, Clone)]
struct RestrictedFunction<F: RealFn>
{
    fcn: F,
    bound_statuses: Vec<BoundStatus>,
}
//}}}
//{{{ impl BoundedFunction
impl<F: RealFn> RestrictedFunction<F>
{
    //{{{ fn: new
    #[trace_fn]
    fn new(
        fcn: F,
        _x: &Vector,
        bound_statuses: Vec<BoundStatus>,
    ) -> Self
    {
        RestrictedFunction {
            fcn,
            bound_statuses,
        }
    }
    //}}}
}
//}}}
//{{{ impl: RealFn for BoundedFunction
impl<F: RealFn> RealFn for RestrictedFunction<F>
{
    //{{{ fn: dimension
    #[trace_fn]
    fn dimension(&self) -> usize
    {
        self.bound_statuses
            .iter()
            .filter(|status| **status == BoundStatus::Free)
            .count()
    }
    //}}}
    //{{{ fn: eval
    #[trace_fn]
    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        let x_full = lift(&self.bound_statuses, x);
        self.fcn.eval(&x_full)
    }
    //}}}
    //{{{ fn: grad
    #[trace_fn]
    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector
    {
        let x_full = lift(&self.bound_statuses, x);
        let grad_f_full = self.fcn.grad(&x_full);
        restrict(&self.bound_statuses, &grad_f_full)
    }
    //}}}
}
//}}}
//{{{ struct: ActiveSetAlgorithm
pub struct ActiveSetAlgorithm<F: RealFn>
{
    fcn: F,
    bounds: BoundsConstraints,
    x_init: Vector,
    norm_grad_fx_init: f64,
    opts: Options,

    fn_history: CircularBuffer<f64>,
    active_signature_history: CircularBuffer<BoundSignature>,
}
//}}}
//{{{ enum: Phase
#[derive(Debug)]
enum Phase
{
    Ngpa,
    Ua,
}
//}}}
//{{{ impl: ActiveSetAlgorithm
impl<F: RealFn> ActiveSetAlgorithm<F>
{
    #[trace_fn]
    pub fn new(
        mut fcn: F,
        bounds: BoundsConstraints,
        mut x0: Vector,
        opts: Options,
    ) -> Self
    {
        bounds.clamp(&mut x0);
        let grad_0 = fcn.grad(&x0);
        let negative_grad_0 = -grad_0.clone();
        let projectd_grad_0 = bounds.projected_direction(&x0, &negative_grad_0, 1.0);
        let norm_projected_grad_0 = projectd_grad_0.abs_max().unwrap();

        let n1 = opts.n1;
        let m = opts.memory;

        //{{{ trace
        info!(target: "asa", "Initializing active-set algorithm");
        trace!(target: "asa", "Initial solution: {}", x0.clone().transpose());
        info!(target: "asa", "Initial ||∇f_proj||_∞ = {norm_projected_grad_0:.4e}");
        trace!(target: "asa", "Active-set history length = {n1}, function history length = {m}");
        //}}}

        Self {
            fcn,
            bounds,
            x_init: x0,
            norm_grad_fx_init: norm_projected_grad_0,
            opts,
            fn_history: CircularBuffer::new(m),
            active_signature_history: CircularBuffer::new(n1),
        }
    }

    #[trace_fn]
    fn is_converged(
        &self,
        iter_k: &IterData,
    ) -> Option<ConvergedReason>
    {
        //{{{ trace
        trace!(target: "asa", "Checking convergence");
        //}}}
        let projected_grad =
            self.bounds
                .projected_direction(&iter_k.x, &(-iter_k.grad_fx.clone()), 1.0);
        let projected_grad_norm = projected_grad.norm();
        let _projected_grad_ratio = if self.norm_grad_fx_init > 0.0
        {
            projected_grad_norm / self.norm_grad_fx_init
        }
        else
        {
            0.0
        };
        let rtol_reached =
            projected_grad_norm < self.opts.bound_opts.base_opts.grad_rtol * self.norm_grad_fx_init;

        //{{{ trace
        trace!(target: "asa",
            "||∇f_proj|| / ||∇f_proj(0)|| = {_projected_grad_ratio:.4e}, ||∇f_proj|| = {projected_grad_norm:.4e}"
        );
        //}}}

        if rtol_reached
        {
            //{{{ trace
            trace!(target: "asa", "Rtol reached");
            //}}}
            return Some(ConvergedReason::Rtol);
        }

        let atol_reached = projected_grad_norm < self.opts.bound_opts.base_opts.grad_atol;

        if atol_reached
        {
            //{{{ trace
            trace!(target: "asa", "Atol reached");
            //}}}
            return Some(ConvergedReason::Atol);
        }

        None
    }

    #[trace_fn]
    fn bb_step(
        &mut self,
        s: &Vector,
        y: &Vector,
        fallback: f64,
    ) -> f64
    {
        let s_dot_y = s.dot(y);
        if s_dot_y <= 0.0
        {
            //{{{ trace
            debug!(target: "asa", "Skipping BB update because sᵀy = {s_dot_y:.4e}; fallback alpha = {fallback:.4e}");
            //}}}
            return fallback;
        }
        let a = s.dot(s) / s_dot_y;
        let alpha = a.clamp(self.opts.alpha_min, self.opts.alpha_max);
        //{{{ trace
        debug!(target: "asa", "BB update: sᵀy = {s_dot_y:.4e}, raw alpha = {a:.4e}, clamped alpha = {alpha:.4e}");
        //}}}
        alpha
    }

    #[trace_fn]
    fn ngpa_step(
        &mut self,
        iter_k: &IterData,
        alpha_init: f64,
    ) -> Option<IterData>
    {
        let IterData {
            x,
            fx,
            grad_fx,
            norm_grad_fx: _,
        } = iter_k;

        let d = self
            .bounds
            .projected_direction(x, &(-grad_fx.clone()), alpha_init);
        let _d_norm = d.norm();
        let _d_inf_norm = d.abs_max().unwrap_or(0.0);

        //{{{ trace
        debug!(target: "asa", "NGPA step: alpha_init = {alpha_init:.4e}, ||d|| = {_d_norm:.4e}, ||d||_∞ = {_d_inf_norm:.4e}");
        //}}}

        if d.abs_max().unwrap() < SMALL
        {
            //{{{ trace
            debug!(target: "asa", "Projected direction is below SMALL = {SMALL:.4e}");
            //}}}
            return None;
        }

        let f_max = self.fn_history.max().unwrap_or(*fx);
        let delta = self.opts.delta;
        let mut alpha = 1.0;
        let mut x_trial: Vector = (x + &d).into();
        let mut f_trial = self.fcn.eval(&x_trial);
        let gradfk_dot_d = grad_fx.dot(&d);

        //{{{ trace
        trace!(target: "asa", "Running nonmonotone Armijo search: f_max = {f_max:.4e}, gᵀd = {gradfk_dot_d:.4e}, delta = {delta:.4e}");
        //}}}
        let max_iterations = 25;
        for _i in 0..max_iterations
        {
            let armijo_rhs = f_max + delta * alpha * gradfk_dot_d;
            trace!(target: "asa", "Armijo trial {_i}: alpha = {alpha:.4e}, f_trial = {f_trial:.4e}, rhs = {armijo_rhs:.4e}");
            if f_trial < armijo_rhs || alpha < SMALL
            {
                //{{{ trace
                debug!(target: "asa", "Accepted NGPA step at trial {_i}: alpha = {alpha:.4e}, f_trial = {f_trial:.4e}");
                //}}}
                break;
            }
            alpha *= self.opts.eta;
            x_trial = (x + alpha * &d).into();
            f_trial = self.fcn.eval(&x_trial)
        }

        let grad_fx_new = self.fcn.grad(&x_trial);
        let norm_grad_fx_new = grad_fx_new.norm();
        //{{{ trace
        debug!(target: "asa", "NGPA accepted point: f = {f_trial:.4e}, ||∇f|| = {norm_grad_fx_new:.4e}");
        //}}}
        Some(IterData {
            x: x_trial,
            fx: f_trial,
            grad_fx: grad_fx_new,
            norm_grad_fx: norm_grad_fx_new,
        })
    }

    #[trace_fn]
    fn undecided_set_is_empy(
        &self,
        x: &Vector,
        inactive_grad: &Vector,
        full_grad_norm: f64,
    ) -> bool
    {
        if full_grad_norm < SMALL
        {
            //{{{ trace
            trace!(target: "asa", "Undecided set empty because projected gradient norm is below SMALL");
            //}}}
            return true;
        }

        let thresh_g = full_grad_norm.powf(ALPHA);
        let thresh_x = full_grad_norm.powf(BETA);
        let distances = self.bounds.all_distances(x);

        //{{{ trace
        trace!(target: "asa", "Checking undecided set: thresh_g = {thresh_g:.4e}, thresh_x = {thresh_x:.4e}");
        //}}}

        for (_i, (gi, di)) in inactive_grad.iter().zip(distances.iter()).enumerate()
        {
            if gi.abs() >= thresh_g && *di > thresh_x
            {
                //{{{ trace
                debug!(target: "asa", "Undecided variable {_i}: |g_i| = {:.4e}, distance = {:.4e}", gi.abs(), di);
                //}}}
                return false;
            }
        }
        //{{{ trace
        trace!(target: "asa", "Undecided set is empty");
        //}}}
        true
    }

    #[trace_fn]
    fn active_sets_are_stable(&self) -> bool
    {
        if self.active_signature_history.len() < self.opts.n1
        {
            //{{{ trace
            trace!(
                target: "asa",
                "Active-set history not full yet: len = {}, required = {}",
                self.active_signature_history.len(),
                self.opts.n1
            );
            //}}}
            return false;
        }
        let mut all_equal = true;
        let most_recent_sig = self.active_signature_history.newest().unwrap().clone();
        for sig in self.active_signature_history.iter()
        {
            all_equal = all_equal && (*sig == most_recent_sig);
        }
        //{{{ trace
        debug!(target: "asa", "Active-set stability over last {} signatures: {all_equal}", self.opts.n1);
        //}}}
        return all_equal;
    }

    fn print_status(
        &self,
        _k: u64,
        _iter_k: &IterData,
    )
    {
        info!(target: "asa", ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> k = {_k}");
        info!(target: "asa", "Current values: {_iter_k}");
        trace!(target: "asa", "x: {}", _iter_k.x.clone().transpose());
        trace!(target: "asa", "∇f: {}", _iter_k.grad_fx.clone().transpose());
        let projected_grad =
            self.bounds
                .projected_direction(&_iter_k.x, &(-_iter_k.grad_fx.clone()), 1.0);
        let projected_grad_norm = projected_grad.norm();
        let _projected_grad_ratio = if self.norm_grad_fx_init > 0.0
        {
            projected_grad_norm / self.norm_grad_fx_init
        }
        else
        {
            0.0
        };
        info!(target: "asa", "Convergence measures:");
        info!(target: "asa", "||∇f_proj(k)|| / ||∇f_proj(0)|| = {_projected_grad_ratio:.4e}");
        info!(target: "asa", "||∇f_proj(k)|| = {projected_grad_norm:.4e}");
        trace!(target: "asa", "∇f_proj: {}", projected_grad.transpose());
    }
}
//}}}
//{{{ impl: Minimizer for ActiveSetAlgorithm
impl<F: RealFn> Minimizer for ActiveSetAlgorithm<F>
{
    type Error = super::common::Error;

    #[trace_fn]
    fn minimize(&mut self) -> Result<crate::Returns, Self::Error>
    {
        let mut iter_k_prev = IterData::new(self.fcn.clone(), &self.x_init);
        let mut iter_k = IterData::new(self.fcn.clone(), &self.x_init);
        let mut phase = Phase::Ngpa;
        let mut mu = self.opts.mu;
        let mut alpha_bb = 1.0;

        self.fn_history.append(iter_k.fx);
        self.active_signature_history
            .append(self.bounds.active_signature(&iter_k.x));

        info!(target: "asa", "Starting active-set iterations with phase = {phase:?}, mu = {mu:.4e}, alpha_bb = {alpha_bb:.4e}");

        for k in 0..self.opts.bound_opts.base_opts.max_iter
        {
            self.print_status(k, &iter_k);
            if let Some(reason) = self.is_converged(&iter_k)
            {
                //{{{ trace
                info!(target: "asa", "=============================================");
                info!(target: "asa", "Converging with reason {reason:?}");
                trace!(target: "asa", "fx = {:.4e} x = {}", iter_k.fx, iter_k.x.clone().transpose());
                info!(target: "asa", "=============================================");
                //}}}
                return Ok(crate::Returns {
                    xmin: iter_k.x,
                    fmin: iter_k.fx,
                    reason,
                    num_iterations: k as usize,
                    num_fun_evals: 0,
                    num_grad_evals: 0,
                });
            }

            match phase
            {
                Phase::Ngpa =>
                {
                    //{{{ trace
                    debug!(target: "asa", "Entering NGPA phase with alpha_bb = {alpha_bb:.4e}, mu = {mu:.4e}");
                    //}}}
                    iter_k_prev.copy_from(&iter_k);

                    let IterData {
                        x: x_prev,
                        fx: _,
                        grad_fx: grad_fx_prev,
                        norm_grad_fx: _,
                    } = iter_k_prev.clone();

                    let npga_ok = self.ngpa_step(&iter_k, alpha_bb);

                    if npga_ok.is_none()
                    {
                        //{{{ trace
                        debug!(target: "asa", "NGPA could not produce a step; terminating iteration loop");
                        //}}}
                        break;
                    }

                    iter_k.copy_from(&npga_ok.unwrap());

                    let IterData {
                        x,
                        fx,
                        grad_fx,
                        norm_grad_fx: _,
                    } = &iter_k;

                    self.fn_history.append(*fx);
                    self.active_signature_history
                        .append(self.bounds.active_signature(x));

                    let s = (x - x_prev).into();
                    let y = (grad_fx - grad_fx_prev).into();
                    alpha_bb = self.bb_step(&s, &y, alpha_bb);
                    //{{{ trace
                    trace!(target: "asa", "alpha_bb = {alpha_bb:.4e}");
                    //}}}

                    let projected_grad =
                        self.bounds.projected_direction(x, &(-grad_fx.clone()), 1.0);
                    let projected_grad_norm = projected_grad.norm();
                    let inactive_grad = self.bounds.masked_gradient(x, grad_fx);
                    let inactive_grad_norm = inactive_grad.norm();
                    let _switch_threshold = mu * projected_grad_norm;

                    //{{{ trace
                    debug!(
                        target: "asa",
                        "NGPA metrics: ||∇f_proj|| = {projected_grad_norm:.4e}, ||∇f_inactive|| = {inactive_grad_norm:.4e}, mu ||∇f_proj|| = {_switch_threshold:.4e}"
                    );
                    //}}}

                    if self.undecided_set_is_empy(x, &inactive_grad, projected_grad_norm)
                    {
                        if inactive_grad_norm < mu * projected_grad_norm
                        {
                            let _old_mu = mu;
                            mu *= self.opts.rho;
                            //{{{ trace
                            debug!(target: "asa", "Undecided set empty and inactive gradient is small; shrinking mu from {_old_mu:.4e} to {mu:.4e}");
                            //}}}
                        }
                        else
                        {
                            //{{{ trace
                            debug!(target: "asa", "Undecided set empty but inactive gradient is large; switching to UA");
                            //}}}
                            phase = Ua;
                        }
                    }

                    if self.active_sets_are_stable()
                        && inactive_grad_norm >= mu * projected_grad_norm
                    {
                        //{{{ trace
                        debug!(target: "asa", "Active set is stable and inactive gradient is large; switching to UA");
                        //}}}
                        phase = Ua;
                    }
                }
                Phase::Ua =>
                {
                    //{{{ trace
                    debug!(target: "asa", "Entering UA phase with mu = {mu:.4e}");
                    //}}}
                    iter_k_prev.copy_from(&iter_k);

                    let IterData {
                        x,
                        fx: _,
                        grad_fx: _,
                        norm_grad_fx: _,
                    } = &iter_k;

                    let active_count_before = self
                        .bounds
                        .bound_statuses(x, None)
                        .iter()
                        .filter(|status| **status != BoundStatus::Free)
                        .count();

                    let bounds_statuses = self.bounds.bound_statuses(x, None);
                    let _free_count = bounds_statuses
                        .iter()
                        .filter(|status| **status == BoundStatus::Free)
                        .count();
                    //{{{ trace
                    debug!(target: "asa", "UA restricted problem: active_count = {active_count_before}, free_count = {_free_count}");
                    //}}}
                    let restricted_fcn =
                        RestrictedFunction::new(self.fcn.clone(), x, bounds_statuses.clone());

                    let x0 = restrict(&bounds_statuses, x);
                    if x0.is_empty()
                    {
                        //{{{ trace
                        debug!(target: "asa", "UA restricted problem has no free variables; switching to NGPA");
                        //}}}
                        phase = Phase::Ngpa;
                        continue;
                    }

                    let res = minimize(
                        restricted_fcn.clone(),
                        x0,
                        self.opts.unconstrained_method.clone(),
                    );

                    let Ok(res) = res
                    else
                    {
                        //{{{ trace
                        debug!(target: "asa", "UA internal minimization failed; switching to NGPA");
                        //}}}
                        phase = Phase::Ngpa;
                        continue;
                    };

                    //{{{ trace
                    debug!(
                        target: "asa",
                        "UA internal minimization returned f = {:.4e}, reason = {:?}, iterations = {}",
                        res.fmin,
                        res.reason,
                        res.num_iterations
                    );
                    //}}}

                    iter_k.x.copy_from(lift(&bounds_statuses, &res.xmin));
                    self.bounds.clamp(&mut iter_k.x);
                    iter_k.fx = res.fmin;
                    iter_k.grad_fx.copy_from(self.fcn.grad(&iter_k.x));
                    iter_k.fx = self.fcn.eval(&iter_k.x);
                    iter_k.norm_grad_fx = iter_k.grad_fx.norm();

                    let projected_grad_new =
                        self.bounds
                            .projected_direction(&iter_k.x, &(-iter_k.grad_fx.clone()), 1.0);
                    let projected_grad_norm_new = projected_grad_new.norm();
                    let inactive_grad_new = self.bounds.masked_gradient(&iter_k.x, &iter_k.grad_fx);
                    let inactive_grad_norm_new = inactive_grad_new.norm();
                    let active_count_after = self
                        .bounds
                        .bound_statuses(&iter_k.x, None)
                        .iter()
                        .filter(|status| **status != BoundStatus::Free)
                        .count();
                    let _switch_threshold_new = mu * projected_grad_norm_new;

                    //{{{ trace
                    debug!(
                        target: "asa",
                        "UA metrics: ||∇f_proj|| = {projected_grad_norm_new:.4e}, ||∇f_inactive|| = {inactive_grad_norm_new:.4e}, mu ||∇f_proj|| = {_switch_threshold_new:.4e}, active_count_before = {active_count_before}, active_count_after = {active_count_after}"
                    );
                    //}}}

                    self.fn_history.append(iter_k.fx);
                    self.active_signature_history
                        .append(self.bounds.active_signature(&iter_k.x));

                    if inactive_grad_norm_new < mu * projected_grad_norm_new
                    {
                        //{{{ trace
                        debug!(target: "asa", "||∇f_inactive|| < mu ||∇f_proj||; switching to NGPA");
                        //}}}
                        phase = Phase::Ngpa;
                    }
                    else if active_count_after > active_count_before
                        && active_count_after <= active_count_before + self.opts.n2
                        && !self.undecided_set_is_empy(
                            &iter_k.x,
                            &inactive_grad_new,
                            projected_grad_norm_new,
                        )
                    {
                        //{{{ trace
                        debug!(target: "asa", "Active bound count increased within n2 and undecided set remains nonempty; switching to NGPA");
                        //}}}
                        phase = Phase::Ngpa;
                    }
                    else
                    {
                        //{{{ trace
                        debug!(target: "asa", "Continuing in UA");
                        //}}}
                        phase = Phase::Ua;
                    }
                }
            }
        }
        info!(
            target: "asa",
            "Finished ASA loop without explicit convergence; returning final iterate after {} iterations",
            self.opts.bound_opts.base_opts.max_iter
        );
        Ok(crate::Returns {
            xmin: iter_k.x,
            fmin: iter_k.fx,
            reason: ConvergedReason::Atol,
            num_iterations: self.opts.bound_opts.base_opts.max_iter as usize,
            num_fun_evals: 0,
            num_grad_evals: 0,
        })
    }
}
//}}}
