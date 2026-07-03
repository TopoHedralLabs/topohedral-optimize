//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::Options as BoundConstrainedOptions;
use super::common::{lift, restrict};
use super::utils::CircularBuffer;
use crate::bound_constrained::asa::Phase::UA;
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
enum Phase
{
    #[allow(clippy::upper_case_acronyms)]
    NGPA,
    UA,
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

        let n1 = opts.n1;
        let m = opts.memory;

        Self {
            fcn,
            bounds,
            x_init: x0,
            norm_grad_fx_init: projectd_grad_0.abs_max().unwrap(),
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
        trace!(target: "bc", "Checking convergence");
        //}}}
        let projected_grad =
            self.bounds
                .projected_direction(&iter_k.x, &(-iter_k.grad_fx.clone()), 1.0);
        let projected_grad_norm = projected_grad.norm();
        let rtol_reached =
            projected_grad_norm < self.opts.bound_opts.base_opts.grad_rtol * self.norm_grad_fx_init;

        //{{{ trace
        trace!(target: "bc",
            "||∇f_proj|| / |∇f_proj_0|| = {}",
            projected_grad_norm / self.norm_grad_fx_init
        );
        //}}}

        if rtol_reached
        {
            //{{{ trace
            trace!(target: "bc", "Rtol reached");
            //}}}
            return Some(ConvergedReason::Rtol);
        }

        let atol_reached = projected_grad_norm < self.opts.bound_opts.base_opts.grad_atol;

        if atol_reached
        {
            //{{{ trace
            trace!(target: "bc", "Atol reached");
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
            return fallback;
        }
        let a = s.dot(s) / s_dot_y;
        //{{{ trace
        trace!(target: "bc", "a = {a:.4e}");
        //}}}
        a.clamp(self.opts.alpha_min, self.opts.alpha_max)
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

        if d.abs_max().unwrap() < SMALL
        {
            //{{{ trace
            trace!(target: "bc", "Projected grad is small");
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
        trace!(target: "bc", "Running backtracking armijo");
        //}}}
        let max_iterations = 25;
        for _i in 0..max_iterations
        {
            if f_trial < f_max + delta * alpha * gradfk_dot_d || alpha < SMALL
            {
                //{{{ trace
                trace!("Found step i = {_i} alpha = {alpha:.4e} f_trial = {f_trial:.4e}");
                //}}}
                break;
            }
            alpha *= self.opts.eta;
            x_trial = (x + alpha * &d).into();
            f_trial = self.fcn.eval(&x_trial)
        }

        let grad_fx_new = self.fcn.grad(&x_trial);
        let norm_grad_fx_new = grad_fx_new.norm();
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
            return true;
        }

        let thresh_g = full_grad_norm.powf(ALPHA);
        let thresh_x = full_grad_norm.powf(BETA);
        let distances = self.bounds.all_distances(x);

        for (gi, di) in inactive_grad.iter().zip(distances.iter())
        {
            if gi.abs() >= thresh_g && *di > thresh_x
            {
                return false;
            }
        }
        true
    }

    #[trace_fn]
    fn active_sets_are_stable(&self) -> bool
    {
        if self.active_signature_history.len() < self.opts.n1
        {
            return false;
        }
        let mut all_equal = true;
        let most_recent_sig = self.active_signature_history.newest().unwrap().clone();
        for sig in self.active_signature_history.iter()
        {
            all_equal = all_equal && (*sig == most_recent_sig);
        }
        return all_equal;
    }

    fn print_status(
        &self,
        _k: u64,
        _iter_k: &IterData,
    )
    {
        info!(target: "bc", ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> k = {_k}");
        trace!(target: "bc", "x: {}", _iter_k.x.clone().transpose());
        trace!(target: "bc", "∇f: {}", _iter_k.grad_fx.clone().transpose());
        trace!(target: "bc", "∇f_proj: {}", self.bounds.projected_direction(&_iter_k.x, &(-_iter_k.grad_fx.clone()), 1.0).transpose());
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
        let mut phase = Phase::NGPA;
        let mut mu = self.opts.mu;
        let mut alpha_bb = 1.0;

        self.fn_history.append(iter_k.fx);
        self.active_signature_history
            .append(self.bounds.active_signature(&iter_k.x));

        for k in 0..self.opts.bound_opts.base_opts.max_iter
        {
            self.print_status(k, &iter_k);
            if let Some(reason) = self.is_converged(&iter_k)
            {
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
                Phase::NGPA =>
                {
                    //{{{ trace
                    trace!(target: "bc", "Entering NGPA Phase");
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
                    trace!(target: "bc", "alpha_bb = {alpha_bb:.4e}");
                    //}}}

                    let projected_grad =
                        self.bounds.projected_direction(x, &(-grad_fx.clone()), 1.0);
                    let projected_grad_norm = projected_grad.norm();
                    let inactive_grad = self.bounds.masked_gradient(x, grad_fx);
                    let inactive_grad_norm = inactive_grad.norm();

                    if self.undecided_set_is_empy(x, &inactive_grad, projected_grad_norm)
                    {
                        if inactive_grad_norm < mu * projected_grad_norm
                        {
                            mu *= self.opts.rho
                        }
                        else
                        {
                            phase = UA;
                        }
                    }

                    if self.active_sets_are_stable()
                        && inactive_grad_norm >= mu * projected_grad_norm
                    {
                        phase = UA;
                    }
                }
                Phase::UA =>
                {
                    //{{{ trace
                    trace!(target: "bc", "Entering AU Phase");
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
                    let restricted_fcn =
                        RestrictedFunction::new(self.fcn.clone(), x, bounds_statuses.clone());

                    let x0 = restrict(&bounds_statuses, x);
                    if x0.is_empty()
                    {
                        phase = Phase::NGPA;
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
                        phase = Phase::NGPA;
                        continue;
                    };

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

                    self.fn_history.append(iter_k.fx);
                    self.active_signature_history
                        .append(self.bounds.active_signature(&iter_k.x));

                    if inactive_grad_norm_new < mu * projected_grad_norm_new
                    {
                        //{{{ trace
                        trace!(target: "bc", "||∇f_inactive|| < mu ||∇f_proj||");
                        trace!(target: "bc", "Switching to NPGP");
                        //}}}
                        phase = Phase::NGPA;
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
                        trace!(target: "bc", "No. of active bounds has increased");
                        trace!(target: "bc", "Switching to NPGA");
                        //}}}
                        phase = Phase::NGPA;
                    }
                    else
                    {
                        //{{{ trace
                        trace!(target: "bc", "No. of active bounds has decreased");
                        trace!(target: "bc", "Sticking to UA");
                        //}}}
                        phase = Phase::UA;
                    }
                }
            }
        }
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
