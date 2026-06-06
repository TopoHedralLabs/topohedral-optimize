//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::Options as BoundConstrainedOptions;
use super::utils::{CircularBuffer, CircularBufferIter};
use crate::bound_constrained::asa::Phase::UA;
use crate::common::CountingRealFn;
use crate::common::Vector;
use crate::constraints::{BoundSignature, BoundStatus};
use crate::unconstrained::{
    minimize, UnconstrainedConvergedReason, UnconstrainedMethod, UnconstrainedReturns,
    UnonstrainedOptions,
};
use crate::ConvergedReason;
use crate::ConvergedReason::Rtol;
use crate::{bound_constrained::common::BoundConstrainedMinimizer, constraints::BoundsConstraints};
use crate::{IterData, RealFn};
use std::collections::HashSet;
//}}}
//{{{ std imports
use std::sync::{Arc, Mutex};
//}}}
//{{{ dep imports
use approx::RelativeEq;
use serde_json::map::Iter;
use topohedral_linalg::{ReduceOps, VecType, VectorOps};
//}}}
//--------------------------------------------------------------------------------------------------

const ALPHA: f64 = 0.5;
const BETA: f64 = 1.5;
const SMALL: f64 = 1e-20;

//{{{ struct: Options
pub struct Options
{
    bound_opts: BoundConstrainedOptions,
    /// Settings for internal minimization
    unconstrained_method: UnconstrainedMethod,
    /// Tolerance for ||g_I|| < mu * ||d^1|| means "face solved enough"
    mu: f64,
    /// Decay factor applied to mu when test triggers
    rho: f64,
    /// First counter controlling switch phase
    n1: usize,
    /// Second counter controlling switch phase
    n2: usize,
    /// No. of previous steps stored in memory
    memory: usize,
    /// Armijo descent constant
    delta: f64,
    /// Backtracking factor
    eta: f64,
    /// Minimum allowed step
    alpha_min: f64,
    /// Maximum allowed step
    alpha_max: f64,
}
//}}}
//{{{ struct: BoundedFunction
#[derive(Debug, Clone)]
struct RestrictedFunction<F: RealFn>
{
    fcn: F,
    inactive_indices: Vec<usize>,
}
//}}}
//{{{ impl BoundedFunction
impl<F: RealFn> RestrictedFunction<F>
{
    //{{{ fn: new
    fn new(
        fcn: F,
        inactive_indices: Vec<usize>,
    ) -> Self
    {
        RestrictedFunction {
            fcn,
            inactive_indices,
        }
    }
    //}}}
    //{{{ fn: lift
    fn lift(
        &self,
        x: &Vector,
    ) -> Vector
    {
        let n_full = self.fcn.dimension();
        let mut x_full = Vector::zeros_vec(n_full, VecType::Col);
        for (loc_idx, glob_idx) in self.inactive_indices.iter().enumerate()
        {
            x_full[*glob_idx] = x[loc_idx];
        }
        x_full
    }
    //}}}
    //{{{ fn: restrict
    fn restrict(
        &self,
        x: &Vector,
    ) -> Vector
    {
        let n_restricted = self.inactive_indices.len();
        let mut x_restricted = Vector::zeros_vec(n_restricted, VecType::Col);
        for (loc_idx, glob_idx) in self.inactive_indices.iter().enumerate()
        {
            x_restricted[loc_idx] = x[*glob_idx];
        }
        return x_restricted;
    }
    //}}}
}
//}}}
//{{{ impl: RealFn for BoundedFunction
impl<F: RealFn> RealFn for RestrictedFunction<F>
{
    //{{{ fn: dimension
    fn dimension(&self) -> usize
    {
        self.inactive_indices.len()
    }
    //}}}
    //{{{ fn: eval
    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        let x_full = self.lift(x);
        self.fcn.eval(&x_full)
    }
    //}}}
    //{{{ fn: grad
    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector
    {
        let x_full = self.lift(x);
        let grad_f_full = self.fcn.grad(&x_full);
        let grad_f_restricted = self.restrict(&grad_f_full);
        grad_f_restricted
    }
    //}}}
}
//}}}
//{{{ struct: ActiveSetAlgorithm
pub struct ActiveSetAlgorithm<F: RealFn>
{
    fcn: F,
    x_init: Vector,
    norm_grad_fx_init: f64,
    bounds: BoundsConstraints,
    opts: Options,

    fn_history: CircularBuffer<f64>,
    active_signature_history: CircularBuffer<BoundSignature>,
}
//}}}
//{{{ enum: Phase
enum Phase
{
    NGPA,
    UA,
}
//}}}
//{{{ impl: ActiveSetAlgorithm
impl<F: RealFn> ActiveSetAlgorithm<F>
{
    pub fn new(
        mut fcn: F,
        mut x0: Vector,
        bounds: BoundsConstraints,
        opts: Options,
    ) -> Self
    {
        bounds.clamp(&mut x0);
        let grad_0 = fcn.grad(&x0);
        let projectd_grad_0 = bounds.projected_direction(&x0, &grad_0, 1.0);

        let n1 = opts.n1;
        let n2 = opts.n2;
        let m = opts.memory;

        Self {
            fcn: fcn,
            x_init: x0,
            norm_grad_fx_init: projectd_grad_0.abs_max().unwrap(),
            bounds: bounds,
            opts,
            fn_history: CircularBuffer::new(m),
            active_signature_history: CircularBuffer::new(n1),
        }
    }

    fn is_converged(
        &self,
        iter_k: &IterData,
    ) -> Option<ConvergedReason>
    {
        let projected_grad = self
            .bounds
            .projected_direction(&iter_k.x, &iter_k.grad_fx, 1.0);
        let projected_grad_norm = projected_grad.norm();
        let rtol_reached =
            projected_grad_norm < self.opts.bound_opts.grad_rtol * self.norm_grad_fx_init;

        if rtol_reached
        {
            return Some(ConvergedReason::Rtol);
        }

        let atol_reached = projected_grad_norm < self.opts.bound_opts.grad_atol;

        if atol_reached
        {
            return Some(ConvergedReason::Atol);
        }

        None
    }

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
        a.clamp(self.opts.alpha_min, self.opts.alpha_max);
        a
    }

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
            norm_grad_fx,
        } = iter_k;

        let d = self.bounds.projected_direction(x, grad_fx, alpha_init);

        if d.abs_max().unwrap() < SMALL
        {
            return None;
        }

        let f_max = self.fn_history.max().unwrap();
        let delta = self.opts.delta;
        let mut alpha = 1.0;
        let mut x_trial: Vector = (x + &d).into();
        let mut f_trial = self.fcn.eval(&x_trial);
        let gradfk_dot_d = grad_fx.dot(&d);

        while f_trial > f_max + delta * alpha * gradfk_dot_d && alpha > SMALL
        {
            alpha *= delta;
            x_trial = (x + alpha * &d).into();
            f_trial = self.fcn.eval(x)
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
                return true;
            }
        }
        false
    }

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
            all_equal = all_equal && (*sig != most_recent_sig);
        }
        return all_equal;
    }
}
//}}}
//{{{ impl: BoundConstrainedMinimizer for ActiveSetAlgorithm
impl<F: RealFn> BoundConstrainedMinimizer for ActiveSetAlgorithm<F>
{
    fn minimize(&mut self) -> Result<crate::Returns, super::common::Error>
    {
        let n = self.fcn.dimension();
        let mut iter_k_prev = IterData::new(self.fcn.clone(), &self.x_init);
        let mut iter_k = IterData::new(self.fcn.clone(), &self.x_init);
        let mut phase = Phase::NGPA;
        let mut mu = self.opts.mu;
        let mut alpha_bb = 1.0;

        for i in 0..self.opts.bound_opts.max_iter
        {
            if let Some(reason) = self.is_converged(&iter_k)
            {
                return Ok(crate::Returns {
                    xmin: iter_k.x,
                    fmin: iter_k.fx,
                    reason: reason,
                    num_iterations: 0,
                    num_fun_evals: 0,
                    num_grad_evals: 0,
                });
            }

            match phase
            {
                Phase::NGPA =>
                {
                    iter_k_prev.copy_from(&iter_k);

                    let IterData {
                        x: x_prev,
                        fx: fx_prev,
                        grad_fx: grad_fx_prev,
                        norm_grad_fx: norm_grad_fx_prev,
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
                        norm_grad_fx,
                    } = &iter_k;

                    self.fn_history.append(*fx);
                    self.active_signature_history
                        .append(self.bounds.active_signature(x));

                    let s = (x - x_prev).into();
                    let y = (grad_fx - grad_fx_prev).into();
                    alpha_bb = self.bb_step(&s, &y, alpha_bb);

                    let projected_grad = self.bounds.projected_direction(&x, &grad_fx, 1.0);
                    let projected_grad_norm = projected_grad.norm();
                    let inactive_grad = self.bounds.masked_gradient(&x, &grad_fx);
                    let inactive_grad_norm = inactive_grad.norm();

                    if self.undecided_set_is_empy(&x, &inactive_grad, projected_grad_norm)
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
                    iter_k_prev.copy_from(&iter_k);

                    let IterData {
                        x,
                        fx,
                        grad_fx,
                        norm_grad_fx,
                    } = &iter_k;

                    let (_, inactive_set) = self.bounds.active_and_inactive_sets(&x, &grad_fx);

                    let mut restricted_fcn =
                        RestrictedFunction::new(self.fcn.clone(), inactive_set);
                    let x0 = restricted_fcn.restrict(x);
                    let res = minimize(
                        restricted_fcn.clone(),
                        x0,
                        self.opts.unconstrained_method.clone(),
                    )?;
                    iter_k.x.copy_from(restricted_fcn.lift(&res.xmin));
                    iter_k.fx = res.fmin;
                    iter_k.grad_fx.copy_from(self.fcn.grad(&iter_k.x));

                    let projected_grad_new =
                        self.bounds
                            .projected_direction(&iter_k.x, &iter_k.grad_fx, 1.0);
                }
            }
        }
        todo!()
    }
}
//}}}
