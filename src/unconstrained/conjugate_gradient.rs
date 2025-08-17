//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------


//{{{ crate imports
use super::common::Options as UnonstrainedOptions;
use super::common::{Error, Returns, UnconstrainedMinimizer};
use crate::common::{arc_real_fn, CountingRealFn};
use crate::line_search as ls;
use crate::line_search::LineSearch;
use crate::line_search::LineSearchFcn;
use crate::unconstrained::common::ConvergedReason;
use crate::RealFn;
//}}}
//{{{ std imports
use std::ops::{Add, Mul, Neg, Sub};
use std::fmt;
use std::sync::{Arc, Mutex};
//}}}
//{{{ dep imports
use topohedral_linalg::VectorOps;
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

#[derive(Copy, Clone)]
pub enum Direction {
    Steepest,
    FletcherReeves,
    PolakRibiere,
}

#[derive(Copy, Clone)]
pub struct Options {
    pub uncon_opts: UnonstrainedOptions,
    pub direction: Direction,
    pub restart: u64,
}

pub struct ConjugateGradient<F: RealFn> {
    fcn: Arc<Mutex<CountingRealFn<F>>>,
    x_init: F::Vector,
    grad_fx_init: F::Vector,
    opts: Options,
}

impl<F: RealFn> ConjugateGradient<F>
where
    F::Vector: VectorOps<ScalarType = f64>
        + Add<Output = F::Vector>
        + Sub<Output = F::Vector>
        + Clone
        + fmt::Display,
    f64: Mul<F::Vector, Output = F::Vector>,
{
    pub fn new(mut fcn: F, x0: F::Vector, opts: Options) -> Self {
        let grad_0 = fcn.grad(&x0);
        let fcn_shared = arc_real_fn(CountingRealFn::new(fcn));
        Self {
            fcn: fcn_shared.clone(),
            x_init: x0.clone(),
            grad_fx_init: grad_0,
            opts: opts,
        }
    }

    /// Updates the search direction for the conjugate gradient method based on the
    /// current and previous gradients, and the current search direction.
    /// The update formula used depends on the `DirectionMethod` specified in the
    /// `Opts` struct.
    fn update_direction(
        &self,
        grad_fk1: &F::Vector,
        grad_fk: &F::Vector,
        norm_grad_fk1: f64,
        norm_grad_fk: f64,
        dir_k: &F::Vector,
    ) -> F::Vector {
        //{{{ trace
        debug!(target: "cg", "\t--- Entering update_direction ---");
        trace!(target: "cg", "\t\n\ngrad_fk1 = \n{grad_fk1}\n\ngrad_fk = \n{grad_fk}\n\n");
        trace!(target: "cg", "\tnorm_grad_fk1 = {norm_grad_fk1:1.4e} norm_grad_fk = {norm_grad_fk:1.4e}");
        //}}}
        // direction updates
        let beta = match self.opts.direction {
            Direction::Steepest => {
                //{{{ trace
                debug!("Applying Steepest Descent update");
                //}}}
                0.0
            }
            Direction::FletcherReeves => {
                //{{{ trace
                debug!(target: "cg", "Applying fletcher-reeves update");
                //}}}
                let beta_tmp = grad_fk.dot(&grad_fk1) / norm_grad_fk1.powi(2);
                beta_tmp
            }
            Direction::PolakRibiere => {
                //{{{ trace
                debug!(target: "cg", "Applying polak-ribiere update");
                //}}}
                let yk = grad_fk.clone() - grad_fk1.clone();
                let mut beta_tmp = grad_fk.dot(&yk) / norm_grad_fk1.powi(2);
                beta_tmp = beta_tmp.max(0.0);
                beta_tmp
            }
        };

        let new_dir_k = beta * dir_k.clone() - grad_fk.clone();
        //{{{ trace
        debug!(target: "cg", "beta = {:1.4e}", beta);
        debug!(target: "cg", "--- Leaving update_direction ---");
        //}}}
        new_dir_k
    }

    fn is_converged(&self, grad_norm: f64) -> Option<ConvergedReason> {
        let rtol_converged = grad_norm < self.opts.uncon_opts.grad_rtol;
        if rtol_converged {
            return Some(ConvergedReason::Rtol);
        }
        let atol_converged = grad_norm < self.opts.uncon_opts.grad_atol;
        if atol_converged {
            return Some(ConvergedReason::Atol);
        }
        None
    }
}

impl<F: RealFn> UnconstrainedMinimizer for ConjugateGradient<F>
where
    F::Vector: VectorOps<ScalarType = f64>
        + Add<Output = F::Vector>
        + Sub<Output = F::Vector>
        + Neg<Output = F::Vector>
        + Clone
        + fmt::Display,
    f64: Mul<F::Vector, Output = F::Vector>,
{
    type Vector = F::Vector;

    fn minimize(&mut self) -> Result<Returns<Self::Vector>, Error> {
        info!(target: "cg", "--- Entering minimize() ---");
        let mut xk = self.x_init.clone();
        let mut xk_prev = self.x_init.clone();
        let mut fk = self.fcn.eval(&xk);
        let mut fk_prev = fk;
        let mut grad_fk = self.fcn.grad(&xk);
        let mut grad_fk_prev = grad_fk.clone();
        let mut grad_fk_prev_norm = grad_fk_prev.norm();
        let mut grad_fk_norm = grad_fk.norm();
        let mut direction = -grad_fk.clone();

        //{{{ trace
        info!(target: "cg", "Initial values upon entry: ");
        info!(target: "cg", "f0 = {fk:1.4e} norm_f0 = {grad_fk_norm:1.4e}");
        trace!(target: "cg", "\n\nx0 = \n{xk}\n\ngrad_f0 = \n{grad_fk}\n\n");
        //}}}

        let max_iter = self.opts.uncon_opts.max_iter;

        let mut line_searcher = ls::create(
                LineSearchFcn::new(self.fcn.clone(), self.x_init.clone(), self.grad_fx_init.clone()),
                self.opts.uncon_opts.ls_method,
        );

        let grad_fx_norm_init = self.grad_fx_init.norm();

        for i in 1..max_iter {
            //{{{ trace
            info!(target: "cg", "======================================================================== i = {i}");
            info!(target: "cg", "Current values fk = {fk:1.4e} grad_fk_norm = {grad_fk_norm:1.4e}");
            info!(target: "cg","Convergence measures:");
            info!(target: "cg", "\t||∇f(k)|| / ||∇f(0)|| = {:1.4e} ", grad_fk_norm / grad_fx_norm_init);
            info!(target: "cg", "\t||x(k) - x(k-1)|| = {:1.4e}", (xk.clone() - xk_prev.clone()).norm());
            trace!(target: "cg", "\n\nxk = \n{xk}\n\ndir = \n{direction}\n\n");
            //}}}
            let phi0 = fk;
            let mut dphi0 = grad_fk.dot(&direction);
            let needs_restart = i % self.opts.restart == 0;
            let not_decreaseing = dphi0 >= 0.0;
            if needs_restart || not_decreaseing {
                info!(target: "cg", "\tDoing restart for reasons:  restart? {needs_restart} descent direction? {not_decreaseing}");
                direction = -grad_fk.clone();
                dphi0 = grad_fk.dot(&direction);
            }

            let line_search_fcn =
                LineSearchFcn::new(self.fcn.clone(), xk.clone(), direction.clone());
            line_searcher.update_fcn(line_search_fcn);

            let ls_ret = line_searcher.search(phi0, dphi0)?;
            xk_prev = xk.clone();
            xk = xk + ls_ret.alpha * direction.clone();
            fk_prev = fk;
            fk = ls_ret.phi_alpha;
            grad_fk_prev = grad_fk.clone();
            grad_fk = self.fcn.grad(&xk);
            grad_fk_prev_norm = grad_fk_prev.norm();
            grad_fk_norm = grad_fk.norm();

            if let Some(reason) = self.is_converged(grad_fk_norm) {
                //{{{ trace
                info!(target: "cg", "Converging with reason {reason:?}");
                info!(target: "cg", "--- Leaving minimize() ---");
                //}}}

                let fcn_lock = self.fcn.lock().unwrap();
                return Ok(Returns {
                    fmin: fk,
                    xmin: xk,
                    reason: reason,
                    num_iterations: i as usize,
                    num_fun_evals: fcn_lock.num_func_evals,
                    num_grad_evals: fcn_lock.num_grad_evals,
                });
            }

            direction = self.update_direction(
                &grad_fk_prev,
                &grad_fk,
                grad_fk_prev_norm,
                grad_fk_norm,
                &direction,
            );

            //{{{ trace
            trace!(target: "cg", "fk_prev = {fk_prev:1.4e}, fk = {fk:1.4e}");
            trace!(target: "cg", "grad_fk_prev_norm = {grad_fk_prev_norm:1.4e}, fk = {grad_fk_norm:1.4e}");
            //}}}
        }
        //{{{ trace
        let maxiter = self.opts.uncon_opts.max_iter;
        info!("Did not converge within {maxiter} iterations");
        info!(target: "cg", "--- Leaving minimize() ---");
        //}}}
        Err(Error::MaxIterations(self.opts.uncon_opts.max_iter as usize))
    }
}
