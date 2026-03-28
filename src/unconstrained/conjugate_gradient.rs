//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::Options as UnonstrainedOptions;
use super::common::{Error, Returns, UnconstrainedMinimizer};
use crate::common::{arc_real_fn, CountingRealFn, IterData};
use crate::line_search as ls;
use crate::unconstrained::common::ConvergedReason;
use crate::{RealFn, Vector};
//}}}
//{{{ std imports
use serde_json::map::Iter;
use std::sync::{Arc, Mutex};
//}}}
//{{{ dep imports
use topohedral_linalg::VectorOps;
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum: Direction
#[derive(Copy, Clone)]
pub enum Direction
{
    Steepest,
    FletcherReeves,
    PolakRibiere,
}
//}}}
//{{{ struct: Options
#[derive(Copy, Clone)]
pub struct Options
{
    pub uncon_opts: UnonstrainedOptions,
    pub direction: Direction,
    pub restart: u64,
}
//}}}
//{{{ struct: ConjugateGradient
pub struct ConjugateGradient<F: RealFn>
{
    fcn: F,
    x_init: Vector,
    grad_fx_init: Vector,
    norm_grad_fx_init: f64,
    opts: Options,
}
//}}}
//{{{ impl: ConjugateGradient
impl<F: RealFn> ConjugateGradient<F>
{
    #[trace_fn]
    pub fn new(
        mut fcn: F,
        x0: Vector,
        opts: Options,
    ) -> Self
    {
        let grad_0 = fcn.grad(&x0);
        let norm_grad_0 = grad_0.norm();
        Self {
            fcn: fcn,
            x_init: x0.clone(),
            grad_fx_init: grad_0,
            norm_grad_fx_init: norm_grad_0,
            opts,
        }
    }

    /// Updates the search direction for the conjugate gradient method based on the
    /// current and previous gradients, and the current search direction.
    /// The update formula used depends on the `DirectionMethod` specified in the
    /// `Opts` struct.
    #[trace_fn]
    fn update_direction(
        &self,
        k: u64,
        grad_fk_prev: &Vector,
        grad_fk: &Vector,
        norm_grad_fk_prev: f64,
        norm_grad_fk: f64,
        dir_k: &Vector,
    ) -> Vector
    {
        //{{{ trace
        trace!(target: "cg", "norm_grad_fk1 = {norm_grad_fk_prev:1.4e} norm_grad_fk = {norm_grad_fk:1.4e}");
        //}}}
        let is_increasing = grad_fk.dot(dir_k) >= 0.0;
        if self.needs_restart(k) || is_increasing
        {
            return -grad_fk.clone();
        }

        // direction updates
        let beta = match self.opts.direction
        {
            Direction::Steepest =>
            {
                //{{{ trace
                debug!("Applying Steepest Descent update");
                //}}}
                0.0
            }
            Direction::FletcherReeves =>
            {
                //{{{ trace
                debug!(target: "cg", "Applying fletcher-reeves update");
                //}}}

                grad_fk.dot(grad_fk_prev) / norm_grad_fk_prev.powi(2)
            }
            Direction::PolakRibiere =>
            {
                //{{{ trace
                debug!(target: "cg", "Applying polak-ribiere update");
                //}}}
                let yk = grad_fk.clone() - grad_fk_prev.clone();
                let mut beta_tmp = grad_fk.dot(&yk) / norm_grad_fk_prev.powi(2);
                beta_tmp = beta_tmp.max(0.0);
                beta_tmp
            }
        };

        let new_dir_k = beta * dir_k.clone() - grad_fk.clone();
        //{{{ trace
        debug!(target: "cg", "beta = {:1.4e}", beta);
        //}}}
        new_dir_k
    }

    #[trace_fn]
    fn is_converged(
        &self,
        grad_norm: f64,
        grad_norm_init: f64,
    ) -> Option<ConvergedReason>
    {
        let rtol = self.opts.uncon_opts.grad_rtol;
        let rtol_converged = (grad_norm / grad_norm_init) < rtol;
        if rtol_converged
        {
            return Some(ConvergedReason::Rtol);
        }
        let atol_converged = grad_norm < self.opts.uncon_opts.grad_atol;
        if atol_converged
        {
            return Some(ConvergedReason::Atol);
        }
        None
    }

    #[trace_fn]
    fn needs_restart(
        &self,
        iter: u64,
    ) -> bool
    {
        iter % self.opts.restart == 0
    }

    fn print_status(
        &self,
        k: u64,
        current_iter: &IterData,
    )
    {
        //{{{ trace
        info!(target: "cg", "======================================================================== i = {k}");
        info!(target: "cg", "Current values: {current_iter}");
        info!(target: "cg","Convergence measures:");
        let grad_ratio = current_iter.norm_grad_fx / self.norm_grad_fx_init;
        info!(target: "cg", "\t||∇f(k)|| / ||∇f(0)|| = {grad_ratio:1.4e}");
        //}}}
    }
}
//}}}
//{{{ impl: UnconstrainedMinimizer for ConjugateGradient
impl<F: RealFn> UnconstrainedMinimizer for ConjugateGradient<F>
{
    #[trace_fn]
    fn minimize(&mut self) -> Result<Returns, Error>
    {
        let mut iter_k = IterData::new(self.fcn.clone(), &self.x_init);
        let mut iter_k_prev: IterData;
        let mut direction = -iter_k.grad_fx.clone();
        let mut alpha_init =
            ls::initial_step(iter_k.fx, iter_k.fx + 0.5 * iter_k.norm_grad_fx, -1.0);
        let max_iter = self.opts.uncon_opts.max_iter;
        let grad_fx_norm_init = self.grad_fx_init.norm();

        for k in 1..max_iter
        {
            self.print_status(k, &iter_k);

            iter_k_prev = iter_k;
            iter_k = ls::search(
                self.fcn.clone(),
                &iter_k_prev,
                &direction,
                alpha_init,
                self.opts.uncon_opts.ls_method,
            )?;
            direction = self.update_direction(
                k,
                &iter_k_prev.grad_fx,
                &iter_k.grad_fx,
                iter_k_prev.norm_grad_fx,
                iter_k.norm_grad_fx,
                &direction,
            );
            alpha_init =
                ls::initial_step(iter_k.fx, iter_k_prev.fx, iter_k.grad_fx.dot(&direction));

            if let Some(reason) = self.is_converged(iter_k.norm_grad_fx, grad_fx_norm_init)
            {
                //{{{ trace
                info!(target: "cg", "Converging with reason {reason:?}");
                //}}}
                return Ok(Returns {
                    fmin: iter_k.fx,
                    xmin: iter_k.x,
                    reason,
                    num_iterations: k as usize,
                    num_fun_evals: 0,
                    num_grad_evals: 0,
                });
            }
        }
        //{{{ trace
        info!(target: "cg", "Did not converge within {max_iter} iterations");
        //}}}
        Err(Error::MaxIterations(self.opts.uncon_opts.max_iter as usize))
    }
}
//}}}
