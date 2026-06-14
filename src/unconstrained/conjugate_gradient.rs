//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::Error;
use crate::common::BaseOptions;
use crate::line_search::{self as ls, LineSearchMethod};
use crate::{ConvergedReason, IterData, Minimizer, RealFn, Returns, Vector};
//}}}
//{{{ std imports
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
#[derive(Clone)]
pub struct Options
{
    pub uncon_opts: BaseOptions,
    pub ls_method: LineSearchMethod,
    pub direction: Direction,
    pub restart: u64,
}
//}}}
//{{{ struct: ConjugateGradient
pub struct ConjugateGradient<F: RealFn>
{
    fcn: F,
    x_init: Vector,
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
            fcn,
            x_init: x0.clone(),
            norm_grad_fx_init: norm_grad_0,
            opts,
        }
    }

    #[trace_fn]
    fn apply_restart(
        &self,
        k: u64,
        grad_fk: &Vector,
        dir_k: &mut Vector,
    )
    {
        let needs_restart = k.is_multiple_of(self.opts.restart);
        let is_increasing = grad_fk.dot(dir_k) >= 0.0;
        if needs_restart || is_increasing
        {
            *dir_k = -grad_fk.clone();
        }
    }

    /// Updates the search direction for the conjugate gradient method based on the
    /// current and previous gradients, and the current search direction.
    /// The update formula used depends on the `DirectionMethod` specified in the
    /// `Opts` struct.
    #[trace_fn]
    fn update_direction(
        &self,
        _k: u64,
        grad_fk_prev: &Vector,
        grad_fk: &Vector,
        norm_grad_fk_prev: f64,
        norm_grad_fk: f64,
        dir_k: &Vector,
    ) -> Vector
    {
        //{{{ trace
        trace!(target: "cg", "norm_grad_fk1 = {norm_grad_fk_prev:.4e} norm_grad_fk = {norm_grad_fk:.4e}");
        //}}}

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

                (norm_grad_fk.powi(2) / norm_grad_fk_prev.powi(2)).max(0.0)
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
        debug!(target: "cg", "beta = {:.4e}", beta);
        //}}}
        new_dir_k
    }

    #[trace_fn]
    fn is_converged(
        &self,
        grad_norm: f64,
    ) -> Option<ConvergedReason>
    {
        let rtol = self.opts.uncon_opts.grad_rtol;
        let rtol_converged = (grad_norm / self.norm_grad_fx_init) < rtol;
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

    fn print_status(
        &self,
        _k: u64,
        current_iter: &IterData,
    )
    {
        //{{{ trace
        info!(target: "cg", "======================================================================== i = {_k}");
        trace!(target: "aug", "Current solution: {}", current_iter.x.clone().transpose());
        info!(target: "cg", "Current values: {current_iter}");
        info!(target: "cg","Convergence measures:");
        let _grad_ratio = current_iter.norm_grad_fx / self.norm_grad_fx_init;
        info!(target: "cg", "||∇f(k)|| / ||∇f(0)|| = {_grad_ratio:.4e}");
        //}}}
    }
}
//}}}
//{{{ impl: Minimizer for ConjugateGradient
impl<F: RealFn> Minimizer for ConjugateGradient<F>
{
    type Error = Error;

    #[trace_fn]
    fn minimize(&mut self) -> Result<Returns, Self::Error>
    {
        let mut iter_k = IterData::new(self.fcn.clone(), &self.x_init);

        let mut iter_k_prev = iter_k.clone();
        iter_k_prev.fx = iter_k.fx + 0.5 * iter_k.norm_grad_fx;

        let mut dir_k = -iter_k.grad_fx.clone();
        let max_iter = self.opts.uncon_opts.max_iter;

        for k in 1..max_iter
        {
            self.print_status(k, &iter_k);
            self.apply_restart(k, &iter_k.grad_fx, &mut dir_k);

            let alpha_init =
                ls::initial_step(iter_k.fx, iter_k_prev.fx, iter_k.grad_fx.dot(&dir_k));

            iter_k_prev = iter_k;

            iter_k = ls::search(
                self.fcn.clone(),
                &iter_k_prev,
                &dir_k,
                alpha_init,
                self.opts.ls_method.clone(),
            )?;

            dir_k = self.update_direction(
                k,
                &iter_k_prev.grad_fx,
                &iter_k.grad_fx,
                iter_k_prev.norm_grad_fx,
                iter_k.norm_grad_fx,
                &dir_k,
            );

            if let Some(reason) = self.is_converged(iter_k.norm_grad_fx)
            {
                //{{{ trace
                info!(target: "cg", "=============================================");
                info!(target: "cg", "Converging with reason {reason:?}");
                info!(target: "cg","Convergence measures:");
                let _grad_ratio = iter_k.norm_grad_fx / self.norm_grad_fx_init;
                info!(target: "cg", "||∇f(k)|| / ||∇f(0)|| = {_grad_ratio:.4e}");
                info!(target: "cg", "||∇f(k)|| = {:.4e}", iter_k.norm_grad_fx);
                trace!(target: "cg", "fx = {:.4e} x = {}", iter_k.fx, iter_k.x.clone().transpose());
                info!(target: "cg", "=============================================");
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
