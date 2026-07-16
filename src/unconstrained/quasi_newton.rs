//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::{Error, Options as UnconstrainedOptions};
use crate::common::Minimizer;
use crate::common::Vector;
use crate::line_search as ls;
use crate::line_search::LineSearchMethod;
use crate::quadratic_model::{QuadraticModel, UpdateType::Inverse};
use crate::{ConvergedReason, IterData, RealFn, VectorReturns};
//}}}
//{{{ std imports
#[allow(unused_imports)]
use topohedral_linalg::MatrixOps;
use topohedral_linalg::{MatMul, ReduceOps, VectorOps};
//}}}
//{{{ dep imports
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum: UpdateMethod
#[derive(Copy, Clone)]
pub enum UpdateMethod {
    BFGS,
    DFP,
}
//}}}
//{{{ struct: Options
#[derive(Clone)]
pub struct Options {
    pub uncon_opts: UnconstrainedOptions,
    pub ls_method: LineSearchMethod,
    pub method: UpdateMethod,
    pub restart: u64,
}
//}}}
//{{{ struct: QuasiNewton
pub struct QuasiNewton<F: RealFn> {
    fcn: F,
    x_init: Vector,
    norm_grad_fx_init: f64,
    opts: Options,
    quadratic_model: QuadraticModel,
}
//}}}
//{{{ impl: QuasiNewton
impl<F: RealFn> QuasiNewton<F> {
    #[trace_fn]
    pub fn new(
        mut fcn: F,
        x0: Vector,
        opts: Options,
    ) -> Self {
        let grad_0 = fcn.derivative(&x0);
        let norm_grad_0 = grad_0.abs_max().unwrap_or(0.0);
        let n = x0.len();
        Self {
            fcn,
            x_init: x0,
            norm_grad_fx_init: norm_grad_0,
            opts,
            quadratic_model: QuadraticModel::new(n),
        }
    }

    #[trace_fn]
    fn apply_restart(
        &self,
        k: u64,
        grad_fk: &Vector,
        dir_k: &mut Vector,
    ) {
        let needs_restart = k.is_multiple_of(self.opts.restart);
        let is_increasing = grad_fk.dot(dir_k) >= 0.0;
        if needs_restart || is_increasing {
            *dir_k = -grad_fk.clone();
        }
    }

    #[trace_fn]
    fn is_converged(
        &self,
        grad_norm: f64,
    ) -> Option<ConvergedReason> {
        let rtol = self.opts.uncon_opts.grad_rtol;
        let rtol_converged = (grad_norm / self.norm_grad_fx_init) < rtol;
        if rtol_converged {
            return Some(ConvergedReason::Rtol);
        }
        let atol_converged = grad_norm < self.opts.uncon_opts.grad_atol;
        if atol_converged {
            return Some(ConvergedReason::Atol);
        }
        None
    }

    #[trace_fn]
    fn update_hessian(
        &mut self,
        xk_prev: &Vector,
        xk: &Vector,
        grad_fk_prev: &Vector,
        grad_fk: &Vector,
    ) -> bool {
        match self.opts.method {
            UpdateMethod::BFGS => {
                let sk: Vector = (xk - xk_prev).into();
                let yk: Vector = (grad_fk - grad_fk_prev).into();
                self.quadratic_model.try_update(&sk, &yk, Inverse)
            }
            UpdateMethod::DFP => {
                todo!()
            }
        }
    }

    #[trace_fn]
    fn update_direction(
        &mut self,
        xk_prev: &Vector,
        xk: &Vector,
        grad_fk_prev: &Vector,
        grad_fk: &Vector,
    ) -> Vector {
        if !self.update_hessian(xk_prev, xk, grad_fk_prev, grad_fk) {
            self.quadratic_model.reset();
            return -grad_fk.clone();
        }
        -self.quadratic_model.inv_hess_k.matmul(grad_fk)
    }

    fn print_status(
        &self,
        _k: u64,
        current_iter: &IterData,
    ) {
        //{{{ trace
        info!(target: "qn", "======================================================================== i = {_k}");
        trace!(target: "qn", "Current solution: {}", current_iter.x.clone().transpose());
        info!(target: "qn", "Current values: {current_iter}");
        info!(target: "qn","Convergence measures:");
        let stationarity = current_iter.grad_fx.abs_max().unwrap_or(0.0);
        let _grad_ratio = stationarity / self.norm_grad_fx_init;
        info!(target: "qn", "||∇f(k)|| / ||∇f(0)|| = {_grad_ratio:.4e}");
        //}}}
    }
}
//}}}
//{{{ impl: Minimizer for QuasiNewton
impl<F: RealFn> Minimizer for QuasiNewton<F> {
    type Error = Error;
    type Returns = VectorReturns;

    #[trace_fn]
    fn minimize(&mut self) -> Result<VectorReturns, Self::Error> {
        let mut iter_k = IterData::new(self.fcn.clone(), &self.x_init);
        let mut iter_prev_k = iter_k.clone();
        iter_prev_k.fx = iter_k.fx + 0.5 * iter_k.norm_grad_fx;

        let mut dir_k = -iter_k.grad_fx.clone();
        let max_iter = self.opts.uncon_opts.max_iter;

        self.quadratic_model
            .update_iterate(&iter_k.x, iter_k.fx, &iter_k.grad_fx);

        for k in 1..max_iter {
            self.print_status(k, &iter_k);
            self.apply_restart(k, &iter_k.grad_fx, &mut dir_k);

            let alpha_init =
                ls::initial_step(iter_k.fx, iter_prev_k.fx, iter_k.grad_fx.dot(&dir_k));

            let fx_prev = iter_prev_k.fx;
            iter_prev_k = iter_k;

            let search_result = ls::search(
                self.fcn.clone(),
                &iter_prev_k,
                &dir_k,
                alpha_init,
                self.opts.ls_method.clone(),
            );

            iter_k = match search_result {
                Ok(iter) => iter,
                Err(_) => {
                    self.quadratic_model.reset();
                    dir_k = -iter_prev_k.grad_fx.clone();
                    let alpha_init =
                        ls::initial_step(iter_prev_k.fx, fx_prev, iter_prev_k.grad_fx.dot(&dir_k));
                    ls::search(
                        self.fcn.clone(),
                        &iter_prev_k,
                        &dir_k,
                        alpha_init,
                        self.opts.ls_method.clone(),
                    )?
                }
            };

            dir_k = self.update_direction(
                &iter_prev_k.x,
                &iter_k.x,
                &iter_prev_k.grad_fx,
                &iter_k.grad_fx,
            );
            self.quadratic_model
                .update_iterate(&iter_k.x, iter_k.fx, &iter_k.grad_fx);

            let stationarity = iter_k.grad_fx.abs_max().unwrap_or(0.0);
            if let Some(reason) = self.is_converged(stationarity) {
                //{{{ trace
                info!(target: "qn", "=============================================");
                info!(target: "qn", "Converging with reason {reason:?}");
                info!(target: "qn","Convergence measures:");
                let _grad_ratio = stationarity / self.norm_grad_fx_init;
                info!(target: "qn", "||∇f(k)|| / ||∇f(0)|| = {_grad_ratio:.4e}");
                info!(target: "qn", "||∇f(k)|| = {:.4e}", stationarity);
                trace!(target: "qn", "fx = {:.4e} x = {}", iter_k.fx, iter_k.x.clone().transpose());
                info!(target: "qn", "=============================================");
                //}}}
                return Ok(VectorReturns {
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
        let _maxiter = self.opts.uncon_opts.max_iter;
        info!(target: "qn", "Did not converge within {_maxiter} iterations");
        //}}}
        Err(Error::MaxIterations(self.opts.uncon_opts.max_iter as usize))
    }
}
//}}}
