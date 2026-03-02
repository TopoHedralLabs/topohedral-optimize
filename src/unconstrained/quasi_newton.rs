//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::Options as UnonstrainedOptions;
use super::common::{ConvergedReason, Error, Returns};
use crate::line_search as ls;
use crate::line_search::initial_step;
use crate::line_search::LineSearchFcn;
use crate::unconstrained::UnconstrainedMinimizer;
use crate::{common::arc_real_fn, common::CountingRealFn, RealFn};
//}}}
//{{{ std imports
use std::sync::{Arc, Mutex};
use topohedral_linalg::MatrixOps;
//}}}
//{{{ dep imports
use topohedral_linalg::{dmatrix::DMatrix, dvector::DVector, dvector::VecType, MatMul, VectorOps};
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

#[derive(Copy, Clone)]
pub enum UpdateMethod
{
    BFGS,
    DFP,
}

#[derive(Copy, Clone)]
pub struct Options
{
    pub uncon_opts: UnonstrainedOptions,
    pub method: UpdateMethod,
    pub restart: u64,
}

struct Data
{
    identity: DMatrix<f64>,
    mat1: DMatrix<f64>,
    mat2: DMatrix<f64>,
    mat3: DMatrix<f64>,
    sk: DVector<f64>,
    yk: DVector<f64>,
}

pub struct QuasiNewton<F: RealFn>
{
    fcn: Arc<Mutex<CountingRealFn<F>>>,
    x_init: DVector<f64>,
    grad_fx_init: DVector<f64>,
    opts: Options,
    data: Data,
}

impl<F: RealFn> QuasiNewton<F>
{
    #[trace_fn]
    pub fn new(
        mut fcn: F,
        x0: DVector<f64>,
        opts: Options,
    ) -> Self
    {
        let grad_0 = fcn.grad(&x0);
        let fcn_shared = arc_real_fn(CountingRealFn::new(fcn));
        Self {
            fcn: fcn_shared.clone(),
            x_init: x0.clone(),
            grad_fx_init: grad_0,
            opts,
            data: Data {
                identity: DMatrix::<f64>::identity(x0.len(), x0.len()),
                mat1: DMatrix::<f64>::zeros(x0.len(), x0.len()),
                mat2: DMatrix::<f64>::zeros(x0.len(), x0.len()),
                mat3: DMatrix::<f64>::zeros(x0.len(), x0.len()),
                sk: DVector::zeros_cvec(x0.len(), VecType::Col),
                yk: DVector::zeros_cvec(x0.len(), VecType::Col),
            },
        }
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
    fn update_hessian(
        &mut self,
        xk_prev: DVector<f64>,
        xk: DVector<f64>,
        grad_fk_prev: DVector<f64>,
        grad_fk: DVector<f64>,
        hess_k: &mut DMatrix<f64>,
    )
    {
        match self.opts.method
        {
            UpdateMethod::BFGS =>
            {
                let Data {
                    identity,
                    mat1,
                    mat2,
                    mat3,
                    sk,
                    yk,
                } = &mut self.data;

                *sk = xk - xk_prev;

                *yk = grad_fk - grad_fk_prev;

                let rho_k = 1.0 / (sk.dot(yk));

                *mat1 = (&*identity - rho_k * &sk.matmul(yk.transpose())).into();

                *mat2 = (&*identity - rho_k * &yk.matmul(sk.transpose())).into();

                *mat3 = rho_k * sk.matmul(sk.transpose());

                *hess_k = (&mat1.matmul(hess_k.clone().matmul(mat2)) + &*mat3).into();
            }
            UpdateMethod::DFP =>
            {
                todo!()
            }
        }
    }
}

impl<F: RealFn> UnconstrainedMinimizer for QuasiNewton<F>
{
    #[trace_fn]
    fn minimize(&mut self) -> Result<Returns, Error>
    {
        let xk = self.x_init.clone();
        let mut xk_prev = self.x_init.clone();
        let mut grad_fk = self.fcn.grad(&xk);
        let mut grad_fk_prev: DVector<f64>;
        let mut grad_fk_norm: f64 = grad_fk.norm();
        let mut fk: f64 = self.fcn.eval(&xk);
        let fk_prev_offset: f64 = 0.5 * grad_fk_norm;
        let mut fk_prev = fk + fk_prev_offset;
        let mut direction = -grad_fk.clone();
        let mut xk = self.x_init.clone();

        let n = xk.len();

        let mut hess_k = DMatrix::<f64>::identity(n, n);

        let max_iter = self.opts.uncon_opts.max_iter;

        let mut line_searcher = ls::create(
            LineSearchFcn::new(
                self.fcn.clone(),
                self.x_init.clone(),
                self.grad_fx_init.clone(),
            ),
            self.opts.uncon_opts.ls_method,
        );

        let grad_fx_norm_init = self.grad_fx_init.norm();

        for i in 1..max_iter
        {
            //{{{ trace
            info!(target: "qn", "======================================================================== i = {i}");
            info!(target: "qn", "Current values fk = {fk:1.4e} grad_fk_norm = {grad_fk_norm:1.4e}");
            info!(target: "qn","Convergence measures:");
            info!(target: "qn", "\t||∇f(k)|| / ||∇f(0)|| = {:1.4e} ", grad_fk_norm / grad_fx_norm_init);
            info!(target: "qn", "\t||x(k) - x(k-1)|| = {:1.4e}", (xk.clone() - xk_prev.clone()).norm());
            //}}}
            let mut dphi0 = grad_fk.dot(&direction);
            let needs_restart = i % self.opts.restart == 0;
            let not_decreaseing = dphi0 >= 0.0;
            if needs_restart || not_decreaseing
            {
                //{{{ trace
                info!(target: "qn", "\tDoing restart for reasons:  restart? {needs_restart} descent direction? {not_decreaseing}");
                //}}}
                direction = -grad_fk.clone();
                dphi0 = grad_fk.dot(&direction);
            }

            let line_search_fcn =
                LineSearchFcn::new(self.fcn.clone(), xk.clone(), direction.clone());

            line_searcher.update_fcn(line_search_fcn);

            let phi0 = fk;
            let old_phi0 = fk_prev;

            let alpha_init: f64 = initial_step(phi0, old_phi0, dphi0);
            let ls_ret = line_searcher.search(phi0, dphi0, alpha_init)?;

            xk_prev = xk.clone();
            xk += ls_ret.alpha * direction.clone();

            fk_prev = fk;
            fk = ls_ret.phi_alpha;

            grad_fk_prev = grad_fk.clone();
            grad_fk = self.fcn.grad(&xk);
            grad_fk_norm = grad_fk.norm();

            if let Some(reason) = self.is_converged(grad_fk_norm, grad_fx_norm_init)
            {
                //{{{ trace
                info!(target: "qn", "Converging with reason {reason:?}");
                //}}}

                let fcn_lock = self.fcn.lock().unwrap();
                return Ok(Returns {
                    fmin: fk,
                    xmin: xk,
                    reason,
                    num_iterations: i as usize,
                    num_fun_evals: fcn_lock.num_func_evals,
                    num_grad_evals: fcn_lock.num_grad_evals,
                });
            }

            self.update_hessian(
                xk_prev.clone(),
                xk.clone(),
                grad_fk_prev.clone(),
                grad_fk.clone(),
                &mut hess_k,
            );

            direction = -hess_k.matmul(&grad_fk);
        }
        //{{{ trace
        let maxiter = self.opts.uncon_opts.max_iter;
        info!(target: "qn", "Did not converge within {maxiter} iterations");
        //}}}
        Err(Error::MaxIterations(self.opts.uncon_opts.max_iter as usize))
    }
}
