//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::Options as UnonstrainedOptions;
use super::common::{ConvergedReason, Error, Returns, UnconstrainedMinimizer};
use crate::common::IterData;
use crate::line_search as ls;
use crate::line_search::initial_step;
use crate::line_search::LineSearchFcn;
use crate::{RealFn, Vector};
//}}}
//{{{ std imports
use topohedral_linalg::MatrixOps;
//}}}
//{{{ dep imports
use topohedral_linalg::{dmatrix::DMatrix, dvector::DVector, dvector::VecType, MatMul, VectorOps};
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum: UpdateMethod
#[derive(Copy, Clone)]
pub enum UpdateMethod
{
    BFGS,
    DFP,
}
//}}}
//{{{ struct: Options
#[derive(Copy, Clone)]
pub struct Options
{
    pub uncon_opts: UnonstrainedOptions,
    pub method: UpdateMethod,
    pub restart: u64,
}
//}}}
//{{{ struct: Data
struct Data
{
    identity: DMatrix<f64>,
    mat1: DMatrix<f64>,
    mat2: DMatrix<f64>,
    mat3: DMatrix<f64>,
    sk: Vector,
    yk: Vector,
}
//}}}
//{{{ struct: QuasiNewton
pub struct QuasiNewton<F: RealFn>
{
    fcn: F,
    x_init: Vector,
    norm_grad_fx_init: f64,
    opts: Options,
    data: Data,
}
//}}}
//{{{ impl: QuasiNewton
impl<F: RealFn> QuasiNewton<F>
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
            norm_grad_fx_init: norm_grad_0,
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

    fn apply_restart(
        &self,
        k: u64,
        grad_fk: &Vector,
        dir_k: &mut Vector,
    )
    {
        let needs_restart = k % self.opts.restart == 0;
        let is_increasing = grad_fk.dot(dir_k) >= 0.0;
        if needs_restart || is_increasing
        {
            *dir_k = -grad_fk.clone();
        }
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

    #[trace_fn]
    fn update_hessian(
        &mut self,
        xk_prev: &Vector,
        xk: &Vector,
        grad_fk_prev: &Vector,
        grad_fk: &Vector,
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

                *sk = (xk - xk_prev).into();

                *yk = (grad_fk - grad_fk_prev).into();

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

    #[trace_fn]
    fn update_direction(
        &mut self,
        xk_prev: &Vector,
        xk: &Vector,
        grad_fk_prev: &Vector,
        grad_fk: &Vector,
        hess_k: &mut DMatrix<f64>,
    ) -> Vector
    {
        self.update_hessian(xk_prev, xk, grad_fk_prev, grad_fk, hess_k);
        let dir_k = -hess_k.matmul(grad_fk);
        dir_k
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
        info!(target: "cg", "||∇f(k)|| / ||∇f(0)|| = {grad_ratio:1.4e}");
        //}}}
    }
}
//}}}
//{{{ impl: UnconstrainedMinimizer for QuasiNewton
impl<F: RealFn> UnconstrainedMinimizer for QuasiNewton<F>
{
    #[trace_fn]
    fn minimize(&mut self) -> Result<Returns, Error>
    {
        let mut iter_k = IterData::new(self.fcn.clone(), &self.x_init);
        let mut iter_prev_k = iter_k.clone();
        iter_prev_k.fx = iter_k.fx + 0.5 * iter_k.norm_grad_fx;

        let mut dir_k = -iter_k.grad_fx.clone();
        let max_iter = self.opts.uncon_opts.max_iter;

        let n = iter_k.x.len();
        let mut hess_k = DMatrix::<f64>::identity(n, n);

        for k in 1..max_iter
        {
            self.print_status(k, &iter_k);
            self.apply_restart(k, &iter_k.grad_fx, &mut dir_k);

            let alpha_init =
                ls::initial_step(iter_k.fx, iter_prev_k.fx, iter_k.grad_fx.dot(&dir_k));

            iter_prev_k = iter_k;

            iter_k = ls::search(
                self.fcn.clone(),
                &iter_prev_k,
                &dir_k,
                alpha_init,
                self.opts.uncon_opts.ls_method,
            )?;

            dir_k = self.update_direction(
                &iter_prev_k.x,
                &iter_k.x,
                &iter_prev_k.grad_fx,
                &iter_k.grad_fx,
                &mut hess_k,
            );

            if let Some(reason) = self.is_converged(iter_k.norm_grad_fx)
            {
                //{{{ trace
                info!(target: "qn", "Converging with reason {reason:?}");
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
        let maxiter = self.opts.uncon_opts.max_iter;
        info!(target: "qn", "Did not converge within {maxiter} iterations");
        //}}}
        Err(Error::MaxIterations(self.opts.uncon_opts.max_iter as usize))
    }
}
//}}}
