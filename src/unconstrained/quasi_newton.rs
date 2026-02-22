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
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{Arc, Mutex};
//}}}
//{{{ dep imports
use topohedral_linalg::{dmatrix::DMatrix, VectorOps};
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

pub struct QuasiNewton<F: RealFn>
{
    fcn: Arc<Mutex<CountingRealFn<F>>>,
    x_init: F::Vector,
    grad_fx_init: F::Vector,
    opts: Options,
}

impl<F: RealFn> QuasiNewton<F>
where
    F::Vector: VectorOps<ScalarType = f64>
        + Add<Output = F::Vector>
        + Sub<Output = F::Vector>
        + Clone
        + fmt::Display,
    f64: Mul<F::Vector, Output = F::Vector>,
{
    #[trace_fn]
    pub fn new(
        mut fcn: F,
        x0: F::Vector,
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
        &self,
        xk_prev: F::Vector,
        xk: F::Vector,
        grad_fk_prev: F::Vector,
        grad_fk: F::Vector,
        identity: &DMatrix<f64>,
        hess_k: &mut DMatrix<f64>,
    )
    {
        match self.opts.method
        {
            UpdateMethod::BFGS =>
            {
                let sk = xk - xk_prev;
                let yk = grad_fk - grad_fk_prev;
            }
            UpdateMethod::DFP =>
            {
                todo!()
            }
        }
    }
}

impl<F: RealFn> UnconstrainedMinimizer for QuasiNewton<F>
where
    F::Vector: VectorOps<ScalarType = f64>
        + Add<Output = F::Vector>
        + Sub<Output = F::Vector>
        + Neg<Output = F::Vector>
        + Div<Output = F::Vector>
        + Clone
        + fmt::Display,
    f64: Mul<F::Vector, Output = F::Vector>,
{
    type Vector = F::Vector;

    #[trace_fn]
    fn minimize(&mut self) -> Result<Returns<Self::Vector>, Error>
    {
        let mut xk = self.x_init.clone();
        let mut xk_prev = self.x_init.clone();
        let mut grad_fk = self.fcn.grad(&xk);
        let mut grad_fk_prev: Self::Vector;
        let mut grad_fk_norm: f64 = grad_fk.norm();
        let mut grad_fk_prev_norm: f64;
        let mut fk: f64 = self.fcn.eval(&xk);
        let fk_prev_offset: f64 = <f64 as Mul>::mul(0.5, grad_fk_norm);
        let mut fk_prev = fk + fk_prev_offset;
        let mut direction = -grad_fk.clone();
        let mut xk = self.x_init.clone();

        let n = xk.len();

        let identity = DMatrix::<f64>::identity(n, n);
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
            info!(target: "cg", "======================================================================== i = {i}");
            info!(target: "cg", "Current values fk = {fk:1.4e} grad_fk_norm = {grad_fk_norm:1.4e}");
            info!(target: "cg","Convergence measures:");
            info!(target: "cg", "\t||∇f(k)|| / ||∇f(0)|| = {:1.4e} ", grad_fk_norm / grad_fx_norm_init);
            info!(target: "cg", "\t||x(k) - x(k-1)|| = {:1.4e}", (xk.clone() - xk_prev.clone()).norm());
            //}}}
            let mut dphi0 = grad_fk.dot(&direction);
            let needs_restart = i % self.opts.restart == 0;
            let not_decreaseing = dphi0 >= 0.0;
            if needs_restart || not_decreaseing
            {
                //{{{ trace
                info!(target: "cg", "\tDoing restart for reasons:  restart? {needs_restart} descent direction? {not_decreaseing}");
                //}}}
                direction = -grad_fk.clone();
                dphi0 = grad_fk.dot(&direction);
            }

            let line_search_fcn =
                LineSearchFcn::new(self.fcn.clone(), xk.clone(), direction.clone());
            line_searcher.update_fcn(line_search_fcn);

            let phi0 = fk;
            let old_phi0 = fk_prev;
            let alpha1: f64 = initial_step(phi0, old_phi0, dphi0);
            let ls_ret = line_searcher.search(phi0, dphi0, alpha1)?;
            xk_prev = xk.clone();
            xk = xk + ls_ret.alpha * direction.clone();
            fk_prev = fk;
            fk = ls_ret.phi_alpha;
            grad_fk_prev = grad_fk.clone();
            grad_fk = self.fcn.grad(&xk);
            grad_fk_prev_norm = grad_fk_prev.norm();
            grad_fk_norm = grad_fk.norm();
        }

        todo!()
    }
}
