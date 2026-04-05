//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{
    common::{arc_real_fn, ConvergedReason, CountingRealFn, IterData, Returns},
    constrained::{ConstrainedMinimizer, ConstriainedOptions},
    unconstrained::{minimize, UnconstrainedMethod},
    Matrix, RealFn, RealVectorFn, Vector,
};
use core::f64;
//}}}
//{{{ std imports
use std::sync::{Arc, Mutex};
//}}}
//{{{ dep imports
use topohedral_linalg::{
    dvector::VecType::{self, Col},
    MatrixOps, ReduceOps, VectorOps,
};
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------
//{{{ struct Options
#[derive(Copy, Clone)]
pub struct Options
{
    pub constrained_opts: ConstriainedOptions,
    uncon_method: UnconstrainedMethod,
    initial_penalty: f64,
    constraint_improvement_factor: f64,
    penalty_growth_factor: f64,
}
//}}}
//{{{ impl: Options
impl Options
{
    pub fn new(
        constrained_opts: ConstriainedOptions,
        uncon_method: UnconstrainedMethod,
        initial_penalty: f64,
        constraint_improvement_factor: f64,
        penalty_growth_factor: f64,
    ) -> Self
    {
        Self {
            constrained_opts,
            uncon_method,
            initial_penalty,
            constraint_improvement_factor,
            penalty_growth_factor,
        }
    }
}
//}}}
//{{{ struct: ConstraintData
#[derive(Debug, Clone)]
struct ConstraintData<F: RealVectorFn>
{
    pub function: F,
    pub penalties: Vector,
    pub shifts: Vector,
    pub values: Vector,
    pub gradients: Matrix,
}
//}}}
//{{{ impl: ConstraintData
impl<F: RealVectorFn> ConstraintData<F>
{
    pub fn new(
        fcn: F,
        initial_penalty: f64,
    ) -> Self
    {
        let num_constraints = fcn.dimension_range();
        let dimension = fcn.dimension_domain();
        //{{{ trace
        trace!(target: "aug", "Creating constraint data object with {num_constraints} constraints");
        //}}}

        let initial_penalties = Vector::from_value_vec(initial_penalty, num_constraints, Col);
        let zero_vector = Vector::zeros_cvec(num_constraints, Col);
        let zero_matrix = Matrix::zeros(dimension, num_constraints);

        Self {
            function: fcn,
            penalties: initial_penalties,
            shifts: zero_vector.clone(),
            values: zero_vector,
            gradients: zero_matrix,
        }
    }

    fn update_values(
        &mut self,
        x: &Vector,
    )
    {
        self.function.eval(x, &mut self.values);
    }

    fn update_gradients(
        &mut self,
        x: &Vector,
    )
    {
        self.function.grad(x, &mut self.gradients);
    }
}
//}}}
//{{{ fun: eq_penalty_value
fn eq_penalty_value<F: RealVectorFn>(constaint_data: &ConstraintData<F>) -> f64
{
    let n = constaint_data.function.dimension_range();
    let mut constraint_value = 0.0;
    for i in 0..n
    {
        let p_i = constaint_data.penalties[i];
        let theta_i = constaint_data.shifts[i];
        let h_i = constaint_data.values[i];
        constraint_value += p_i * (h_i + theta_i).powi(2);
    }
    constraint_value *= 0.5;
    //{{{ trace
    trace!(target: "aug", "Evaluated equality constraint value: {constraint_value:1.4e}");
    //}}}
    constraint_value
}
//}}}
//{{{ fun: eq_penalty_gradient
fn eq_penalty_gradient<F: RealVectorFn>(constaint_data: &ConstraintData<F>) -> Vector
{
    let dim = constaint_data.function.dimension_domain();
    let n = constaint_data.function.dimension_range();
    let mut constraint_gradient = Vector::zeros_cvec(dim, Col);

    for i in 0..n
    {
        let p_i = constaint_data.penalties[i];
        let theta_i = constaint_data.shifts[i];
        let h_i = constaint_data.values[i];
        let grad_h_i = constaint_data.gradients.col(i).to_dmatrix();
        constraint_gradient += (p_i * (h_i + theta_i)) * grad_h_i;
    }

    //{{{ trace
    trace!(
        target: "aug",
        "Evaluated equality constraint gradient norm: {:1.4e}",
        constraint_gradient.norm()
    );
    //}}}
    constraint_gradient
}
//}}}
//{{{ fun: ieq_penalty_value
fn ieq_penalty_value<F: RealVectorFn>(constaint_data: &ConstraintData<F>) -> f64
{
    let n = constaint_data.function.dimension_range();
    let mut constraint_value = 0.0;
    for i in 0..n
    {
        let q_i = constaint_data.penalties[i];
        let phi_i = constaint_data.shifts[i];
        let g_i = constaint_data.values[i];
        constraint_value += q_i * ((g_i + phi_i).max(0.0)).powi(2);
    }
    constraint_value *= 0.5;
    //{{{ trace
    trace!(target: "aug", "Evaluated inequality constraint value: {constraint_value:1.4e}");
    //}}}
    constraint_value
}
//}}}
//{{{ fun: ieq_penalty_gradient
fn ieq_penalty_gradient<F: RealVectorFn>(constaint_data: &ConstraintData<F>) -> Vector
{
    let dim = constaint_data.function.dimension_domain();
    let n = constaint_data.function.dimension_range();
    let mut constraint_gradient = Vector::zeros_cvec(dim, Col);

    for i in 0..n
    {
        let q_i = constaint_data.penalties[i];
        let phi_i = constaint_data.shifts[i];
        let g_i = constaint_data.values[i];
        let grad_g_i = constaint_data.gradients.col(i).to_dmatrix();
        constraint_gradient += (q_i * (g_i + phi_i).max(0.0)) * grad_g_i;
    }
    //{{{ trace
    trace!(
        target: "aug",
        "Evaluated inequality constraint gradient norm: {:1.4e}",
        constraint_gradient.norm()
    );
    //}}}
    constraint_gradient
}
//}}}
//{{{ struct: AugmentedLagrangianFcn
#[derive(Debug, Clone)]
pub struct AugmentedLagrangianFcn<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn>
{
    fcn: F1,
    eq_constraint_data: Option<ConstraintData<F2>>,
    ieq_constraint_data: Option<ConstraintData<F3>>,
    unimproved_eq_constraints: Vec<usize>,
    unimproved_ieq_constraints: Vec<usize>,
}
//}}}
//{{{ impl: AugmentedLagrangianFcn
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> AugmentedLagrangianFcn<F1, F2, F3>
{
    //{{{ fn: new
    #[trace_fn]
    pub fn new(
        fcn: F1,
        eq_constraints: Option<F2>,
        ieq_constraints: Option<F3>,
        initial_penalty: f64,
    ) -> Self
    {
        let mut num_eq_constraints = 0;
        let eq_constraint_data = match eq_constraints
        {
            Some(eq_con) =>
            {
                num_eq_constraints = eq_con.dimension_range();
                Some(ConstraintData::new(eq_con, initial_penalty))
            }
            None => None,
        };

        let mut num_ieq_constriants = 0;
        let ieq_constraint_data = match ieq_constraints
        {
            Some(ieq_con) =>
            {
                num_ieq_constriants = ieq_con.dimension_range();
                Some(ConstraintData::new(ieq_con, initial_penalty))
            }
            None => None,
        };

        Self {
            fcn: fcn,
            eq_constraint_data: eq_constraint_data,
            ieq_constraint_data: ieq_constraint_data,
            unimproved_eq_constraints: Vec::with_capacity(num_eq_constraints),
            unimproved_ieq_constraints: Vec::with_capacity(num_ieq_constriants),
        }
    }
    //}}}
    //{{{ fn: max_constraint_violation
    #[trace_fn]
    fn compute_max_constraint_violation(&self) -> f64
    {
        let mut max_violation = 0.0;

        if let Some(eq_constraint_data) = &self.eq_constraint_data
        {
            let max_eq_violation = eq_constraint_data.values.abs_max().unwrap();
            //{{{ trace
            trace!(target: "aug", "Max equality violation, is {max_eq_violation:1.4e}");
            //}}}
            max_violation = f64::max(max_violation, max_eq_violation);
        }

        if let Some(ieq_constraint_data) = &self.ieq_constraint_data
        {
            let values = &ieq_constraint_data.values;
            let shifts = &ieq_constraint_data.shifts;
            //{{{ trace
            trace!(target: "aug", "Computing inequality constraint value");
            trace!(target: "aug", "values: {values:?}");
            trace!(target: "aug", "shifts: {shifts:?}");
            //}}}
            let max_ieq_violation = values
                .iter()
                .zip(shifts.iter())
                .map(|(&gi, &phi_i)| f64::max(gi, -phi_i))
                .reduce(f64::max)
                .unwrap();

            max_violation = f64::max(max_violation, max_ieq_violation);
        }
        max_violation
    }
    //}}}
    //{{{ fn: unimproved_constraint_violations
    #[trace_fn]
    fn update_unimproved_constraint_violation_indices(
        &mut self,
        new_kmax: f64,
    )
    {
        if let Some(eq_constraint_data) = &self.eq_constraint_data
        {
            //{{{ trace
            trace!(target: "aug", "Updating set of unimproved equality constraints");
            //}}}
            self.unimproved_eq_constraints.clear();
            self.unimproved_eq_constraints.extend(
                eq_constraint_data
                    .values
                    .iter()
                    .enumerate()
                    .filter_map(|(i, h_i)| (h_i.abs() > new_kmax).then_some(i)),
            );
            //{{{ trace
            trace!(target: "aug", "New set of unimproved equality constraints {:?} ",
            self.unimproved_eq_constraints);
            //}}}
        }

        if let Some(ieq_constraint_data) = &self.ieq_constraint_data
        {
            //{{{ trace
            trace!(target: "aug", "Updating set of unimproved inequality constraints");
            //}}}
            self.unimproved_ieq_constraints.clear();
            let values = &ieq_constraint_data.values;
            let shifts = &ieq_constraint_data.shifts;
            self.unimproved_ieq_constraints.extend(
                values
                    .iter()
                    .zip(shifts)
                    .enumerate()
                    .filter_map(|(i, (g_i, phi_i))| ((g_i).max(-*phi_i) > new_kmax).then_some(i)),
            );
            //{{{ trace
            trace!(target: "aug", "New set of unimproved inequality constraints {:?} ",
                    self.unimproved_ieq_constraints);
            //}}}
        }
    }
    //}}}
    //{{{ fn: increase_penalties
    #[trace_fn]
    fn increase_penalties(
        &mut self,
        penalty_increase_factor: f64,
    )
    {
        if let Some(eq_constraint_data) = &mut self.eq_constraint_data
        {
            //{{{ trace
            trace!(target: "aug", "Increasing equality penalties");
            //}}}
            for constraint_index in &self.unimproved_eq_constraints
            {
                eq_constraint_data.penalties[*constraint_index] *= penalty_increase_factor;
                eq_constraint_data.shifts[*constraint_index] /= penalty_increase_factor;
                //{{{ trace
                trace!(target: "aug", "Constraint index {}, new penalty {:1.4e} new_shift {:1.4e}",
                            *constraint_index,
                            eq_constraint_data.penalties[*constraint_index],
                            eq_constraint_data.shifts[*constraint_index]);
                //}}}
            }
        }
        if let Some(ieq_constraint_data) = &mut self.ieq_constraint_data
        {
            //{{{ trace
            trace!(target: "aug", "Increasing equality penalties");
            //}}}
            for constraint_index in &self.unimproved_ieq_constraints
            {
                ieq_constraint_data.penalties[*constraint_index] *= penalty_increase_factor;
                ieq_constraint_data.shifts[*constraint_index] /= penalty_increase_factor;

                //{{{ trace
                trace!(target: "aug", "Constraint index {}, new penalty {:1.4e} new_shift {:1.4e}",
                            *constraint_index,
                            ieq_constraint_data.penalties[*constraint_index],
                            ieq_constraint_data.shifts[*constraint_index]);
                //}}}
            }
        }
    }
    //}}}
    //{{{ fn: increase_shifts
    #[trace_fn]
    fn increase_shifts(&mut self)
    {
        if let Some(eq_constraint_data) = &mut self.eq_constraint_data
        {
            //{{{ trace
            trace!(target: "aug", "Increasing shifts on equality constraints");
            //}}}
            eq_constraint_data
                .shifts
                .iter_mut()
                .zip(eq_constraint_data.values.iter())
                .for_each(|(shift_i, hi)| {
                    *shift_i += hi;
                });

            //{{{ trace
            trace!(target: "aug", "New shifts {}", eq_constraint_data.shifts.transpose());
            //}}}
        }
        if let Some(ieq_constriant_data) = &mut self.ieq_constraint_data
        {
            //{{{ trace
            trace!(target: "aug", "Increasing shifts on inequality constraints");
            //}}}
            ieq_constriant_data
                .shifts
                .iter_mut()
                .zip(ieq_constriant_data.values.iter())
                .for_each(|(shift_i, gi)| {
                    let old_shift_i = *shift_i;
                    *shift_i = f64::max(0.0, old_shift_i + gi);
                });

            //{{{ trace
            trace!(target: "aug", "New shifts {}", ieq_constriant_data.shifts.transpose());
            //}}}
        }
    }
    //}}}
}

//}}}
//{{{ impl: RealFn for AugmentedLagrangianFcn
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> RealFn for AugmentedLagrangianFcn<F1, F2, F3>
{
    //{{{ fn: dimension
    fn dimension(&self) -> usize
    {
        self.fcn.dimension()
    }
    //}}}
    //{{{ fn: eval
    #[trace_fn]
    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        let mut aug_lag = self.fcn.eval(x);

        if let Some(eq_constraint_data) = &mut self.eq_constraint_data
        {
            eq_constraint_data.update_values(x);
            aug_lag += eq_penalty_value(eq_constraint_data);
        }

        if let Some(ieq_constraint_data) = &mut self.ieq_constraint_data
        {
            ieq_constraint_data.update_values(x);
            aug_lag += ieq_penalty_value(ieq_constraint_data);
        }

        aug_lag
    }
    //}}}
    //{{{ fn: grad
    #[trace_fn]
    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector
    {
        let mut grad_aug_lag = self.fcn.grad(x);

        if let Some(eq_constraint_data) = &mut self.eq_constraint_data
        {
            eq_constraint_data.update_gradients(x);
            grad_aug_lag += eq_penalty_gradient(eq_constraint_data);
        }

        if let Some(ieq_constraint_data) = &mut self.ieq_constraint_data
        {
            ieq_constraint_data.update_gradients(x);
            grad_aug_lag += ieq_penalty_gradient(ieq_constraint_data);
        }

        grad_aug_lag
    }
    //}}}
}
//}}}
//{{{ struct: AugmentedLagrangian
pub struct AugmentedLagrangian<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn>
{
    fcn: Arc<Mutex<CountingRealFn<AugmentedLagrangianFcn<F1, F2, F3>>>>,
    x_init: Vector,
    norm_grad_fx_init: f64,
    opts: Options,
}
//}}}
//{{{ impl: AugmentedLagrangian
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> AugmentedLagrangian<F1, F2, F3>
{
    //{{{ fn: new
    #[trace_fn]
    pub fn new(
        fcn: F1,
        eq_constraints: Option<F2>,
        ieq_constraints: Option<F3>,
        x0: Vector,
        opts: Options,
    ) -> Self
    {
        assert!(!opts.uncon_method.uncon_opts().make_counting);

        //{{{ trace
        trace!("Creating new Augmented Lagrangian function");
        //}}}
        let mut fcn_shared = arc_real_fn(CountingRealFn::new(AugmentedLagrangianFcn::new(
            fcn,
            eq_constraints,
            ieq_constraints,
            opts.initial_penalty,
        )));

        let _ = fcn_shared.eval(&x0);
        let norm_grad_f0 = fcn_shared.grad(&x0).norm();
        Self {
            fcn: fcn_shared,
            x_init: x0,
            norm_grad_fx_init: norm_grad_f0,
            opts: opts,
        }
    }
    //}}}
    //{{{ fn: set_innter_rtol
    #[trace_fn]
    fn set_inner_tol(
        &mut self,
        tol: f64,
    )
    {
        self.opts.uncon_method.uncon_opts_mut().grad_rtol = tol;
    }
    //}}}
    //{{{ fn: is_converged
    #[trace_fn]
    fn is_converged(
        &self,
        grad_norm: f64,
        max_constraint_violation: f64,
    ) -> Option<ConvergedReason>
    {
        //{{{ trace
        trace!(target: "aug",
        "grad_norm = {grad_norm:1.4e} max_constraint_violation = {max_constraint_violation:1.4e}");
        //}}}
        let rtol = self.opts.constrained_opts.grad_rtol;
        let atol = self.opts.constrained_opts.grad_atol;
        let ctol = self.opts.constrained_opts.constraint_tol;

        let rtol_converged = grad_norm / self.norm_grad_fx_init < rtol;
        let atol_converged = grad_norm < atol;
        let ctol_converged = max_constraint_violation < ctol;

        //{{{  trace
        trace!(target: "aug", "rtol_converged = {rtol_converged}");
        trace!(target: "aug", "atol_converged = {atol_converged}");
        trace!(target: "aug", "ctol_converged = {ctol_converged}");
        //}}}

        if rtol_converged && ctol_converged
        {
            return Some(ConvergedReason::Rtol);
        }

        if atol_converged && ctol_converged
        {
            return Some(ConvergedReason::Atol);
        }

        None
    }
    //}}}
    //{{{ fn:print
    fn print_status(
        &self,
        k: u64,
        current_iter: &IterData,
        current_max_violation: f64,
    )
    {
        //{{{ trace
        info!(target: "aug", "******************************************************************************************** i = {k}");
        info!(target: "aug", "Current values: {current_iter}");
        info!(target: "aug","Convergence measures:");
        let grad_ratio = current_iter.norm_grad_fx / self.norm_grad_fx_init;
        info!(target: "aug", "||∇f(k)|| / ||∇f(0)|| = {grad_ratio:1.4e}");
        info!(target: "aug", "Kmax = {current_max_violation:1.4e}");
        //}}}
    }
    //}}}
    //{{{ fn: update_penalties_shifts
    #[trace_fn]
    fn update_penalties_shifts(
        &mut self,
        mut constraint_violation_max: f64,
    ) -> (f64, f64)
    {
        let alpha = self.opts.constraint_improvement_factor;
        let beta = self.opts.penalty_growth_factor;
        let mut counting_fcn = self.fcn.lock().unwrap();
        let auglag_fcn = counting_fcn.inner_mut();
        let max_violation_k = auglag_fcn.compute_max_constraint_violation();

        //{{{ trace
        trace!(target: "aug", "constraint_violation_max = {constraint_violation_max:1.4e}");
        trace!(target: "aug", "max_violation_k = {max_violation_k:1.4e}");
        //}}}

        auglag_fcn.update_unimproved_constraint_violation_indices(constraint_violation_max / alpha);

        if max_violation_k >= constraint_violation_max / alpha
        {
            //{{{ trace
            trace!(target: "aug", "Constraint did not improve, increasing penalties");
            //}}}
            auglag_fcn.increase_penalties(beta);
            constraint_violation_max = constraint_violation_max.min(max_violation_k);
        }
        else
        {
            //{{{ trace
            trace!(target: "aug", "Constraint improved");
            //}}}
            auglag_fcn.increase_shifts();
            constraint_violation_max = max_violation_k;
        }
        (max_violation_k, constraint_violation_max)
    }
    //}}}
}
//}}}
//{{{ impl: ConstrainedMinimizer for AugmentedLagrangian
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> ConstrainedMinimizer
    for AugmentedLagrangian<F1, F2, F3>
{
    #[trace_fn]
    fn minimize(&mut self) -> Result<crate::Returns, super::common::Error>
    {
        let n = self.x_init.len();
        let n_iter = self.opts.constrained_opts.max_iter;
        let mut inner_rtol = 1e-2;
        let mut constraint_violation_max = f64::INFINITY;

        let mut iter_k = IterData::new(self.fcn.clone(), &self.x_init);
        let mut iter_prev_k: IterData;

        let mut max_violation_k = self
            .fcn
            .lock()
            .unwrap()
            .inner_mut()
            .compute_max_constraint_violation();

        for k in 1..n_iter
        {
            self.print_status(k, &iter_k, max_violation_k);
            self.set_inner_tol(inner_rtol);

            iter_prev_k = iter_k;
            let ret = minimize(self.fcn.clone(), iter_prev_k.x.clone(), self.opts.uncon_method)?;
            iter_k = IterData {
                fx: ret.fmin,
                x: ret.xmin,
                grad_fx: Vector::zeros_cvec(n, Col),
                norm_grad_fx: 0.0,
            };

            (max_violation_k, constraint_violation_max) =
                self.update_penalties_shifts(constraint_violation_max);

            iter_k.grad_fx = self.fcn.grad(&iter_k.x);
            iter_k.norm_grad_fx = iter_k.grad_fx.norm();

            if let Some(reason) = self.is_converged(iter_k.norm_grad_fx, max_violation_k)
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

        todo!()
    }
}
//}}}
