//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{
    common::{arc_real_fn, ConvergedReason, CountingRealFn, IterData, Returns},
    constrained::{ConstrainedError, ConstrainedMinimizer, ConstriainedOptions},
    unconstrained::{minimize, UnconstrainedMethod},
    Matrix, RealFn, RealVectorFn, Vector,
};
use core::f64;
//}}}
//{{{ std imports
use std::sync::{Arc, Mutex};
//}}}
//{{{ dep imports
use topohedral_linalg::{dvector::VecType::Col, MatrixOps, ReduceOps, VectorOps};
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------
//{{{ constants
const MIN_RTOL: f64 = 1e-5;
const MAX_RTOL: f64 = 1e-3;
//}}}
//{{{ struct: GradientDiagnostics
#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
struct AugLagDiagnostics
{
    grad_f_norm: f64,
    grad_p_norm: f64,
    grad_l_norm: f64,
    grad_cancellation_ratio: f64,
    objective_penalty_cosine: f64,
    penalty_to_objective_ratio: f64,
    num_active_ieq_constraints: usize,
    max_ieq_activation: f64,
    max_weighted_ieq_activation: f64,
}
//}}}
//{{{ impl: GradientDiagnostics
impl AugLagDiagnostics
{
    fn new() -> Self
    {
        AugLagDiagnostics {
            grad_f_norm: 0.0,
            grad_p_norm: 0.0,
            grad_l_norm: 0.0,
            grad_cancellation_ratio: 0.0,
            objective_penalty_cosine: 0.0,
            penalty_to_objective_ratio: 0.0,
            num_active_ieq_constraints: 0,
            max_ieq_activation: 0.0,
            max_weighted_ieq_activation: 0.0,
        }
    }
}
//}}}
//{{{ struct Options
#[derive(Clone)]
pub struct Options
{
    pub constrained_opts: ConstriainedOptions,
    pub uncon_method: UnconstrainedMethod,
    pub initial_penalty: f64,
    pub constraint_improvement_factor: f64,
    pub penalty_growth_factor: f64,
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

    pub(crate) fn uncon_method_mut(&mut self) -> &mut UnconstrainedMethod
    {
        &mut self.uncon_method
    }

    pub(crate) fn uncon_method(&self) -> &UnconstrainedMethod
    {
        &self.uncon_method
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
    //{{{ fn: new
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
    //}}}
    //{{{ fn: update_values
    fn update_values(
        &mut self,
        x: &Vector,
    )
    {
        self.function.eval(x, &mut self.values);
    }
    //}}}
    //{{{ fn: update_gradients
    fn update_gradients(
        &mut self,
        x: &Vector,
    )
    {
        self.function.grad(x, &mut self.gradients);
    }
    //}}}
    //{{{ fun: eq_penalty_value
    fn eq_penalty_value(&self) -> f64
    {
        let n = self.function.dimension_range();
        let mut constraint_value = 0.0;
        for i in 0..n
        {
            let p_i = self.penalties[i];
            let theta_i = self.shifts[i];
            let h_i = self.values[i];
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
    fn eq_penalty_gradient(&self) -> Vector
    {
        let dim = self.function.dimension_domain();
        let n = self.function.dimension_range();
        let mut constraint_gradient = Vector::zeros_cvec(dim, Col);

        for i in 0..n
        {
            let p_i = self.penalties[i];
            let theta_i = self.shifts[i];
            let h_i = self.values[i];
            let grad_h_i = self.gradients.col(i).to_dmatrix();
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
    fn ieq_penalty_value(&self) -> f64
    {
        let n = self.function.dimension_range();
        let mut constraint_value = 0.0;
        for i in 0..n
        {
            let q_i = self.penalties[i];
            let phi_i = self.shifts[i];
            let g_i = self.values[i];
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
    fn ieq_penalty_gradient(&self) -> Vector
    {
        let dim = self.function.dimension_domain();
        let n = self.function.dimension_range();
        let mut constraint_gradient = Vector::zeros_cvec(dim, Col);

        for i in 0..n
        {
            let q_i = self.penalties[i];
            let phi_i = self.shifts[i];
            let g_i = self.values[i];
            let grad_g_i = self.gradients.col(i).to_dmatrix();
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
}
//}}}
//{{{ struct CachedValues
#[derive(Debug, Clone)]
struct CachedValues
{
    fcn_value: f64,
    fcn_grad: Vector,
    eq_constraint_value: f64,
    eq_constriant_grad: Vector,
    ieq_constraint_value: f64,
    ieq_constraint_grad: Vector,
}
//}}}
//{{{ impl: CachedValues
impl CachedValues
{
    fn new(n: usize) -> Self
    {
        let zero_vector = Vector::zeros_cvec(n, Col);
        Self {
            fcn_value: 0.0,
            fcn_grad: zero_vector.clone(),
            eq_constraint_value: 0.0,
            eq_constriant_grad: zero_vector.clone(),
            ieq_constraint_value: 0.0,
            ieq_constraint_grad: zero_vector.clone(),
        }
    }
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
    cached_value: CachedValues,
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

        let n = fcn.dimension();
        Self {
            fcn,
            eq_constraint_data,
            ieq_constraint_data,
            unimproved_eq_constraints: Vec::with_capacity(num_eq_constraints),
            unimproved_ieq_constraints: Vec::with_capacity(num_ieq_constriants),
            cached_value: CachedValues::new(n),
        }
    }
    //}}}
    //{{{ fn: compute_max_penalty
    #[trace_fn]
    fn compute_max_penalty(&self) -> f64
    {
        let mut max_penalty = 0.0f64;

        if let Some(eq_constraint_data) = &self.eq_constraint_data
        {
            max_penalty = max_penalty.max(eq_constraint_data.penalties.max().unwrap());
        }

        if let Some(ieq_constraint_data) = &self.ieq_constraint_data
        {
            max_penalty = max_penalty.max(ieq_constraint_data.penalties.max().unwrap());
        }

        max_penalty
    }
    //}}}
    //{{{ fn: compute_max_constraint_violation
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
            trace!(target: "aug", "New shifts {}", topohedral_linalg::MatrixOps::transpose(&eq_constraint_data.shifts));
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
            trace!(target: "aug", "New shifts {}", topohedral_linalg::MatrixOps::transpose(&ieq_constriant_data.shifts));
            //}}}
        }
    }
    //}}}
    //{{{ fn: compute_diagnostics
    fn compute_diagnostics(&self) -> AugLagDiagnostics
    {
        let mut diag = AugLagDiagnostics::new();

        diag.grad_f_norm = self.cached_value.fcn_grad.norm();

        let penalty_grad: Vector =
            (&self.cached_value.ieq_constraint_grad + &self.cached_value.eq_constriant_grad).into();
        diag.grad_p_norm = penalty_grad.norm();

        let auglag_grad: Vector = (&self.cached_value.fcn_grad + &penalty_grad).into();
        diag.grad_l_norm = auglag_grad.norm();

        if diag.grad_f_norm * diag.grad_p_norm > 0.0
        {
            diag.objective_penalty_cosine = self.cached_value.fcn_grad.dot(&penalty_grad)
                / (diag.grad_f_norm * diag.grad_p_norm);
        }

        if diag.grad_l_norm > 0.0
        {
            let diff_vector: Vector = (&self.cached_value.fcn_grad - &penalty_grad).into();
            diag.grad_cancellation_ratio = diff_vector.norm() / diag.grad_l_norm;
        }

        if diag.grad_f_norm > 0.0
        {
            diag.penalty_to_objective_ratio = diag.grad_p_norm / diag.grad_f_norm;
        }

        if let Some(ieq_constraint_data) = &self.ieq_constraint_data
        {
            for ((&q_i, &g_i), &phi_i) in ieq_constraint_data
                .penalties
                .iter()
                .zip(ieq_constraint_data.values.iter())
                .zip(ieq_constraint_data.shifts.iter())
            {
                let activation_i = (g_i + phi_i).max(0.0);
                if activation_i > 0.0
                {
                    diag.num_active_ieq_constraints += 1;
                }
                diag.max_ieq_activation = diag.max_ieq_activation.max(activation_i);
                diag.max_weighted_ieq_activation =
                    diag.max_weighted_ieq_activation.max(q_i * activation_i);
            }
        }
        diag
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
        let cached_value = &mut self.cached_value;
        let fcn_value = self.fcn.eval(x);
        let mut eq_value = 0.0;
        let mut ieq_value = 0.0;

        cached_value.fcn_value = fcn_value;

        if let Some(eq_constraint_data) = &mut self.eq_constraint_data
        {
            eq_constraint_data.update_values(x);
            eq_value = eq_constraint_data.eq_penalty_value();
            cached_value.eq_constraint_value = eq_value;
        }

        if let Some(ieq_constraint_data) = &mut self.ieq_constraint_data
        {
            ieq_constraint_data.update_values(x);
            ieq_value = ieq_constraint_data.ieq_penalty_value();
            cached_value.ieq_constraint_value = eq_value;
        }

        let aug_lag_value = fcn_value + eq_value + ieq_value;
        aug_lag_value
    }
    //}}}
    //{{{ fn: grad
    #[trace_fn]
    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector
    {
        let n = self.dimension();

        let fcn_grad = self.fcn.grad(x);
        self.cached_value.fcn_grad = fcn_grad.clone();

        let mut eq_penalty_grad = Vector::zeros_cvec(n, Col);
        let mut ieq_penalty_grad = Vector::zeros_cvec(n, Col);

        if let Some(eq_constraint_data) = &mut self.eq_constraint_data
        {
            eq_constraint_data.update_gradients(x);
            eq_penalty_grad = eq_constraint_data.eq_penalty_gradient();
            self.cached_value.eq_constriant_grad = eq_penalty_grad.clone();
        }

        if let Some(ieq_constraint_data) = &mut self.ieq_constraint_data
        {
            ieq_constraint_data.update_gradients(x);
            ieq_penalty_grad = ieq_constraint_data.ieq_penalty_gradient();
            self.cached_value.ieq_constraint_grad = ieq_penalty_grad.clone();
        }

        let grad_aug_lag = (&fcn_grad + &eq_penalty_grad + &ieq_penalty_grad).into();
        return grad_aug_lag;
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
            opts,
        }
    }
    //}}}
    //{{{ fn: set_innter_rtol
    #[trace_fn]
    fn set_uncon_options(
        &mut self,
        max_violation_k: f64,
    )
    {
        let mut counting_fcn = self.fcn.lock().unwrap();
        let auglag_fcn = counting_fcn.inner_mut();
        let max_penalty_k = auglag_fcn.compute_max_penalty();
        let rk = max_violation_k.max(1.0 / max_penalty_k);
        let grad_rtol_k = (0.01 * rk).clamp(MIN_RTOL, MAX_RTOL);

        let max_iter_k = if grad_rtol_k > 1e-3
        {
            30
        }
        else if grad_rtol_k > 5e-4
        {
            50
        }
        else
        {
            100
        };

        self.opts.uncon_method.uncon_opts_mut().grad_atol = grad_rtol_k;
        self.opts.uncon_method.uncon_opts_mut().grad_rtol = grad_rtol_k;
        self.opts.uncon_method.uncon_opts_mut().max_iter = max_iter_k;
    }
    //}}}
    //{{{ fn: is_converged
    #[trace_fn]
    fn is_converged(&self) -> Option<ConvergedReason>
    {
        let mut counting_fcn = self.fcn.lock().unwrap();
        let auglag_fcn = counting_fcn.inner_mut();
        let d = auglag_fcn.compute_diagnostics();
        let max_constraint_violation = auglag_fcn.compute_max_constraint_violation();

        let rtol = self.opts.constrained_opts.grad_rtol;
        let atol = self.opts.constrained_opts.grad_atol;
        let ctol = self.opts.constrained_opts.constraint_tol;

        let normalized_grad = d.grad_l_norm / f64::max(1.0, d.grad_f_norm + d.grad_p_norm);
        let rtol_converged = normalized_grad < rtol;
        let atol_converged = d.grad_l_norm < atol;
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
        _k: u64,
        current_iter: &IterData,
        _current_max_violation: f64,
    )
    {
        let mut counting_fcn = self.fcn.lock().unwrap();
        let auglag_fcn = counting_fcn.inner_mut();
        let d = auglag_fcn.compute_diagnostics();
        //{{{ trace
        info!(target: "aug", "******************************************************************************************** i = {_k}");
        trace!(target: "aug", "Current solution: {}", current_iter.x.clone().transpose());
        if let Some(eq_constraint_data) = &auglag_fcn.eq_constraint_data
        {
            trace!(target: "aug", "Current EQ penalties: {}", eq_constraint_data.penalties.clone().transpose());
            trace!(target: "aug", "Current EQ shifts: {}", eq_constraint_data.shifts.clone().transpose());
        }

        if let Some(ieq_constraint_data) = &auglag_fcn.ieq_constraint_data
        {
            trace!(target: "aug", "Current IEQ penalties: {}", ieq_constraint_data.penalties.clone().transpose());
            trace!(target: "aug", "Current IEQ shifts: {}", ieq_constraint_data.shifts.clone().transpose());
        }

        info!(target: "aug", "Current values: {current_iter}");
        info!(target: "aug","Convergence measures:");
        let normalized_grad = d.grad_l_norm / f64::max(1.0, d.grad_f_norm + d.grad_p_norm);
        info!(target: "aug", "||∇L|| / max(1, ||∇F|| + ||∇P||) = {normalized_grad:1.4e}");
        info!(target: "aug", "Kmax = {_current_max_violation:1.4e}");
        info!(
            target: "aug",
            "Gradient split: ||∇f|| = {:1.4e} ||∇P|| = {:1.4e} ||∇L|| = {:1.4e}",
            d.grad_f_norm,
            d.grad_p_norm,
            d.grad_l_norm,
        );
        info!(
            target: "aug",
            "Gradient split:  penalty/objective = {:1.4e} cancellation = {:1.4e} cos(∇f,∇P) = {:1.4e}",
            d.penalty_to_objective_ratio,
            d.grad_cancellation_ratio,
            d.objective_penalty_cosine,
        );
        info!(
            target: "aug",
            "Ineq activity: active = {} max(g+phi)+ = {:1.4e} max(q(g+phi)+) = {:1.4e}",
            d.num_active_ieq_constraints,
            d.max_ieq_activation,
            d.max_weighted_ieq_activation,
        );
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
    fn minimize(&mut self) -> Result<Returns, ConstrainedError>
    {
        let n = self.x_init.len();
        let n_iter = self.opts.constrained_opts.max_iter;
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
            self.set_uncon_options(max_violation_k);

            iter_prev_k = iter_k;
            let ret = minimize(
                self.fcn.clone(),
                iter_prev_k.x.clone(),
                self.opts.uncon_method.clone(),
            )?;
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

            if let Some(reason) = self.is_converged()
            {
                //{{{ trace
                info!(target: "cg", "*********************************************");
                info!(target: "cg", "Converging with reason {reason:?}");
                info!(target: "cg","Convergence measures:");
                let _grad_ratio = iter_k.norm_grad_fx / self.norm_grad_fx_init;
                info!(target: "cg", "||∇L(k)|| / ||∇L(0)|| = {_grad_ratio:1.4e}");
                info!(target: "cg", "||∇L(k)|| = {:1.4e}", iter_k.norm_grad_fx);
                info!(target: "cg", "Kmax = {:1.4e}", max_violation_k);
                info!(target: "cg", "*********************************************");
                //}}}
                let fmin = self.fcn.lock().unwrap().inner_mut().fcn.eval(&iter_k.x);
                let xmin = iter_k.x;
                return Ok(Returns {
                    fmin,
                    xmin,
                    reason,
                    num_iterations: k as usize,
                    num_fun_evals: 0,
                    num_grad_evals: 0,
                });
            }
        }
        Err(ConstrainedError::MaxIterations(n_iter as usize))
    }
}
//}}}
