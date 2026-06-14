//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{
    common::{arc_real_fn, ConvergedReason, CountingRealFn, IterData, Returns},
    constrained::{ConstrainedError, ConstriainedOptions},
    unconstrained::{minimize, UnconstrainedMethod, UnconstrainedReturns},
    Matrix, Minimizer, RealFn, RealVectorFn, Vector,
};
use core::f64;
//}}}
//{{{ std imports
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
//}}}
//{{{ dep imports
use topohedral_linalg::{
    FloatTransformOps, MatMul, MatrixOps, ReduceOps, TransformOps, VecType::Col, VectorOps,
};
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum: LagrangianType
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LagrangianType
{
    AugmentedLagrangian,
    Lagrangian,
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
    #[trace_fn]
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

    #[trace_fn]
    pub(crate) fn uncon_method_mut(&mut self) -> &mut UnconstrainedMethod
    {
        &mut self.uncon_method
    }

    #[trace_fn]
    pub(crate) fn uncon_method(&self) -> &UnconstrainedMethod
    {
        &self.uncon_method
    }
}
//}}}

//{{{ struct: ConstraintData
#[derive(Debug, Clone)]
struct LagrangianPenaltyData<F: RealVectorFn>
{
    pub function: F,
    pub penalties: Vector,
    pub shifts: Vector,
    pub values: Vector,
    pub gradients: Matrix,
    pub max_violations: Vec<(f64, bool)>,
}
//}}}
//{{{ impl: ConstraintData
impl<F: RealVectorFn> LagrangianPenaltyData<F>
{
    //{{{ fn: new
    #[trace_fn]
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
        let zero_vector = Vector::zeros_vec(num_constraints, Col);
        let zero_matrix = Matrix::zeros(dimension, num_constraints);

        Self {
            function: fcn,
            penalties: initial_penalties,
            shifts: zero_vector.clone(),
            values: zero_vector,
            gradients: zero_matrix,
            max_violations: vec![(0.0, false); num_constraints],
        }
    }
    //}}}
    //{{{ fn: update_values
    #[trace_fn]
    fn update_values(
        &mut self,
        x: &Vector,
    )
    {
        self.function.eval(x, &mut self.values);
    }
    //}}}
    //{{{ fn: update_gradients
    #[trace_fn]
    fn update_gradients(
        &mut self,
        x: &Vector,
    )
    {
        self.function.grad(x, &mut self.gradients);
    }
    //}}}
    //{{{ fn: update_max_violations
    #[trace_fn]
    fn update_max_violations(
        &mut self,
        improvement_factor: f64,
        is_ineq: bool,
    )
    {
        let current_max_violations = &mut self.max_violations;
        let current_values = &self.values;

        //{{{ trace
        trace!(target: "aug", "current_max_violations = {:?}", current_max_violations);
        trace!(target: "aug", "current_values= {}", current_values.clone().transpose());
        //}}}

        for (i, max_violation, was_violated, constraint_value_i) in current_max_violations
            .iter_mut()
            .zip(current_values.iter())
            .enumerate()
            .map(|(i, ((max_violation, was_violated), hi))| (i, max_violation, was_violated, hi))
        {
            let max_violation_val = *max_violation;
            let constraint_val_tmp = if is_ineq
            {
                constraint_value_i.max(0.0)
            }
            else
            {
                constraint_value_i.abs()
            };
            *was_violated = constraint_val_tmp > improvement_factor * max_violation_val;
            if *was_violated
            {
                *max_violation = constraint_value_i.abs()
            }
            //{{{ trace
            info!(target: "aug", "i = {} was_violated = {} max_violation = {:.4e}",
               i, was_violated, max_violation);
            //}}}
        }
    }
    //}}}
    //{{{ fn: compute_lagrange_multiplier_estimates
    #[trace_fn]
    fn compute_lagrange_multiplier_estimates(&self) -> Vector
    {
        (&self.penalties * &self.shifts).into()
    }
    //}}}
}
//}}}

//{{{ struct: EqPenalty
#[derive(Debug, Clone)]
struct EqPenalty<F: RealVectorFn>
{
    data: LagrangianPenaltyData<F>,
    lagrangian_type: LagrangianType,
}
//}}}
//{{{ impl EqPenalty
impl<F: RealVectorFn> EqPenalty<F>
{
    //{{{ fn: new
    #[trace_fn]
    fn new(
        fcn: F,
        initial_penalty: f64,
        lagrangian_type: LagrangianType,
    ) -> Self
    {
        Self {
            data: LagrangianPenaltyData::new(fcn, initial_penalty),
            lagrangian_type,
        }
    }
    //}}}
    //{{{ fn: update_penalties_shifts
    #[trace_fn]
    fn update_penalties_shifts(
        &mut self,
        improvement_factor: f64,
        penalty_increase_factor: f64,
    )
    {
        self.data.update_max_violations(improvement_factor, false);

        for (i, (_, was_violated)) in self.data.max_violations.iter().enumerate()
        {
            if *was_violated
            {
                self.data.penalties[i] *= penalty_increase_factor;
                self.data.shifts[i] /= penalty_increase_factor;
            }
            else
            {
                let hi = self.data.values[i];
                self.data.shifts[i] += hi;
            }
        }
    }
    //}}}
}
//}}}
//{{{ impl RealFn for EqPenalty
impl<F: RealVectorFn> RealFn for EqPenalty<F>
{
    //{{{ fn: dimension
    #[trace_fn]
    fn dimension(&self) -> usize
    {
        self.data.function.dimension_domain()
    }
    //}}}
    //{{{ fn: eval
    #[trace_fn]
    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        self.data.update_values(x);
        let constraint_value = match self.lagrangian_type
        {
            LagrangianType::AugmentedLagrangian =>
            {
                //{{{ trace
                trace!(target: "aug", "Computing augmented lagrangian");
                //}}}

                let mut shifted_h: Vector = (&self.data.values + &self.data.shifts).into();
                shifted_h.transform(|hi| hi.powi(2));
                0.5 * self.data.penalties.dot(&shifted_h)
            }
            LagrangianType::Lagrangian =>
            {
                //{{{ trace
                trace!(target: "aug", "Computing classical lagrangian");
                //}}}
                let lambda = self.data.compute_lagrange_multiplier_estimates();
                lambda.dot(&self.data.values)
            }
        };
        //{{{ trace
        trace!(target: "aug", "Evaluated equality constraint value: {constraint_value:.4e}");
        //}}}
        constraint_value
    }
    //}}}
    //{{{ fn: grad
    #[trace_fn]
    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector
    {
        self.data.update_gradients(x);
        let weighted_constraint_values: Vector = match self.lagrangian_type
        {
            LagrangianType::AugmentedLagrangian =>
            {
                (&self.data.penalties * (&self.data.values + &self.data.shifts)).into()
            }
            LagrangianType::Lagrangian => self.data.compute_lagrange_multiplier_estimates(),
        };
        self.data.gradients.matmul(&weighted_constraint_values)
    }
    //}}}
}
//}}}

//{{{ struct: IeqPenalty
#[derive(Debug, Clone)]
struct IeqPenalty<F: RealVectorFn>
{
    data: LagrangianPenaltyData<F>,
    lagrangian_type: LagrangianType,
}
//}}}
//{{{ impl IeqPenalty
impl<F: RealVectorFn> IeqPenalty<F>
{
    //{{{ fn: new
    #[trace_fn]
    fn new(
        fcn: F,
        initial_penalty: f64,
        lagrangian_type: LagrangianType,
    ) -> Self
    {
        Self {
            data: LagrangianPenaltyData::new(fcn, initial_penalty),
            lagrangian_type,
        }
    }
    //}}}
    //{{{ fn: update_penalties_shifts
    #[trace_fn]
    fn update_penalties_shifts(
        &mut self,
        improvement_factor: f64,
        penalty_increase_factor: f64,
    )
    {
        self.data.update_max_violations(improvement_factor, true);

        for (i, (_, was_violated)) in self.data.max_violations.iter().enumerate()
        {
            //{{{ trace
            debug!(target: "aug", "i = {i} was_violated = {was_violated}");
            //}}}
            if *was_violated
            {
                //{{{ trace
                debug!(target: "aug", "i = {i} Constraint not improved, increasing penalties");
                //}}}
                self.data.penalties[i] *= penalty_increase_factor;
                self.data.shifts[i] /= penalty_increase_factor;
            }
            else
            {
                let gi = self.data.values[i];
                let old_shift = self.data.shifts[i];
                self.data.shifts[i] = f64::max(0.0, old_shift + gi);

                //{{{ trace
                debug!(target: "aug", "i = {i} Constraint improved, updating shifts: old_shift = {old_shift:.4e} new_shift = {:.4e}",
                            self.data.shifts[i]);
                //}}}
            }
        }
    }
    //}}}
}
//}}}
//{{{ impl RealFn  for IeqPenalty
impl<F: RealVectorFn> RealFn for IeqPenalty<F>
{
    //{{{ fn: dimension
    #[trace_fn]
    fn dimension(&self) -> usize
    {
        self.data.function.dimension_domain()
    }
    //}}}
    //{{{ fn: eval
    #[trace_fn]
    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        self.data.update_values(x);
        match self.lagrangian_type
        {
            LagrangianType::AugmentedLagrangian =>
            {
                let mut shifted_g: Vector = (&self.data.values + &self.data.shifts).into();
                shifted_g.transform(|value| value.max(0.0).powi(2));
                0.5 * self.data.penalties.dot(&shifted_g)
            }
            LagrangianType::Lagrangian =>
            {
                let mu = self.data.compute_lagrange_multiplier_estimates();
                mu.dot(&self.data.values)
            }
        }
    }
    //}}}
    //{{{ fn: grad
    #[trace_fn]
    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector
    {
        self.data.update_gradients(x);
        let weighted_constraint_values: Vector = match self.lagrangian_type
        {
            LagrangianType::AugmentedLagrangian =>
            {
                let mut shifted_values: Vector = (&self.data.values + &self.data.shifts).into();
                shifted_values.pos();
                (&self.data.penalties * &shifted_values).into()
            }
            LagrangianType::Lagrangian => self.data.compute_lagrange_multiplier_estimates(),
        };
        self.data.gradients.matmul(&weighted_constraint_values)
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
    all_constraint_value: f64,
    all_constraint_grad: Vector,
    auglag_value: f64,
    auglag_grad: Vector,
}
//}}}
//{{{ impl: CachedValues
impl CachedValues
{
    //{{{ fn: new
    #[trace_fn]
    fn new(n: usize) -> Self
    {
        let zero_vector = Vector::zeros_vec(n, Col);
        Self {
            fcn_value: 0.0,
            fcn_grad: zero_vector.clone(),
            eq_constraint_value: 0.0,
            eq_constriant_grad: zero_vector.clone(),
            ieq_constraint_value: 0.0,
            ieq_constraint_grad: zero_vector.clone(),
            all_constraint_value: 0.0,
            all_constraint_grad: zero_vector.clone(),
            auglag_value: 0.0,
            auglag_grad: zero_vector.clone(),
        }
    }
    //}}}
}
//}}}

//{{{ struct: AugmentedLagrangianFcn
#[derive(Debug, Clone)]
pub struct AugmentedLagrangianFcn<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn>
{
    fcn: F1,
    eq_penalty: Option<EqPenalty<F2>>,
    ieq_penalty: Option<IeqPenalty<F3>>,
    lagrangian_type: LagrangianType,
    cached_values: HashMap<LagrangianType, CachedValues>,
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
        lagrangian_type: LagrangianType,
    ) -> Self
    {
        let eq_penalty =
            eq_constraints.map(|eq_con| EqPenalty::new(eq_con, initial_penalty, lagrangian_type));

        let ieq_penalty = ieq_constraints
            .map(|ieq_con| IeqPenalty::new(ieq_con, initial_penalty, lagrangian_type));

        let n = fcn.dimension();
        let mut cached_values = HashMap::<LagrangianType, CachedValues>::new();
        cached_values.insert(LagrangianType::AugmentedLagrangian, CachedValues::new(n));
        cached_values.insert(LagrangianType::Lagrangian, CachedValues::new(n));

        Self {
            fcn,
            eq_penalty,
            ieq_penalty,
            lagrangian_type,
            cached_values,
        }
    }
    //}}}
    //{{{ fn: set_lagrangian_type
    #[trace_fn]
    fn set_lagrangian_type(
        &mut self,
        lagrangian_type: LagrangianType,
    )
    {
        if let Some(eq_penalty) = &mut self.eq_penalty
        {
            eq_penalty.lagrangian_type = lagrangian_type
        }

        if let Some(ieq_penalty) = &mut self.ieq_penalty
        {
            ieq_penalty.lagrangian_type = lagrangian_type
        }
    }
    //}}}
    //{{{ fn: update_max_eq_violations
    #[trace_fn]
    fn update_max_eq_violations(
        &self,
        max_eq_violations: &mut Vector,
    )
    {
        let eq_penalty = self.eq_penalty.as_ref().unwrap();

        for (max_violation, hi) in max_eq_violations
            .iter_mut()
            .zip(eq_penalty.data.values.iter())
        {
            *max_violation = max_violation.max(hi.abs());
        }
    }
    //}}}
    //{{{ fn: update_max_ieq_violations
    #[trace_fn]
    fn update_max_ieq_violations(
        &self,
        max_ieq_violations: &mut Vector,
    )
    {
        let ieq_penalty = self.ieq_penalty.as_ref().unwrap();

        for (max_violation, gi, shift_i) in max_ieq_violations
            .iter_mut()
            .zip(ieq_penalty.data.values.iter())
            .zip(ieq_penalty.data.shifts.iter())
            .map(|((max_violation, gi), phi_i)| (max_violation, gi, phi_i))
        {
            *max_violation = max_violation.max(((-gi).min(*shift_i)).abs());
        }
    }
    //}}}
    //{{{ fn: has_eq_penalty
    #[trace_fn]
    fn has_eq_penalty(&self) -> bool
    {
        self.eq_penalty.is_some()
    }
    //}}}
    //{{{ fn: has_ieq_penalty
    #[trace_fn]
    fn has_ieq_penalty(&self) -> bool
    {
        self.ieq_penalty.is_some()
    }
    //}}}
    //{{{ fn: get_cached_values
    #[trace_fn]
    fn get_cached_values(&self) -> &CachedValues
    {
        self.cached_values.get(&self.lagrangian_type).unwrap()
    }
    //}}}
    //{{{ fn: get_cached_values_mut
    #[trace_fn]
    fn get_cached_values_mut(&mut self) -> &mut CachedValues
    {
        self.cached_values.get_mut(&self.lagrangian_type).unwrap()
    }
    //}}}
}

//}}}
//{{{ impl: RealFn for AugmentedLagrangianFcn
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> RealFn for AugmentedLagrangianFcn<F1, F2, F3>
{
    //{{{ fn: dimension
    #[trace_fn]
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
        let fcn_value = self.fcn.eval(x);

        //{{{ trace
        info!(target: "aug", "Evaluated objective: {fcn_value:.4e}");
        //}}}

        let eq_value = if let Some(eq_penalty) = &mut self.eq_penalty
        {
            let eq_value = eq_penalty.eval(x);
            //{{{ trace
            debug!(target: "aug", "Evaluated EQ penalty: {eq_value:.4e}");
            //}}}
            eq_value
        }
        else
        {
            0.0
        };

        let ieq_value = if let Some(ieq_penalty) = &mut self.ieq_penalty
        {
            let ieq_value = ieq_penalty.eval(x);
            //{{{ trace
            debug!(target: "aug", "Evaluated IEQ penalty: {ieq_value:.4e}");
            //}}}
            ieq_value
        }
        else
        {
            0.0
        };

        let auglag_value = fcn_value + eq_value + ieq_value;

        {
            let cached_value = self.get_cached_values_mut();
            cached_value.fcn_value = fcn_value;
            cached_value.eq_constraint_value = eq_value;
            cached_value.ieq_constraint_value = ieq_value;
            cached_value.all_constraint_value = eq_value + ieq_value;
            cached_value.auglag_value = auglag_value;
        }

        //{{{ trace
        debug!(target: "aug", "Augmented Lagrangian value: {auglag_value:.4e}");
        //}}}
        auglag_value
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

        //{{{ trace
        trace!(target: "aug", "Evaluated Objective Gradient = {}",
                    fcn_grad.clone().transpose());
        //}}}

        let eq_penalty_grad = if let Some(eq_constraint_data) = &mut self.eq_penalty
        {
            let eq_penalty_grad = eq_constraint_data.grad(x);
            //{{{ trace
            trace!(target: "aug", "Evaluated EQ Gradient = {}",
                    eq_penalty_grad.clone().transpose());
            //}}}
            eq_penalty_grad
        }
        else
        {
            Vector::zeros_vec(n, Col)
        };

        let ieq_penalty_grad = if let Some(ieq_constraint_data) = &mut self.ieq_penalty
        {
            let ieq_penalty_grad = ieq_constraint_data.grad(x);
            //{{{ trace
            trace!(target: "aug", "Evaluated IEQ Gradient = {}",
                    ieq_penalty_grad.clone().transpose());
            //}}}
            ieq_penalty_grad
        }
        else
        {
            Vector::zeros_vec(n, Col)
        };

        let grad_aug_lag: Vector = (&fcn_grad + &eq_penalty_grad + &ieq_penalty_grad).into();

        {
            let cached_value = self.get_cached_values_mut();
            cached_value.fcn_grad = fcn_grad;
            cached_value.eq_constriant_grad = eq_penalty_grad.clone();
            cached_value.ieq_constraint_grad = ieq_penalty_grad.clone();
            cached_value.all_constraint_grad = (&eq_penalty_grad + &ieq_penalty_grad).into();
            cached_value.auglag_grad = grad_aug_lag.clone();
        }

        //{{{ trace
        trace!(target: "aug", "Evaluated Augmented Lagrangian Gradient = {}",
                    grad_aug_lag.clone().transpose());
        //}}}
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
        trace!(target: "aug", "Creating new Augmented Lagrangian function");
        //}}}

        let mut fcn_shared = arc_real_fn(CountingRealFn::new(AugmentedLagrangianFcn::new(
            fcn,
            eq_constraints,
            ieq_constraints,
            opts.initial_penalty,
            LagrangianType::AugmentedLagrangian,
        )));

        let _ = fcn_shared.eval(&x0);
        let _ = fcn_shared.grad(&x0);

        Self {
            fcn: fcn_shared,
            x_init: x0,
            opts,
        }
    }
    //}}}
    //{{{ fn: set_innter_rtol
    #[trace_fn]
    fn set_uncon_options(&mut self) {}
    //}}}
    //{{{ fn: is_converged
    #[trace_fn]
    fn is_converged(
        &self,
        iter_k: &IterData,
    ) -> Option<ConvergedReason>
    {
        //{{{ trace
        info!(target: "aug", "Checking convergence");
        //}}}
        self.fcn.lock().unwrap().with_inner_mut(|fcn| {
            let mut classical_auglag = fcn.clone();
            classical_auglag.set_lagrangian_type(LagrangianType::Lagrangian);
            let _ = classical_auglag.eval(&iter_k.x);
            let _ = classical_auglag.grad(&iter_k.x);

            let norm_eq = if let Some(eq_penalty) = &classical_auglag.eq_penalty
            {
                eq_penalty.data.values.abs_max().unwrap()
            }
            else {
                0.0
            };
            let norm_ieq = if let Some(ieq_penalty) = &classical_auglag.ieq_penalty
            {
                ieq_penalty.data.values.posed().abs_max().unwrap()
            }
            else {
                0.0
            };

            let cached_values = classical_auglag.get_cached_values();
            let norm_grad_f = cached_values.fcn_grad.abs_max().unwrap();
            let norm_grad_penalty = cached_values.all_constraint_grad.abs_max().unwrap();
            let norm_grad_auglag = cached_values.auglag_grad.abs_max().unwrap();

            let residual_stationarity = norm_grad_auglag;
            let residual_stationarity_scaled = norm_grad_auglag / 1.0f64.max(norm_grad_f).max(norm_grad_penalty);
            let residual_primal = norm_eq.max(norm_ieq);
            //{{{ trace
            info!(target: "aug", "||∇P|| = {norm_grad_penalty:.4e} ||∇F|| = {norm_grad_f:.4} ||h|| = {norm_eq:.4e} ||g|| = {norm_ieq:.4e}");
            info!(target: "aug", "||∇L|| = {norm_grad_auglag:.4e})");
            info!(target: "aug", "||∇L|| / max(1, ||∇F||, ||∇P||) = {residual_stationarity_scaled:.4e}");
            //}}}
            let ctol = self.opts.constrained_opts.constraint_tol;
            let rtol = self.opts.constrained_opts.base_opts.grad_rtol;
            let atol = self.opts.constrained_opts.base_opts.grad_atol;
            let constraints_satsifed = residual_primal < ctol;
            let stationarity_rtol_satisfied = residual_stationarity_scaled < rtol;
            let stationarity_atol_satisfied = residual_stationarity< atol;

            if constraints_satsifed && stationarity_rtol_satisfied
            {
                Some(ConvergedReason::Rtol)
            }
            else if constraints_satsifed && stationarity_atol_satisfied
            {
                Some(ConvergedReason::Atol)
            }
            else
            {
                None
            }
        })
    }
    //}}}
    //{{{ fn: print_status
    fn print_status(
        &self,
        k: u64,
        iter_k: &IterData,
    )
    {
        self.fcn.lock().unwrap().with_inner_mut(|fcn| {

            info!(target: "aug", "******************************************************************************************** k = {k}");
            trace!(target: "aug", "Current solution: {}", iter_k.x.clone().transpose());
            trace!(target: "aug", "Current gradient: {}", iter_k.grad_fx.clone().transpose());

            if let Some(eq_penalty) = &fcn.eq_penalty
            {
                trace!(target: "aug", "Current EQ penalties: {}", eq_penalty.data.penalties.clone().transpose());
                trace!(target: "aug", "Current EQ shifts: {}", eq_penalty.data.shifts.clone().transpose());
            }

            if let Some(ieq_penalty) = &fcn.ieq_penalty
            {
                trace!(target: "aug", "Current IEQ penalties: {}", ieq_penalty.data.penalties.clone().transpose());
                trace!(target: "aug", "Current IEQ shifts: {}", ieq_penalty.data.shifts.clone().transpose());
            }


            info!(target: "aug", "******************************************************************************************** k = {k}");
        })
    }
    //}}}
    //{{{ fn: update_penalties_shifts
    #[trace_fn]
    fn update_lagrangian(
        &mut self,
        constraint_improvement_factor: f64,
        penalty_increase_factor: f64,
        uncon_ret: UnconstrainedReturns,
    ) -> IterData
    {
        self.fcn.lock().unwrap().with_inner_mut(|fcn| {
            if let Some(eq_penalty) = &mut fcn.eq_penalty
            {
                //{{{ trace
                info!(target: "aug", "Updating penalties and shifts for EQ");
                //}}}
                eq_penalty.update_penalties_shifts(
                    constraint_improvement_factor,
                    penalty_increase_factor,
                );
            }

            if let Some(ieq_penalty) = &mut fcn.ieq_penalty
            {
                //{{{ trace
                info!(target: "aug", "Updating penalties and shifts for IEQ");
                //}}}
                ieq_penalty.update_penalties_shifts(
                    constraint_improvement_factor,
                    penalty_increase_factor,
                );
            }
        });
        IterData::new(self.fcn.clone(), &uncon_ret.xmin)
    }
    //}}}
    #[trace_fn]
    fn set_inner_tolerances(&self) -> UnconstrainedMethod
    {
        let mut uncon_method = self.opts.uncon_method.clone();
        uncon_method.uncon_opts_mut().grad_rtol = 0.0;

        self.fcn.lock().unwrap().with_inner_mut(|fcn| {
            let (norm_eq, max_penalty_eq) = if let Some(eq_penalty) = &fcn.eq_penalty
            {
                (
                    eq_penalty.data.values.abs_max().unwrap(),
                    eq_penalty.data.penalties.allmax().unwrap(),
                )
            }
            else
            {
                (0.0, 1.0)
            };
            let (norm_ieq, max_penalty_ieq) = if let Some(ieq_penalty) = &fcn.ieq_penalty
            {
                (
                    ieq_penalty.data.values.posed().abs_max().unwrap(),
                    ieq_penalty.data.penalties.allmax().unwrap(),
                )
            }
            else
            {
                (0.0, 1.0)
            };

            let residual_primal = norm_eq.max(norm_ieq);
            let penalty_max = max_penalty_eq.max(max_penalty_ieq);
            let atol_min = 1e-6;
            let atol_max = 1e-2;
            let atol = (0.1 * (residual_primal.max(1.0 / penalty_max)).powf(1.5))
                .clamp(atol_min, atol_max);
            //{{{ trace
            info!(target: "aug", "Setting innner atol to {atol:.4e}");
            //}}}
            uncon_method.uncon_opts_mut().grad_atol = atol;
        });

        uncon_method
    }
}
//}}}
//{{{ impl: Minimizer for AugmentedLagrangian
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> Minimizer for AugmentedLagrangian<F1, F2, F3>
{
    type Error = ConstrainedError;

    #[trace_fn]
    fn minimize(&mut self) -> Result<Returns, Self::Error>
    {
        let alpha = self.opts.constraint_improvement_factor;
        let beta = self.opts.penalty_growth_factor;
        let n_iter = self.opts.constrained_opts.base_opts.max_iter;
        let mut iter_k = IterData::new(self.fcn.clone(), &self.x_init);
        let mut iter_prev_k = iter_k.clone();

        for k in 1..n_iter
        {
            self.print_status(k, &iter_k);
            let uncon_method = self.set_inner_tolerances();
            let ret = minimize(self.fcn.clone(), iter_prev_k.x.clone(), uncon_method)?;
            iter_k = self.update_lagrangian(alpha, beta, ret);
            iter_prev_k.copy_from(&iter_k);
            if let Some(reason) = self.is_converged(&iter_k)
            {
                //{{{ trace
                info!(target: "cg", "*********************************************");
                info!(target: "cg", "Converging with reason {reason:?}");
                info!(target: "cg","Convergence measures:");
                info!(target: "cg", "||∇L(k)|| = {:.4e}", iter_k.norm_grad_fx);
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

        Err(ConstrainedError::MaxIterations(0_usize))
    }
}
//}}}
